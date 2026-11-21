"""Drive the *unchanged* simulator environment from live measurements.

``ShadowState`` wraps a normal ``ThresholdBasedProactiveSDN`` instance and
replaces only its traffic source: instead of reading the next trace row it is
fed the switch loads measured in Mininet. Everything else - observation
layout (46 / 86 values for OS3E), valid-action masking, migration-target
selection, park/evoke safety checks, the multi-objective reward and the
graph-based latency/energy metrics - is the simulator's own code, so the
emulated run is scored exactly like the simulated one.

After each decision the controller mapping/active-set diff is handed to
``action_executor`` which applies it to the real OVS switches.

No simulator file is modified; only public/private methods are *called*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from mininet_integration import _paths  # noqa: F401
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN
from experiments.evaluate_fair_baselines import (
    load_rainbow_model,
    select_random_valid_action,
    select_rdqn_action,
    select_threshold_action,
)
from forecasting.gru_traffic_predictor import GRUTrafficForecaster


def make_simulator_env(
    topology: str,
    num_slave_controllers: int,
    num_parked_controllers: int,
    trace: np.ndarray,
    observation_mode: str,
    gru_model_path: Optional[Path],
    max_steps: int,
) -> ThresholdBasedProactiveSDN:
    """Identical construction to ``evaluate_fair_baselines.make_env``."""
    predictor = (
        GRUTrafficForecaster(gru_model_path, device="cpu")
        if observation_mode == "predictive"
        else None
    )
    return ThresholdBasedProactiveSDN(
        topology_name=topology,
        num_slave_controllers=num_slave_controllers,
        num_parked_controllers=num_parked_controllers,
        traffic_data=trace,
        traffic_update_interval=1,
        max_steps=max_steps,
        observation_mode=observation_mode,
        traffic_predictor=predictor,
    )


@dataclass
class DecisionResult:
    action: int
    raw_action: Optional[int]
    action_name: str
    action_type: str
    success: bool
    reward: float
    info: Dict
    mapping_before: Dict[int, int]
    mapping_after: Dict[int, int]
    active_before: Set[int]
    active_after: Set[int]


class ShadowState:
    def __init__(self, env: ThresholdBasedProactiveSDN):
        self.env = env
        self.observation_mode = env.observation_mode

    @property
    def history_length(self) -> int:
        return int(self.env.traffic_history.maxlen or 1)

    @property
    def state_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.n)

    def reset(self, seed: int, start_index: int) -> None:
        self.env.reset(seed=seed, options={"traffic_start_index": start_index})
        # From here on, loads come only from measurements.
        self.env.traffic_history.clear()
        self.env.traffic_data = None
        self.env.current_switch_loads = np.zeros(self.env.num_switches, dtype=np.float32)
        self.env.predicted_switch_loads = np.zeros(self.env.num_switches, dtype=np.float32)
        self.env._recompute_controller_loads()

    @property
    def ready(self) -> bool:
        return len(self.env.traffic_history) == self.history_length

    def ingest(self, switch_loads: np.ndarray) -> Optional[np.ndarray]:
        """Same order as ``_simulate_traffic_variation``; returns the forecast if any."""
        loads = np.clip(np.asarray(switch_loads, dtype=np.float32), 0.0, 1.0)
        if loads.shape != (self.env.num_switches,):
            raise ValueError(f"Expected {self.env.num_switches} switch loads, got {loads.shape}")
        self.env.current_switch_loads = loads.copy()
        self.env.traffic_history.append(loads.copy())
        self.env._recompute_controller_loads()
        if self.observation_mode == "predictive":
            if not self.ready:
                return None
            self.env._update_prediction()
            return self.env.predicted_switch_loads.copy()
        self.env._update_prediction()
        return None

    def observation(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("History not filled yet; run warm-up intervals first")
        return self.env._get_observation()

    def apply(self, action: int, raw_action: Optional[int] = None) -> DecisionResult:
        """``env.step`` minus the traffic advance (traffic comes from Mininet)."""
        env = self.env
        mapping_before = dict(env.switch_to_controller)
        active_before = set(env.active_slaves)
        action_name = env.describe_action(int(action))

        current = env._check_thresholds()
        predicted = (
            env._check_predicted_thresholds()
            if env.observation_mode == "predictive"
            else {"overloaded": [], "underloaded": [], "normal": []}
        )
        decision = env._decision_thresholds()
        context = {
            "overloaded": list(current["overloaded"]),
            "underloaded": list(current["underloaded"]),
            "can_park": bool(current["underloaded"] and len(env.active_slaves) > 1),
            "current_overloaded": list(current["overloaded"]),
            "predicted_overloaded": list(predicted["overloaded"]),
            "decision_overloaded": list(decision["overloaded"]),
        }
        action_type, success = env._execute_action(int(action))
        reward, info = env._calculate_reward(action_type, success, context)
        env.time_step += 1
        return DecisionResult(
            action=int(action),
            raw_action=raw_action,
            action_name=action_name,
            action_type=action_type,
            success=bool(success),
            reward=float(reward),
            info=info,
            mapping_before=mapping_before,
            mapping_after=dict(env.switch_to_controller),
            active_before=active_before,
            active_after=set(env.active_slaves),
        )


class PolicySelector:
    """Chooses actions with the same functions as ``evaluate_fair_baselines``."""

    def __init__(self, policy: str, model_path: Optional[Path], env, seed: int):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.checkpoint = None
        if policy == "rdqn":
            if model_path is None or not Path(model_path).exists():
                raise FileNotFoundError(f"RDQN checkpoint not found: {model_path}")
            self.model, self.checkpoint = load_rainbow_model(str(model_path))
            state_dim = int(self.checkpoint.get("state_dim", -1))
            action_dim = int(self.checkpoint.get("action_dim", -1))
            if state_dim != env.observation_space.shape[0]:
                raise ValueError(
                    f"Checkpoint state_dim={state_dim} but {env.observation_mode} state has "
                    f"{env.observation_space.shape[0]} values"
                )
            if action_dim != env.action_space.n:
                raise ValueError(f"Checkpoint action_dim={action_dim} != {env.action_space.n}")
        elif policy not in {"threshold", "random_valid", "noop"}:
            raise ValueError(f"Unsupported policy: {policy}")

    def select(self, state: np.ndarray, env):
        if self.policy == "rdqn":
            raw, action = select_rdqn_action(self.model, state, env, mask_invalid_actions=True)
            return int(action), int(raw)
        if self.policy == "random_valid":
            return int(select_random_valid_action(env, self.rng)), None
        if self.policy == "threshold":
            return int(select_threshold_action(env)), None
        return 0, None
