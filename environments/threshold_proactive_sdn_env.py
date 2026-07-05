"""Proactive SDN controller-management environment.

The environment uses real Topology Zoo graph structure and either a supplied
switch-level traffic trace or a deterministic day/night synthetic generator.
One environment step represents one controller decision interval.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from utils.topology_loader import TopologyLoader


class ThresholdBasedProactiveSDN(gym.Env):
    """Master-controller environment with active and parked slave controllers."""

    def __init__(
        self,
        topology_name: str = "os3e",
        num_slave_controllers: int = 3,
        num_parked_controllers: int = 2,
        underload_threshold: float = 0.30,
        overload_threshold: float = 0.70,
        use_real_data: bool = True,
        traffic_data: Optional[np.ndarray] = None,
        traffic_update_interval: int = 1,
        max_steps: int = 1000,
        allow_topology_fallback: bool = False,
    ):
        super().__init__()
        if num_slave_controllers < 1:
            raise ValueError("At least one initially active slave controller is required")
        if num_parked_controllers < 0:
            raise ValueError("num_parked_controllers must be non-negative")
        if not 0 <= underload_threshold < overload_threshold <= 1:
            raise ValueError("Expected 0 <= underload_threshold < overload_threshold <= 1")
        if traffic_update_interval < 1:
            raise ValueError("traffic_update_interval must be at least 1")

        self.topology_name = self._normalize_topology_name(topology_name)
        self.num_slave_controllers = num_slave_controllers
        self.num_parked_controllers = num_parked_controllers
        self.total_controllers = 1 + num_slave_controllers + num_parked_controllers
        self.underload_threshold = underload_threshold
        self.overload_threshold = overload_threshold
        self.use_real_data = use_real_data
        self.traffic_update_interval = traffic_update_interval
        self.max_steps = max_steps
        self.allow_topology_fallback = allow_topology_fallback

        self.topology_loader = TopologyLoader()
        self.controller_host_nodes: List[int] = []
        self.topology_info = None
        self.topology_fallback_used = False
        self._load_topology()

        self.master_id = 0
        self.managed_controller_ids = list(range(1, self.total_controllers))
        self.initial_active_ids = self.managed_controller_ids[:num_slave_controllers]
        self.initial_parked_ids = self.managed_controller_ids[num_slave_controllers:]
        # Backward-compatible names used by older scripts.
        self.slave_ids = list(self.managed_controller_ids)
        self.parked_ids = list(self.initial_parked_ids)

        self.active_slaves = set(self.initial_active_ids)
        self.parked_slaves = set(self.initial_parked_ids)
        self.switch_to_controller: Dict[int, int] = {}
        self.controller_loads = {i: 0.0 for i in range(self.total_controllers)}
        self.current_switch_loads = np.zeros(self.num_switches, dtype=np.float32)

        self.energy_active = 100.0
        self.energy_parked = 10.0
        self.energy_master = 50.0

        self.traffic_data = self._validate_traffic_data(traffic_data)
        self.traffic_index = 0
        self.time_step = 0
        self.traffic_phase = 0
        self.last_successful_proactive_action: Optional[str] = None
        self.last_successful_proactive_step = -10_000

        self._initialize_mappings()
        self._setup_spaces()

    @staticmethod
    def _normalize_topology_name(topology_name: str) -> str:
        return (topology_name or "os3e").strip().lower()

    def _validate_traffic_data(self, traffic_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if traffic_data is None:
            return None
        data = np.asarray(traffic_data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("traffic_data must have shape [timesteps, switches]")
        if len(data) == 0:
            raise ValueError("traffic_data cannot be empty")
        if data.shape[1] != self.num_switches:
            raise ValueError(
                f"Traffic has {data.shape[1]} switches; topology has {self.num_switches}"
            )
        if not np.isfinite(data).all():
            raise ValueError("traffic_data contains non-finite values")
        return np.clip(data, 0.0, 1.0)

    def _load_topology(self) -> None:
        try:
            topology_info, delay_matrix, controller_hosts = (
                self.topology_loader.build_switch_to_controller_delay_matrix(
                    self.topology_name,
                    self.total_controllers,
                )
            )
            self.topology_info = topology_info
            self.topology = topology_info.graph.copy()
            self.num_switches = self.topology.number_of_nodes()
            self.distance_matrix = delay_matrix
            self.controller_host_nodes = controller_hosts
        except Exception:
            if not self.allow_topology_fallback:
                raise
            topology_sizes = {
                "gridnet": 9,
                "bellcanada": 48,
                "os3e": 34,
                "interoute": 110,
                "cogentco": 197,
            }
            self.num_switches = topology_sizes.get(self.topology_name, 34)
            self.topology = nx.star_graph(self.num_switches - 1)
            rng = np.random.default_rng(42)
            self.distance_matrix = rng.uniform(
                1, 20, (self.num_switches, self.total_controllers)
            ).astype(np.float32)
            self.controller_host_nodes = list(range(self.total_controllers))
            self.topology_fallback_used = True

    @property
    def evoke_base(self) -> int:
        return 1 + self.num_switches

    @property
    def park_base(self) -> int:
        return self.evoke_base + len(self.managed_controller_ids)

    def action_for_evoke(self, controller_id: int) -> int:
        return self.evoke_base + self.managed_controller_ids.index(controller_id)

    def action_for_park(self, controller_id: int) -> int:
        return self.park_base + self.managed_controller_ids.index(controller_id)

    def _setup_spaces(self) -> None:
        obs_size = self.total_controllers * 2 + self.num_switches
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        max_actions = 1 + self.num_switches + 2 * len(self.managed_controller_ids)
        self.action_space = spaces.Discrete(max_actions)

    def _initialize_mappings(self) -> None:
        if not self.active_slaves:
            self.active_slaves.add(self.initial_active_ids[0])
        self.switch_to_controller.clear()
        active_controllers = sorted(self.active_slaves)
        for switch_id in range(self.num_switches):
            self.switch_to_controller[switch_id] = min(
                active_controllers,
                key=lambda controller_id: self.distance_matrix[switch_id, controller_id],
            )

    def _get_observation(self) -> np.ndarray:
        loads = np.array(
            [self.controller_loads[i] for i in range(self.total_controllers)],
            dtype=np.float32,
        )
        active_status = np.zeros(self.total_controllers, dtype=np.float32)
        active_status[self.master_id] = 1.0
        for controller_id in self.active_slaves:
            active_status[controller_id] = 1.0
        assignments = np.array(
            [
                self.switch_to_controller[switch_id] / max(1, self.total_controllers - 1)
                for switch_id in range(self.num_switches)
            ],
            dtype=np.float32,
        )
        return np.concatenate([loads, active_status, assignments]).astype(np.float32)

    def _next_switch_loads(self) -> np.ndarray:
        if self.traffic_data is not None:
            row = self.traffic_data[self.traffic_index % len(self.traffic_data)]
            self.traffic_index = (self.traffic_index + 1) % len(self.traffic_data)
            return row.copy()

        hour = self.traffic_phase % 24
        if 9 <= hour <= 17:
            base_load = self.np_random.uniform(0.7, 1.0)
        elif 18 <= hour <= 22:
            base_load = self.np_random.uniform(0.4, 0.7)
        else:
            base_load = self.np_random.uniform(0.1, 0.4)
        return np.clip(
            base_load * self.np_random.uniform(0.8, 1.2, self.num_switches),
            0,
            1,
        ).astype(np.float32)

    def _simulate_traffic_variation(self) -> None:
        self.current_switch_loads = self._next_switch_loads()
        self._recompute_controller_loads()

    def _loads_for_mapping(self, mapping: Dict[int, int]) -> Dict[int, float]:
        loads = {i: 0.0 for i in range(self.total_controllers)}
        counts = {i: 0 for i in range(self.total_controllers)}
        for switch_id, controller_id in mapping.items():
            if controller_id not in self.active_slaves:
                continue
            loads[controller_id] += float(self.current_switch_loads[switch_id])
            counts[controller_id] += 1
        for controller_id in self.active_slaves:
            if counts[controller_id] > 0:
                loads[controller_id] = float(
                    np.clip(loads[controller_id] / counts[controller_id], 0, 1)
                )
        return loads

    def _recompute_controller_loads(self) -> None:
        self.controller_loads = self._loads_for_mapping(self.switch_to_controller)

    def _check_thresholds(self) -> Dict[str, List[int]]:
        overloaded, underloaded, normal = [], [], []
        for controller_id in sorted(self.active_slaves):
            load = self.controller_loads[controller_id]
            if load > self.overload_threshold:
                overloaded.append(controller_id)
            elif load < self.underload_threshold:
                underloaded.append(controller_id)
            else:
                normal.append(controller_id)
        return {"overloaded": overloaded, "underloaded": underloaded, "normal": normal}

    def get_valid_actions(self) -> List[int]:
        valid_actions = [0]
        thresholds = self._check_thresholds()

        overloaded = set(thresholds["overloaded"])
        if overloaded:
            valid_actions.extend(
                1 + switch_id
                for switch_id, controller_id in self.switch_to_controller.items()
                if controller_id in overloaded and self._can_migrate_switch(switch_id)
            )
            valid_actions.extend(
                self.action_for_evoke(controller_id)
                for controller_id in sorted(self.parked_slaves)
            )

        if len(self.active_slaves) > 1:
            valid_actions.extend(
                self.action_for_park(controller_id)
                for controller_id in thresholds["underloaded"]
                if self._can_park_controller(controller_id)
            )

        return sorted(set(valid_actions))

    def describe_action(self, action: int) -> str:
        if action == 0:
            return "noop"
        if 1 <= action <= self.num_switches:
            switch_id = action - 1
            return (
                f"migrate_switch_{switch_id}_from_ctrl_"
                f"{self.switch_to_controller.get(switch_id)}"
            )
        if self.evoke_base <= action < self.park_base:
            slot = action - self.evoke_base
            controller_id = self.managed_controller_ids[slot]
            return f"evoke_ctrl_{controller_id}"
        if self.park_base <= action < self.action_space.n:
            slot = action - self.park_base
            controller_id = self.managed_controller_ids[slot]
            return f"park_ctrl_{controller_id}"
        return f"unknown_action_{action}"

    def _migrate_switch(self, switch_id: int, target_controller: int) -> bool:
        if switch_id not in self.switch_to_controller:
            return False
        if target_controller not in self.active_slaves:
            return False
        source_controller = self.switch_to_controller[switch_id]
        if source_controller == target_controller:
            return False

        candidate_mapping = dict(self.switch_to_controller)
        candidate_mapping[switch_id] = target_controller
        candidate_loads = self._loads_for_mapping(candidate_mapping)
        if candidate_loads[target_controller] > self.overload_threshold:
            return False

        self.switch_to_controller = candidate_mapping
        self._recompute_controller_loads()
        return True

    def _can_migrate_switch(self, switch_id: int) -> bool:
        target = self._select_migration_target(switch_id)
        if target is None:
            return False
        candidate_mapping = dict(self.switch_to_controller)
        candidate_mapping[switch_id] = target
        return self._loads_for_mapping(candidate_mapping)[target] <= self.overload_threshold

    def _can_park_controller(self, controller_id: int) -> bool:
        if controller_id not in self.active_slaves or len(self.active_slaves) <= 1:
            return False
        if self.controller_loads[controller_id] >= self.underload_threshold:
            return False
        candidate_mapping = dict(self.switch_to_controller)
        other_active = sorted(self.active_slaves - {controller_id})
        for switch_id in sorted(
            (item for item, assigned in candidate_mapping.items() if assigned == controller_id),
            key=lambda item: self.current_switch_loads[item],
            reverse=True,
        ):
            target = min(
                other_active,
                key=lambda item: self._loads_for_mapping(candidate_mapping)[item],
            )
            candidate_mapping[switch_id] = target
        old_active = self.active_slaves
        self.active_slaves = self.active_slaves - {controller_id}
        candidate_loads = self._loads_for_mapping(candidate_mapping)
        self.active_slaves = old_active
        return all(
            candidate_loads[item] <= self.overload_threshold for item in other_active
        )

    def _evoke_controller(self, controller_id: int) -> bool:
        if controller_id not in self.parked_slaves:
            return False
        self.parked_slaves.remove(controller_id)
        self.active_slaves.add(controller_id)
        self._recompute_controller_loads()

        overloaded = self._check_thresholds()["overloaded"]
        if overloaded:
            source = max(overloaded, key=lambda item: self.controller_loads[item])
            candidates = [
                switch_id
                for switch_id, assigned in self.switch_to_controller.items()
                if assigned == source
            ]
            if candidates:
                switch_id = max(candidates, key=lambda item: self.current_switch_loads[item])
                self._migrate_switch(switch_id, controller_id)
        self._recompute_controller_loads()
        return True

    def _park_controller(self, controller_id: int) -> bool:
        if controller_id not in self.active_slaves or len(self.active_slaves) <= 1:
            return False
        if self.controller_loads[controller_id] >= self.underload_threshold:
            return False

        other_active = sorted(self.active_slaves - {controller_id})
        candidate_mapping = dict(self.switch_to_controller)
        switches = [
            switch_id
            for switch_id, assigned in candidate_mapping.items()
            if assigned == controller_id
        ]
        for switch_id in sorted(
            switches,
            key=lambda item: self.current_switch_loads[item],
            reverse=True,
        ):
            target = min(
                other_active,
                key=lambda item: self._loads_for_mapping(candidate_mapping)[item],
            )
            candidate_mapping[switch_id] = target

        old_active = set(self.active_slaves)
        self.active_slaves.remove(controller_id)
        candidate_loads = self._loads_for_mapping(candidate_mapping)
        if any(
            candidate_loads[item] > self.overload_threshold for item in self.active_slaves
        ):
            self.active_slaves = old_active
            return False

        self.switch_to_controller = candidate_mapping
        self.parked_slaves.add(controller_id)
        self._recompute_controller_loads()
        return True

    def _select_migration_target(self, switch_id: int) -> Optional[int]:
        source = self.switch_to_controller.get(switch_id)
        candidates = [item for item in self.active_slaves if item != source]
        if not candidates:
            return None
        return min(candidates, key=lambda item: self.controller_loads[item])

    def _execute_action(self, action: int) -> Tuple[str, bool]:
        thresholds = self._check_thresholds()
        if action == 0:
            return "noop", True
        if 1 <= action <= self.num_switches:
            switch_id = action - 1
            source = self.switch_to_controller.get(switch_id)
            if source not in thresholds["overloaded"]:
                return "migrate_failed", False
            target = self._select_migration_target(switch_id)
            success = target is not None and self._migrate_switch(switch_id, target)
            return ("migrate" if success else "migrate_failed"), success
        if self.evoke_base <= action < self.park_base:
            controller_id = self.managed_controller_ids[action - self.evoke_base]
            if not thresholds["overloaded"]:
                return "evoke_failed", False
            success = self._evoke_controller(controller_id)
            return ("evoke" if success else "evoke_failed"), success
        if self.park_base <= action < self.action_space.n:
            controller_id = self.managed_controller_ids[action - self.park_base]
            if controller_id not in thresholds["underloaded"]:
                return "park_failed", False
            success = self._park_controller(controller_id)
            return ("park" if success else "park_failed"), success
        return "unknown_failed", False

    def _calculate_latency(self) -> float:
        return float(
            np.mean(
                [
                    self.distance_matrix[switch_id, controller_id]
                    for switch_id, controller_id in self.switch_to_controller.items()
                ]
            )
        )

    def _calculate_worst_case_latency(self) -> float:
        return float(
            max(
                self.distance_matrix[switch_id, controller_id]
                for switch_id, controller_id in self.switch_to_controller.items()
            )
        )

    def _calculate_energy(self) -> float:
        return float(
            self.energy_master
            + len(self.active_slaves) * self.energy_active
            + len(self.parked_slaves) * self.energy_parked
        )

    def _calculate_load_variance(self) -> float:
        loads = [self.controller_loads[item] for item in self.active_slaves]
        return float(np.var(loads)) if loads else 0.0

    def _calculate_load_balance_index(self) -> float:
        """Paper-compatible standard deviation of switch counts per active controller."""
        if not self.active_slaves:
            return 0.0
        counts = [
            sum(1 for assigned in self.switch_to_controller.values() if assigned == item)
            for item in sorted(self.active_slaves)
        ]
        return float(np.std(counts))

    def _calculate_traffic_load_balance_index(self) -> float:
        if not self.active_slaves:
            return 0.0
        loads = np.array(
            [self.controller_loads[item] for item in sorted(self.active_slaves)],
            dtype=np.float32,
        )
        mean_load = float(loads.mean())
        return float(loads.std() / max(mean_load, 1e-6))

    def _calculate_reward(
        self,
        action_type: str,
        success: bool,
        context: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        context = context or {}
        pre_overload = bool(context.get("overloaded"))
        pre_underload = bool(context.get("underloaded"))
        pre_can_park = bool(context.get("can_park"))

        latency = self._calculate_latency()
        worst_latency = self._calculate_worst_case_latency()
        energy = self._calculate_energy()
        load_variance = self._calculate_load_variance()
        paper_load_balance = self._calculate_load_balance_index()
        traffic_load_balance = self._calculate_traffic_load_balance_index()

        latency_scale = max(float(np.max(self.distance_matrix)), 1e-6)
        latency_term = latency / latency_scale
        max_energy = self.energy_master + len(self.managed_controller_ids) * self.energy_active
        min_energy = (
            self.energy_master
            + self.energy_active
            + (len(self.managed_controller_ids) - 1) * self.energy_parked
        )
        energy_term = (energy - min_energy) / max(max_energy - min_energy, 1e-6)
        balance_term = min(traffic_load_balance, 2.0) / 2.0
        base_reward = -(
            0.35 * latency_term + 0.35 * energy_term + 0.30 * balance_term
        )

        action_adjustment = 0.0
        oscillation_penalty = 0.0
        if not success and action_type != "noop":
            action_adjustment -= 0.5
        elif action_type == "park":
            action_adjustment += 0.5 if pre_underload and pre_can_park else -0.25
        elif action_type == "evoke":
            action_adjustment += 0.5 if pre_overload else -0.25
        elif action_type == "migrate":
            action_adjustment += 0.25 if pre_overload else -0.10
        elif action_type == "noop":
            if pre_overload:
                action_adjustment -= 1.0
            if pre_underload and pre_can_park:
                action_adjustment -= 0.8

        if success and action_type in {"park", "evoke"}:
            recent_flip = (
                self.last_successful_proactive_action is not None
                and self.last_successful_proactive_action != action_type
                and (self.time_step - self.last_successful_proactive_step) <= 5
            )
            if recent_flip and not pre_overload:
                oscillation_penalty = 1.0
            self.last_successful_proactive_action = action_type
            self.last_successful_proactive_step = self.time_step

        reward = float(np.clip(base_reward + action_adjustment - oscillation_penalty, -2, 1))
        thresholds = self._check_thresholds()
        info = {
            "latency": latency,
            "cs_avg_latency": latency,
            "worst_case_latency": worst_latency,
            "cs_worst_latency": worst_latency,
            "energy": energy,
            "load_variance": load_variance,
            "load_balance_index": paper_load_balance,
            "paper_load_balance_std": paper_load_balance,
            "traffic_load_balance_index": traffic_load_balance,
            "active_controllers": len(self.active_slaves),
            "parked_controllers": len(self.parked_slaves),
            "action_type": action_type,
            "action_success": success,
            "overloaded_count": len(thresholds["overloaded"]),
            "underloaded_count": len(thresholds["underloaded"]),
            "pre_overloaded_count": len(context.get("overloaded", [])),
            "pre_underloaded_count": len(context.get("underloaded", [])),
            "oscillation_penalty": oscillation_penalty,
            "base_reward": base_reward,
            "action_adjustment": action_adjustment,
            "trace_index": self.traffic_index,
        }
        return reward, info

    def step(self, action: int):
        thresholds = self._check_thresholds()
        context = {
            "overloaded": list(thresholds["overloaded"]),
            "underloaded": list(thresholds["underloaded"]),
            "can_park": bool(thresholds["underloaded"] and len(self.active_slaves) > 1),
        }
        action_type, success = self._execute_action(int(action))
        reward, info = self._calculate_reward(action_type, success, context)

        self.time_step += 1
        terminated = False
        truncated = self.time_step >= self.max_steps
        if not truncated and self.time_step % self.traffic_update_interval == 0:
            self.traffic_phase = (self.traffic_phase + 1) % 24
            self._simulate_traffic_variation()
        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.active_slaves = set(self.initial_active_ids)
        self.parked_slaves = set(self.initial_parked_ids)
        self._initialize_mappings()
        self.time_step = 0
        self.traffic_phase = int(options.get("traffic_phase", 0)) % 24
        self.traffic_index = int(options.get("traffic_start_index", 0))
        self.last_successful_proactive_action = None
        self.last_successful_proactive_step = -10_000
        self._simulate_traffic_variation()
        return self._get_observation(), {
            "traffic_index": self.traffic_index,
            "topology_fallback_used": self.topology_fallback_used,
        }

    def render(self):
        thresholds = self._check_thresholds()
        print(
            f"Step {self.time_step} | Active={sorted(self.active_slaves)} | "
            f"Parked={sorted(self.parked_slaves)} | "
            f"Overloaded={thresholds['overloaded']} | "
            f"Underloaded={thresholds['underloaded']}"
        )
