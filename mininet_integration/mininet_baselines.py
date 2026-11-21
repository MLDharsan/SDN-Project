"""Policies compared in Mininet, plus the matching simulator reference run.

The four policies are exactly those of ``experiments/evaluate_fair_baselines.py``:

- ``rdqn``          trained Rainbow DQN with invalid-action masking
- ``threshold``     overload -> evoke (else migrate hottest switch);
                    underload -> park a controller if safe; otherwise noop
- ``random_valid``  uniform over currently valid actions (seeded)
- ``noop``          never acts

The action-selection functions are imported, not copied, so the Mininet
baselines cannot drift from the simulator baselines.

``run_simulator_reference`` replays the *same* trace window through the
ordinary ``env.step`` loop, giving a simulated result for the identical
window that is stored next to each emulated result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mininet_integration.policy_adapter import PolicySelector, make_simulator_env

POLICIES = ["rdqn", "threshold", "random_valid", "noop"]

STEP_METRIC_KEYS = [
    "cs_avg_latency",
    "cs_worst_latency",
    "energy",
    "load_balance_index",
    "traffic_load_balance_index",
    "load_variance",
    "active_controllers",
    "parked_controllers",
]


def summarize_decisions(records: List[Dict]) -> Dict:
    """Aggregate per-step simulator-style metrics using the fair-baseline definitions."""
    if not records:
        return {}
    info = [r["info"] for r in records]
    action_types = [r["action_type"] for r in records]
    successes = [r["success"] for r in records]

    def mean(key):
        return float(np.mean([float(i.get(key, 0.0)) for i in info]))

    return {
        "steps": len(records),
        "total_reward": float(np.sum([r["reward"] for r in records])),
        "cs_avg_latency": mean("cs_avg_latency"),
        "cs_worst_latency": float(np.max([float(i.get("cs_worst_latency", 0.0)) for i in info])),
        "avg_energy": mean("energy"),
        "load_balance_index": mean("load_balance_index"),
        "traffic_load_balance_index": mean("traffic_load_balance_index"),
        "avg_load_variance": mean("load_variance"),
        "parking_events": sum(1 for t, s in zip(action_types, successes) if t == "park" and s),
        "evoking_events": sum(1 for t, s in zip(action_types, successes) if t == "evoke" and s),
        "migration_events": sum(1 for t, s in zip(action_types, successes) if t == "migrate" and s),
        "noop_events": sum(1 for t in action_types if t == "noop"),
        "failed_actions": sum(1 for t, s in zip(action_types, successes) if not s and t != "noop"),
        "overload_step_fraction": float(np.mean([i.get("pre_overloaded_count", 0) > 0 for i in info])),
        "underload_step_fraction": float(np.mean([i.get("pre_underloaded_count", 0) > 0 for i in info])),
    }


def run_simulator_reference(
    *,
    topology: str,
    num_slave_controllers: int,
    num_parked_controllers: int,
    trace: np.ndarray,
    observation_mode: str,
    gru_model_path: Optional[Path],
    model_path: Optional[Path],
    policy: str,
    start_index: int,
    steps: int,
    seed: int,
) -> List[Dict]:
    """Plain simulator loop (env.step) over ``trace[start_index : start_index + steps]``."""
    env = make_simulator_env(
        topology, num_slave_controllers, num_parked_controllers, trace,
        observation_mode, gru_model_path, max_steps=steps,
    )
    selector = PolicySelector(policy, model_path, env, seed)
    state, _ = env.reset(seed=seed, options={"traffic_start_index": start_index})
    records = []
    for step in range(steps):
        action, raw = selector.select(state, env)
        name = env.describe_action(action)
        state, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": step,
                "trace_index": start_index + step,
                "action": action,
                "raw_action": raw,
                "action_name": name,
                "action_type": info["action_type"],
                "success": bool(info["action_success"]),
                "reward": float(reward),
                "info": {k: info.get(k) for k in STEP_METRIC_KEYS + [
                    "pre_overloaded_count", "pre_underloaded_count"]},
            }
        )
        if terminated or truncated:
            break
    env.close()
    return records
