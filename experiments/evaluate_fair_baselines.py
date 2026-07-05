#!/usr/bin/env python3
"""
Fair evaluation for RDQN and baseline policies on the current SDN environment.

This script evaluates every policy with the same:
- topology
- traffic trace
- reward logic
- episode count
- steps per episode
- metric collection

Policies:
- rdqn: trained Rainbow/RDQN checkpoint
- random_valid: random action sampled only from currently valid actions
- threshold: rule-based threshold policy using overload/underload signals
- noop: no-action policy, useful as a lower reference
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # Support both `python experiments/...` and module imports in tests.
    from experiments.evaluate_threshold_model import (  # noqa: E402
        load_rainbow_model,
        select_action as select_rdqn_action,
    )
except ModuleNotFoundError:
    from evaluate_threshold_model import (  # type: ignore[no-redef]  # noqa: E402
        load_rainbow_model,
        select_action as select_rdqn_action,
    )
from environments.threshold_proactive_sdn_env import ThresholdBasedProactiveSDN  # noqa: E402


TOPOLOGIES = ["gridnet", "bellcanada", "os3e", "interoute", "cogentco"]
POLICIES = ["rdqn", "random_valid", "threshold", "noop"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PolicyResult:
    topology: str
    policy: str
    episodes: int
    steps_per_episode: int
    traffic_path: str | None
    model_path: str | None
    avg_reward: float
    std_reward: float
    cs_avg_latency: float
    std_cs_avg_latency: float
    cs_worst_latency: float
    std_cs_worst_latency: float
    avg_energy: float
    std_energy: float
    load_balance_index: float
    std_load_balance_index: float
    traffic_load_balance_index: float
    avg_load_variance: float
    avg_parking_events: float
    avg_evoking_events: float
    avg_migration_events: float
    avg_noop_events: float
    avg_failed_actions: float
    overload_step_fraction: float
    underload_step_fraction: float
    overload_with_parked_capacity_fraction: float
    underload_with_parking_capacity_fraction: float
    total_steps: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def load_traffic_data(traffic_path: Path | None) -> np.ndarray | None:
    if traffic_path is None:
        return None
    if not traffic_path.exists():
        raise FileNotFoundError(f"Traffic trace not found: {traffic_path}")
    return np.load(traffic_path)


def make_env(topology: str, traffic_data: np.ndarray | None, max_steps: int):
    return ThresholdBasedProactiveSDN(
        topology_name=topology,
        num_slave_controllers=3,
        num_parked_controllers=2,
        traffic_data=traffic_data,
        traffic_update_interval=1,
        max_steps=max_steps,
    )


def select_random_valid_action(env, rng: np.random.Generator) -> int:
    valid_actions = env.get_valid_actions()
    return int(rng.choice(valid_actions))


def select_threshold_action(env) -> int:
    """Simple proactive threshold policy for the current action encoding."""
    thresholds = env._check_thresholds()

    if thresholds["overloaded"]:
        if env.parked_slaves:
            return env.action_for_evoke(min(env.parked_slaves))

        overloaded = set(thresholds["overloaded"])
        valid = set(env.get_valid_actions())
        candidate_switches = [
            switch_id
            for switch_id, controller_id in env.switch_to_controller.items()
            if controller_id in overloaded and 1 + switch_id in valid
        ]
        if candidate_switches:
            most_loaded_switch = max(
                candidate_switches,
                key=lambda switch_id: env.controller_loads[
                    env.switch_to_controller[switch_id]
                ],
            )
            return 1 + most_loaded_switch

    has_spare_active = len(env.active_slaves) > 1
    if thresholds["underloaded"] and has_spare_active:
        valid = set(env.get_valid_actions())
        candidates = [
            item for item in thresholds["underloaded"]
            if env.action_for_park(item) in valid
        ]
        if candidates:
            controller_id = min(candidates, key=lambda item: env.controller_loads[item])
            return env.action_for_park(controller_id)

    return 0


def summarize_episode(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def evaluate_policy(
    topology: str,
    policy: str,
    episodes: int,
    steps_per_episode: int,
    traffic_path: Path | None,
    model_path: Path | None,
    seed: int,
) -> PolicyResult:
    traffic_data = load_traffic_data(traffic_path)
    env = make_env(topology, traffic_data, steps_per_episode)
    rng = np.random.default_rng(seed)

    rdqn_model = None
    if policy == "rdqn":
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"RDQN checkpoint not found: {model_path}")
        rdqn_model, checkpoint = load_rainbow_model(str(model_path))
        if int(checkpoint.get("action_dim", -1)) != env.action_space.n:
            raise ValueError(
                f"Checkpoint action_dim={checkpoint.get('action_dim')} is incompatible with "
                f"environment action_dim={env.action_space.n}. Retrain this topology with "
                "train_rainbow_fixed_rewards.py after the stable action-schema fix."
            )

    episode_rewards = []
    episode_avg_latencies = []
    episode_worst_latencies = []
    episode_energies = []
    episode_load_balance = []
    episode_load_variance = []
    episode_traffic_load_balance = []
    episode_parks = []
    episode_evokes = []
    episode_migrations = []
    episode_noops = []
    episode_failures = []
    overload_steps = 0
    underload_steps = 0
    overload_with_parked_capacity_steps = 0
    underload_with_parking_capacity_steps = 0
    total_steps = 0

    for episode in range(episodes):
        trace_length = len(traffic_data) if traffic_data is not None else steps_per_episode
        # Stratified starts cover low, normal, and peak periods instead of only
        # evaluating the low-load prefix of a chronologically ordered trace.
        start = (episode * trace_length // episodes) % trace_length
        state, _ = env.reset(
            seed=seed + episode, options={"traffic_start_index": start}
        )

        ep_reward = 0.0
        ep_avg_latencies = []
        ep_worst_latencies = []
        ep_energies = []
        ep_load_balance = []
        ep_load_variance = []
        ep_traffic_load_balance = []
        ep_parks = 0
        ep_evokes = 0
        ep_migrations = 0
        ep_noops = 0
        ep_failures = 0

        for _step in range(steps_per_episode):
            if policy == "rdqn":
                _raw_action, action = select_rdqn_action(
                    rdqn_model,
                    state,
                    env,
                    mask_invalid_actions=True,
                )
            elif policy == "random_valid":
                action = select_random_valid_action(env, rng)
            elif policy == "threshold":
                action = select_threshold_action(env)
            elif policy == "noop":
                action = 0
            else:
                raise ValueError(f"Unsupported policy: {policy}")

            next_state, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            ep_reward += float(reward)

            ep_avg_latencies.append(float(info.get("cs_avg_latency", info.get("latency", 0.0))))
            ep_worst_latencies.append(float(info.get("cs_worst_latency", info.get("worst_case_latency", 0.0))))
            ep_energies.append(float(info.get("energy", 0.0)))
            ep_load_balance.append(float(info.get("load_balance_index", 0.0)))
            ep_load_variance.append(float(info.get("load_variance", 0.0)))
            ep_traffic_load_balance.append(float(info.get("traffic_load_balance_index", 0.0)))

            action_type = info.get("action_type", "")
            action_success = bool(info.get("action_success", False))
            has_overload = int(info.get("pre_overloaded_count", 0)) > 0
            has_underload = int(info.get("pre_underloaded_count", 0)) > 0
            has_parked_to_evoke = int(info.get("parked_controllers", 0)) > 0
            has_room_to_park = len(env.active_slaves) > 1
            overload_steps += int(has_overload)
            underload_steps += int(has_underload)
            overload_with_parked_capacity_steps += int(has_overload and has_parked_to_evoke)
            underload_with_parking_capacity_steps += int(has_underload and has_room_to_park)

            if action_type == "noop":
                ep_noops += 1
            elif "park" in action_type and action_success:
                ep_parks += 1
            elif "evoke" in action_type and action_success:
                ep_evokes += 1
            elif "migrate" in action_type and action_success:
                ep_migrations += 1

            if not action_success and action_type != "noop":
                ep_failures += 1

            state = next_state
            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        episode_avg_latencies.append(summarize_episode(ep_avg_latencies))
        episode_worst_latencies.append(float(np.max(ep_worst_latencies)) if ep_worst_latencies else 0.0)
        episode_energies.append(summarize_episode(ep_energies))
        episode_load_balance.append(summarize_episode(ep_load_balance))
        episode_load_variance.append(summarize_episode(ep_load_variance))
        episode_traffic_load_balance.append(summarize_episode(ep_traffic_load_balance))
        episode_parks.append(ep_parks)
        episode_evokes.append(ep_evokes)
        episode_migrations.append(ep_migrations)
        episode_noops.append(ep_noops)
        episode_failures.append(ep_failures)

    env.close()

    return PolicyResult(
        topology=topology,
        policy=policy,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        traffic_path=str(traffic_path) if traffic_path else None,
        model_path=str(model_path) if model_path else None,
        avg_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        cs_avg_latency=float(np.mean(episode_avg_latencies)),
        std_cs_avg_latency=float(np.std(episode_avg_latencies)),
        cs_worst_latency=float(np.mean(episode_worst_latencies)),
        std_cs_worst_latency=float(np.std(episode_worst_latencies)),
        avg_energy=float(np.mean(episode_energies)),
        std_energy=float(np.std(episode_energies)),
        load_balance_index=float(np.mean(episode_load_balance)),
        std_load_balance_index=float(np.std(episode_load_balance)),
        traffic_load_balance_index=float(np.mean(episode_traffic_load_balance)),
        avg_load_variance=float(np.mean(episode_load_variance)),
        avg_parking_events=float(np.mean(episode_parks)),
        avg_evoking_events=float(np.mean(episode_evokes)),
        avg_migration_events=float(np.mean(episode_migrations)),
        avg_noop_events=float(np.mean(episode_noops)),
        avg_failed_actions=float(np.mean(episode_failures)),
        overload_step_fraction=float(overload_steps / total_steps) if total_steps else 0.0,
        underload_step_fraction=float(underload_steps / total_steps) if total_steps else 0.0,
        overload_with_parked_capacity_fraction=(
            float(overload_with_parked_capacity_steps / total_steps) if total_steps else 0.0
        ),
        underload_with_parking_capacity_fraction=(
            float(underload_with_parking_capacity_steps / total_steps) if total_steps else 0.0
        ),
        total_steps=int(total_steps),
    )


def resolve_topologies(value: str) -> list[str]:
    if value == "all":
        return TOPOLOGIES
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def resolve_policies(value: str) -> list[str]:
    if value == "all":
        return ["rdqn", "threshold", "random_valid", "noop"]
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def print_comparison(results: list[PolicyResult]) -> None:
    print("\n" + "=" * 118)
    print("FAIR EVALUATION SUMMARY")
    print("=" * 118)
    print(
        f"{'Topology':<12} {'Policy':<13} {'Reward':>12} {'AvgLat':>9} "
        f"{'WorstLat':>9} {'Energy':>9} {'LoadIdx':>9} {'Park':>7} "
        f"{'Evoke':>7} {'Mig':>7} {'Fail':>7} {'Over%':>7} {'Under%':>7}"
    )
    print("-" * 118)
    for result in results:
        print(
            f"{result.topology:<12} {result.policy:<13} "
            f"{result.avg_reward:>12.2f} {result.cs_avg_latency:>9.2f} "
            f"{result.cs_worst_latency:>9.2f} {result.avg_energy:>9.2f} "
            f"{result.load_balance_index:>9.3f} {result.avg_parking_events:>7.1f} "
            f"{result.avg_evoking_events:>7.1f} {result.avg_migration_events:>7.1f} "
            f"{result.avg_failed_actions:>7.1f} "
            f"{result.overload_step_fraction * 100:>6.1f}% "
            f"{result.underload_step_fraction * 100:>6.1f}%"
        )


def save_results(results: list[PolicyResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"fair_eval_{timestamp}.json"
    csv_path = output_dir / f"fair_eval_{timestamp}.csv"

    payload = [result.as_dict() for result in results]
    json_path.write_text(json.dumps(payload, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload[0].keys()))
        writer.writeheader()
        writer.writerows(payload)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fairly evaluate RDQN against current-environment baselines."
    )
    parser.add_argument(
        "--topologies",
        default="all",
        help="Comma-separated topology list or 'all'.",
    )
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma-separated policy list or 'all'. Options: rdqn,threshold,random_valid,noop.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing LATEST_rainbow_proactive_<topology>.pth checkpoints.",
    )
    parser.add_argument("--output-dir", default="results/fair_evaluations")
    args = parser.parse_args()

    topologies = resolve_topologies(args.topologies)
    policies = resolve_policies(args.policies)
    invalid_topologies = [topology for topology in topologies if topology not in TOPOLOGIES]
    invalid_policies = [policy for policy in policies if policy not in POLICIES]
    if invalid_topologies:
        raise ValueError(f"Unsupported topologies: {invalid_topologies}")
    if invalid_policies:
        raise ValueError(f"Unsupported policies: {invalid_policies}")

    results = []
    for topology in topologies:
        traffic_path = (
            PROJECT_ROOT
            / "data"
            / "traffic"
            / "processed"
            / topology
            / f"{topology}_synthetic_traffic.npy"
        )
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        model_path = model_dir / f"LATEST_rainbow_proactive_{topology}.pth"

        for policy in policies:
            print(f"\nEvaluating topology={topology}, policy={policy}...")
            result = evaluate_policy(
                topology=topology,
                policy=policy,
                episodes=args.episodes,
                steps_per_episode=args.steps_per_episode,
                traffic_path=traffic_path,
                model_path=model_path if policy == "rdqn" else None,
                seed=args.seed,
            )
            results.append(result)

    print_comparison(results)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    json_path, csv_path = save_results(results, output_dir)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
