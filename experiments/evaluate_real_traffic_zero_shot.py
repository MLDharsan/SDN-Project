#!/usr/bin/env python3
"""Evaluate an existing SDN model on an explicit real-traffic trace.

This script reuses the project's fair policy evaluation implementation while
keeping the synthetic-data evaluator unchanged. It supports current-state and
predictive synthetic-to-real zero-shot OS3E case studies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_fair_baselines import (
    POLICIES,
    evaluate_policy,
    print_comparison,
    resolve_policies,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing OS3E model on an explicit real-traffic trace."
    )
    parser.add_argument("--traffic", required=True, help="Normalized [timesteps, 34] real traffic .npy file")
    parser.add_argument("--model", required=True, help="Existing RDQN checkpoint")
    parser.add_argument(
        "--observation-mode",
        choices=["current", "predictive"],
        default="predictive",
        help="Use current network state only, or state augmented with a GRU forecast.",
    )
    parser.add_argument(
        "--gru-model",
        help="GRU checkpoint; required only when --observation-mode predictive.",
    )
    parser.add_argument("--policies", default="all")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--output-dir", default="results/real_case_study/zero_shot_os3e")
    args = parser.parse_args()

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    traffic_path = resolve_path(args.traffic)
    model_path = resolve_path(args.model)
    if args.observation_mode == "predictive" and not args.gru_model:
        parser.error("--gru-model is required when --observation-mode predictive")

    gru_model_path = resolve_path(args.gru_model) if args.gru_model else None
    output_dir = resolve_path(args.output_dir)
    policies = resolve_policies(args.policies)
    invalid_policies = [policy for policy in policies if policy not in POLICIES]
    if invalid_policies:
        raise ValueError(f"Unsupported policies: {invalid_policies}")
    for label, path in {
        "traffic": traffic_path,
        "RDQN model": model_path,
        **({"GRU model": gru_model_path} if gru_model_path else {}),
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    results = []
    for policy in policies:
        print(
            f"\nEvaluating zero-shot OS3E, mode={args.observation_mode}, "
            f"policy={policy}..."
        )
        results.append(
            evaluate_policy(
                topology="os3e",
                policy=policy,
                episodes=args.episodes,
                steps_per_episode=args.steps_per_episode,
                traffic_path=traffic_path,
                model_path=model_path if policy == "rdqn" else None,
                seed=args.seed,
                observation_mode=args.observation_mode,
                gru_model_path=gru_model_path,
            )
        )

    print_comparison(results)
    json_path, csv_path = save_results(results, output_dir)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
