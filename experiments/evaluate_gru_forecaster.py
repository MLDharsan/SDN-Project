#!/usr/bin/env python3
"""Evaluate a frozen GRU forecaster on a chronological traffic split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.gru_traffic_predictor import GRUTrafficForecaster  # noqa: E402
from experiments.train_gru_forecaster import metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen GRU traffic model.")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--model")
    parser.add_argument("--traffic-split", choices=["validation", "test"], default="test")
    parser.add_argument("--traffic")
    parser.add_argument("--output-dir", default="results/forecasting")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else (
        PROJECT_ROOT / "models" / "forecasting" / f"LATEST_gru_{args.topology}.pth"
    )
    traffic_path = Path(args.traffic) if args.traffic else (
        PROJECT_ROOT / "data" / "traffic" / "processed" / args.topology / "splits" /
        f"{args.topology}_{args.traffic_split}.npy"
    )
    forecaster = GRUTrafficForecaster(model_path, device="cpu")
    traffic = np.load(traffic_path)
    history_length = forecaster.history_length
    horizon = forecaster.forecast_horizon
    sample_count = len(traffic) - history_length - horizon + 1
    if sample_count <= 0:
        raise ValueError("Traffic split is too short for this GRU configuration")

    predictions = []
    targets = []
    persistence = []
    for index in range(sample_count):
        end = index + history_length
        target_index = end + horizon - 1
        history = traffic[index:end]
        predictions.append(forecaster.predict(history))
        persistence.append(history[-1])
        targets.append(traffic[target_index])
    predictions = np.asarray(predictions)
    persistence = np.asarray(persistence)
    targets = np.asarray(targets)
    result = {
        "topology": args.topology,
        "traffic_split": args.traffic_split,
        "model_path": str(model_path.resolve()),
        "traffic_path": str(traffic_path.resolve()),
        "samples": sample_count,
        "history_length": history_length,
        "forecast_horizon": horizon,
        "gru_metrics": metrics(predictions, targets),
        "persistence_baseline_metrics": metrics(persistence, targets),
    }
    print(json.dumps(result, indent=2))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"gru_eval_{args.topology}_{args.traffic_split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
