#!/usr/bin/env python3
"""Train a GRU to forecast future per-switch traffic."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.gru_traffic_predictor import GRUTrafficNetwork  # noqa: E402


class TrafficWindowDataset(Dataset):
    def __init__(
        self,
        traffic: np.ndarray,
        history_length: int,
        forecast_horizon: int,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.traffic = np.asarray(traffic, dtype=np.float32)
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.sample_count = len(traffic) - history_length - forecast_horizon + 1
        if self.sample_count <= 0:
            raise ValueError("Traffic split is too short for the selected history and horizon")

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int):
        end = index + self.history_length
        target_index = end + self.forecast_horizon - 1
        history = (self.traffic[index:end] - self.mean) / self.std
        target = (self.traffic[target_index] - self.mean) / self.std
        return torch.from_numpy(history), torch.from_numpy(target)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    overload_target = target > 0.70
    overload_prediction = prediction > 0.70
    true_positive = int(np.logical_and(overload_target, overload_prediction).sum())
    false_positive = int(np.logical_and(~overload_target, overload_prediction).sum())
    false_negative = int(np.logical_and(overload_target, ~overload_prediction).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "mae": mae,
        "rmse": rmse,
        "overload_precision": precision,
        "overload_recall": recall,
        "overload_f1": f1,
    }


@torch.no_grad()
def evaluate(model, loader, mean, std, device) -> tuple[float, dict]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    predictions, targets = [], []
    for history, target in loader:
        history, target = history.to(device), target.to(device)
        prediction = model(history)
        total_loss += float(criterion(prediction, target).item())
        predictions.append(prediction.cpu().numpy() * std + mean)
        targets.append(target.cpu().numpy() * std + mean)
    prediction_values = np.clip(np.concatenate(predictions), 0, 1)
    target_values = np.clip(np.concatenate(targets), 0, 1)
    return total_loss / (len(loader.dataset) * target_values.shape[1]), metrics(prediction_values, target_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a per-switch GRU traffic forecaster.")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--split-dir")
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split_dir = Path(args.split_dir) if args.split_dir else (
        PROJECT_ROOT / "data" / "traffic" / "processed" / args.topology / "splits"
    )
    train = np.load(split_dir / f"{args.topology}_train.npy")
    validation = np.load(split_dir / f"{args.topology}_validation.npy")
    test_path = split_dir / f"{args.topology}_test.npy"
    mean = train.mean(axis=0).astype(np.float32)
    std = np.maximum(train.std(axis=0), 1e-6).astype(np.float32)

    train_dataset = TrafficWindowDataset(train, args.history_length, args.forecast_horizon, mean, std)
    validation_dataset = TrafficWindowDataset(validation, args.history_length, args.forecast_horizon, mean, std)
    validation_targets = validation[
        args.history_length + args.forecast_horizon - 1:
    ]
    persistence_predictions = validation[
        args.history_length - 1:
        args.history_length - 1 + len(validation_targets)
    ]
    persistence_metrics = metrics(persistence_predictions, validation_targets)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUTrafficNetwork(
        num_switches=train.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_state, best_validation_loss, best_metrics = None, float("inf"), {}
    remaining_patience = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for history, target in train_loader:
            history, target = history.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(history), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_loss, validation_metrics = evaluate(model, validation_loader, mean, std, device)
        print(
            f"Epoch {epoch:>3}/{args.epochs} | train MSE={np.mean(losses):.6f} | "
            f"validation MSE={validation_loss:.6f} | MAE={validation_metrics['mae']:.5f}"
        )
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_metrics = validation_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            remaining_patience = args.patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                print("Early stopping: validation loss did not improve.")
                break

    model.load_state_dict(best_state)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "models" / "forecasting"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"gru_{args.topology}_h{args.history_length}_f{args.forecast_horizon}_{timestamp}.pth"
    config = {
        "topology": args.topology,
        "num_switches": int(train.shape[1]),
        "history_length": args.history_length,
        "forecast_horizon": args.forecast_horizon,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
    }
    payload = {
        "model_state": best_state,
        "config": config,
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "source": "training_split_only"},
        "validation_metrics": best_metrics,
        "validation_persistence_baseline": persistence_metrics,
        "split_dir": str(split_dir.resolve()),
    }
    if args.evaluate_test:
        test = np.load(test_path)
        test_loader = DataLoader(
            TrafficWindowDataset(test, args.history_length, args.forecast_horizon, mean, std),
            batch_size=args.batch_size,
            shuffle=False,
        )
        _test_loss, payload["test_metrics"] = evaluate(model, test_loader, mean, std, device)
        print(f"Held-out test metrics: {json.dumps(payload['test_metrics'], indent=2)}")
    torch.save(payload, model_path)
    latest = output_dir / f"LATEST_gru_{args.topology}.pth"
    latest.unlink(missing_ok=True)
    latest.symlink_to(model_path.name)
    print(f"GRU model: {model_path}")
    print(f"Latest:    {latest}")
    print(f"Validation metrics: {json.dumps(best_metrics, indent=2)}")
    print(f"Persistence baseline: {json.dumps(persistence_metrics, indent=2)}")


if __name__ == "__main__":
    main()
