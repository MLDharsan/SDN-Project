#!/usr/bin/env python3
"""Create chronological train/validation/test traffic files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_temporally(
    data: np.ndarray, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if data.ndim != 2 or len(data) < 3:
        raise ValueError("Traffic must have shape [timesteps, switches] with at least 3 rows")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training plus validation fraction must be less than one")
    train_end = int(len(data) * train_fraction)
    validation_end = int(len(data) * (train_fraction + validation_fraction))
    metadata = {
        "total_rows": len(data),
        "num_switches": data.shape[1],
        "train_range": [0, train_end],
        "validation_range": [train_end, validation_end],
        "test_range": [validation_end, len(data)],
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "chronological": True,
    }
    return data[:train_end], data[train_end:validation_end], data[validation_end:], metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronologically split a traffic trace.")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source = Path(args.input) if args.input else (
        PROJECT_ROOT / "data" / "traffic" / "processed" / args.topology /
        f"{args.topology}_synthetic_traffic.npy"
    )
    output_dir = Path(args.output_dir) if args.output_dir else source.parent / "splits"
    data = np.load(source)
    train, validation, test, metadata = split_temporally(
        data, args.train_fraction, args.validation_fraction
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / f"{args.topology}_train.npy",
        "validation": output_dir / f"{args.topology}_validation.npy",
        "test": output_dir / f"{args.topology}_test.npy",
    }
    np.save(paths["train"], train)
    np.save(paths["validation"], validation)
    np.save(paths["test"], test)
    metadata.update({"source": str(source.resolve()), "files": {k: str(v.resolve()) for k, v in paths.items()}})
    manifest = output_dir / f"{args.topology}_split_manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2))
    print(f"Train:      {paths['train']} {train.shape}")
    print(f"Validation: {paths['validation']} {validation.shape}")
    print(f"Test:       {paths['test']} {test.shape}")
    print(f"Manifest:   {manifest}")


if __name__ == "__main__":
    main()
