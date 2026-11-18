#!/usr/bin/env python3
"""Prepare a clean Abilene-derived OS3E split for real-data training.

This is separate from the zero-shot case study. It reserves X21-X24 as a
future final-test period and does not reuse X18-X20, which appeared in the
earlier zero-shot evaluation window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.preprocessing.prepare_abilene_os3e_zero_shot import (
    archive_sha256,
    build_projection,
    load_abilene_node_loads,
)

DEFAULT_TRAIN_CHUNKS = ("X06", "X07", "X08", "X09", "X10", "X11", "X12", "X13", "X14")
DEFAULT_VALIDATION_CHUNKS = ("X15", "X16", "X17")
DEFAULT_EXCLUDED_CHUNKS = ("X18", "X19", "X20")
DEFAULT_TEST_CHUNKS = ("X21", "X22", "X23", "X24")


def parse_chunks(value: str) -> tuple[str, ...]:
    chunks = tuple(chunk.strip().upper() for chunk in value.split(",") if chunk.strip())
    if not chunks:
        raise argparse.ArgumentTypeError("provide one or more comma-separated chunks, such as X06,X07")
    return chunks


def normalise(raw: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(raw / scale, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a pre-registered Abilene-derived OS3E real-training split."
    )
    parser.add_argument(
        "--archive",
        default="data/traffic/real_case_study/raw/abilene/AbileneTM-all.tar",
    )
    parser.add_argument(
        "--output-dir",
        default="data/traffic/real_case_study/os3e_real_trained_v1",
    )
    parser.add_argument("--train-chunks", type=parse_chunks, default=DEFAULT_TRAIN_CHUNKS)
    parser.add_argument("--validation-chunks", type=parse_chunks, default=DEFAULT_VALIDATION_CHUNKS)
    parser.add_argument("--excluded-chunks", type=parse_chunks, default=DEFAULT_EXCLUDED_CHUNKS)
    parser.add_argument("--test-chunks", type=parse_chunks, default=DEFAULT_TEST_CHUNKS)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    output_dir = Path(args.output_dir)
    if not archive_path.is_absolute():
        archive_path = PROJECT_ROOT / archive_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not archive_path.exists():
        raise FileNotFoundError(f"Abilene archive not found: {archive_path}")

    groups = {
        "train": args.train_chunks,
        "validation": args.validation_chunks,
        "excluded": args.excluded_chunks,
        "test": args.test_chunks,
    }
    all_chunks = [chunk for group in groups.values() for chunk in group]
    if len(set(all_chunks)) != len(all_chunks):
        raise ValueError("Chunk groups must not overlap.")

    projection, mapping = build_projection()

    def mapped_loads(chunks: tuple[str, ...]) -> np.ndarray:
        return (load_abilene_node_loads(archive_path, list(chunks), None) @ projection.T).astype(np.float32)

    train_raw = mapped_loads(args.train_chunks)
    validation_raw = mapped_loads(args.validation_chunks)
    test_raw = mapped_loads(args.test_chunks)
    scale = np.maximum(np.percentile(train_raw, 99, axis=0), 1e-6).astype(np.float32)
    train = normalise(train_raw, scale)
    validation = normalise(validation_raw, scale)
    test = normalise(test_raw, scale)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.save(splits_dir / "os3e_train.npy", train)
    np.save(splits_dir / "os3e_validation.npy", validation)
    np.save(splits_dir / "os3e_test.npy", test)

    metadata = {
        "case_study": "Abilene measured traffic temporal dynamics mapped to OS3E",
        "protocol": "real_data_trained_v1",
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": archive_sha256(archive_path),
        "chunk_groups": {name: list(chunks) for name, chunks in groups.items()},
        "normalization": {
            "method": "per_switch_training_p99",
            "fitted_on": "train_chunks_only",
            "scale": scale.tolist(),
        },
        "split_shapes": {
            "train": list(train.shape),
            "validation": list(validation.shape),
            "test": list(test.shape),
        },
        "projection": {
            "type": "fixed_two_source_weighted_projection",
            "note": "This is not measured OS3E traffic. It maps measured Abilene temporal demand onto 34 OS3E switches.",
            "mapping": mapping,
        },
        "final_test_rule": "Do not train, tune, or select models using the test chunks.",
    }
    (output_dir / "os3e_abilene_manifest.json").write_text(json.dumps(metadata, indent=2))

    print(f"Train split:      {splits_dir / 'os3e_train.npy'} {train.shape}")
    print(f"Validation split: {splits_dir / 'os3e_validation.npy'} {validation.shape}")
    print(f"Final test split: {splits_dir / 'os3e_test.npy'} {test.shape}")
    print(f"Excluded chunks:  {', '.join(args.excluded_chunks)}")
    print(f"Manifest:         {output_dir / 'os3e_abilene_manifest.json'}")


if __name__ == "__main__":
    main()
