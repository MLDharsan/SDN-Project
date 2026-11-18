#!/usr/bin/env python3
"""Prepare an isolated Abilene-to-OS3E trace for zero-shot evaluation.

The output is not claimed to be measured OS3E traffic. It preserves the
measured temporal dynamics of Abilene traffic matrices, then applies a fixed,
documented projection from 12 Abilene nodes to 34 OS3E switches.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABILENE_NODES = 12
OS3E_SWITCHES = 34
VALUES_PER_OD = 5
MATRIX_VALUES = ABILENE_NODES * ABILENE_NODES * VALUES_PER_OD

# X06 through X20 are a contiguous weekly sequence in the archive metadata.
DEFAULT_START_CHUNK = "X06"
DEFAULT_CHUNK_COUNT = 15
CHUNK_START_DATES = {
    f"X{index:02d}": date
    for index, date in enumerate(
        [
            "2004-03-01", "2004-03-08", "2004-04-02", "2004-04-09",
            "2004-04-22", "2004-05-01", "2004-05-08", "2004-05-15",
            "2004-05-22", "2004-05-29", "2004-06-05", "2004-06-12",
            "2004-06-19", "2004-06-26", "2004-07-03", "2004-07-10",
            "2004-07-17", "2004-07-24", "2004-07-31", "2004-08-07",
            "2004-08-13", "2004-08-21", "2004-08-28", "2004-09-04",
        ],
        start=1,
    )
}


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_chunk_names(start_chunk: str, chunk_count: int) -> list[str]:
    try:
        start_index = int(start_chunk.removeprefix("X"))
    except ValueError as exc:
        raise ValueError("--start-chunk must look like X06") from exc
    names = [f"X{index:02d}" for index in range(start_index, start_index + chunk_count)]
    unknown = [name for name in names if name not in CHUNK_START_DATES]
    if unknown:
        raise ValueError(f"Requested chunk(s) unavailable in the documented archive: {unknown}")
    return names


def real_node_loads(measured_od: np.ndarray) -> np.ndarray:
    """Convert one measured 12x12 OD matrix into per-node ingress+egress demand."""
    matrix = measured_od.reshape(ABILENE_NODES, ABILENE_NODES).copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix.sum(axis=0) + matrix.sum(axis=1)


def build_projection() -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Build a fixed two-source projection from 12 Abilene nodes to 34 switches."""
    projection = np.zeros((OS3E_SWITCHES, ABILENE_NODES), dtype=np.float32)
    mapping = []
    for switch_id in range(OS3E_SWITCHES):
        primary = switch_id % ABILENE_NODES
        secondary = (primary + 1 + switch_id // ABILENE_NODES) % ABILENE_NODES
        projection[switch_id, primary] = 0.80
        projection[switch_id, secondary] += 0.20
        mapping.append(
            {
                "os3e_switch": switch_id,
                "primary_abilene_node": primary,
                "primary_weight": 0.80,
                "secondary_abilene_node": secondary,
                "secondary_weight": 0.20,
            }
        )
    return projection, mapping


def load_abilene_node_loads(
    archive_path: Path, chunk_names: list[str], max_timesteps: int | None
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with tarfile.open(archive_path, "r") as archive:
        for chunk_name in chunk_names:
            member = archive.getmember(f"{chunk_name}.gz")
            compressed = archive.extractfile(member)
            if compressed is None:
                raise ValueError(f"Could not read {chunk_name}.gz from archive")
            with gzip.GzipFile(fileobj=compressed) as stream:
                for raw_line in stream:
                    values = np.fromstring(raw_line.decode("ascii"), sep=" ", dtype=np.float32)
                    if values.size != MATRIX_VALUES:
                        raise ValueError(
                            f"{chunk_name} contains {values.size} values; expected {MATRIX_VALUES}"
                        )
                    measured_od = values[0::VALUES_PER_OD]
                    rows.append(real_node_loads(measured_od))
                    if max_timesteps is not None and len(rows) >= max_timesteps:
                        return np.stack(rows).astype(np.float32)
    return np.stack(rows).astype(np.float32)


def split_and_normalize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    train_raw = data[:train_end]
    validation_raw = data[train_end:validation_end]
    test_raw = data[validation_end:]
    scale = np.maximum(np.percentile(train_raw, 99, axis=0), 1e-6).astype(np.float32)
    return (
        np.clip(train_raw / scale, 0.0, 1.0),
        np.clip(validation_raw / scale, 0.0, 1.0),
        np.clip(test_raw / scale, 0.0, 1.0),
        scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert measured Abilene matrices into an isolated OS3E zero-shot trace."
    )
    parser.add_argument(
        "--archive",
        default="data/traffic/real_case_study/raw/abilene/AbileneTM-all.tar",
    )
    parser.add_argument(
        "--output-dir",
        default="data/traffic/real_case_study/os3e",
    )
    parser.add_argument("--start-chunk", default=DEFAULT_START_CHUNK)
    parser.add_argument("--chunk-count", type=int, default=DEFAULT_CHUNK_COUNT)
    parser.add_argument("--max-timesteps", type=int)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    output_dir = Path(args.output_dir)
    if not archive_path.is_absolute():
        archive_path = PROJECT_ROOT / archive_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not archive_path.exists():
        raise FileNotFoundError(f"Abilene archive not found: {archive_path}")
    if args.chunk_count < 1:
        raise ValueError("--chunk-count must be positive")

    chunks = selected_chunk_names(args.start_chunk, args.chunk_count)
    node_loads = load_abilene_node_loads(archive_path, chunks, args.max_timesteps)
    projection, mapping = build_projection()
    os3e_raw = (node_loads @ projection.T).astype(np.float32)
    train, validation, test, scale = split_and_normalize(os3e_raw)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "os3e_abilene_raw.npy", os3e_raw)
    np.save(splits_dir / "os3e_train.npy", train)
    np.save(splits_dir / "os3e_validation.npy", validation)
    np.save(splits_dir / "os3e_test.npy", test)

    start_time = datetime.fromisoformat(CHUNK_START_DATES[chunks[0]])
    metadata = {
        "case_study": "Abilene measured traffic temporal dynamics mapped to OS3E",
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": archive_sha256(archive_path),
        "source_matrix_shape": [ABILENE_NODES, ABILENE_NODES],
        "source_matrix_interval_minutes": 5,
        "selected_chunks": chunks,
        "start_timestamp": start_time.isoformat(),
        "end_timestamp_approximate": (
            start_time + timedelta(minutes=5 * (len(os3e_raw) - 1))
        ).isoformat(),
        "timesteps": int(len(os3e_raw)),
        "output_shape": list(os3e_raw.shape),
        "normalization": {
            "method": "per_switch_training_p99",
            "fitted_on": "first_70_percent_training_split_only",
            "scale": scale.tolist(),
        },
        "split_ranges": {
            "train": [0, len(train)],
            "validation": [len(train), len(train) + len(validation)],
            "test": [len(train) + len(validation), len(os3e_raw)],
        },
        "projection": {
            "type": "fixed_two_source_weighted_projection",
            "note": "This is not measured OS3E traffic. It maps measured Abilene temporal demand onto 34 OS3E switches.",
            "mapping": mapping,
        },
    }
    (output_dir / "os3e_abilene_manifest.json").write_text(json.dumps(metadata, indent=2))

    print(f"Raw trace:   {output_dir / 'os3e_abilene_raw.npy'} {os3e_raw.shape}")
    print(f"Train split: {splits_dir / 'os3e_train.npy'} {train.shape}")
    print(f"Validation:  {splits_dir / 'os3e_validation.npy'} {validation.shape}")
    print(f"Test split:  {splits_dir / 'os3e_test.npy'} {test.shape}")
    print(f"Manifest:    {output_dir / 'os3e_abilene_manifest.json'}")


if __name__ == "__main__":
    main()
