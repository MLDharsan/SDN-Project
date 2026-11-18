#!/usr/bin/env python3
"""Create final report figures from frozen synthetic and real-case-study CSVs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sdn-drl")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures"

METRICS = (
    ("avg_reward", "Reward (higher is better)", "reward"),
    ("avg_energy", "Average energy (W, lower is better)", "energy"),
    ("cs_avg_latency", "Average latency (ms, lower is better)", "average_latency"),
    ("cs_worst_latency", "Worst-case latency (ms, lower is better)", "worst_case_latency"),
    ("load_balance_index", "Load index (lower is better)", "load_index"),
)

COLORS = {
    "Learned policy": "#147D8A",
    "Current RDQN": "#147D8A",
    "Predictive GRU-RDQN": "#D35F42",
    "Threshold": "#7D8C44",
    "Random Valid": "#9A6B9E",
    "Noop": "#6C757D",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_for(path: Path, policy: str) -> dict[str, str]:
    rows = read_rows(path)
    return next(row for row in rows if row["policy"] == policy)


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def synthetic_sources() -> list[tuple[str, Path, str]]:
    current = PROJECT_ROOT / "results/final_test/current_selected/fair_eval_20260829_141042.csv"
    cogentco = PROJECT_ROOT / "results/final_test/current_cogentco_120k/fair_eval_20260829_095206.csv"
    predictive = PROJECT_ROOT / "results/final_test/predictive_os3e/fair_eval_20260829_142424.csv"
    return [
        ("Gridnet", current, "rdqn"),
        ("BellCanada", current, "rdqn"),
        ("OS3E", predictive, "rdqn"),
        ("Interoute", current, "rdqn"),
        ("Cogentco", cogentco, "rdqn"),
    ]


def plot_synthetic() -> None:
    policies = (
        ("rdqn", "Learned policy"),
        ("threshold", "Threshold"),
        ("random_valid", "Random Valid"),
        ("noop", "Noop"),
    )
    sources = synthetic_sources()
    topology_names = [name for name, _path, _policy in sources]
    x = np.arange(len(topology_names))
    width = 0.19

    for metric, ylabel, stem in METRICS:
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        for index, (policy, label) in enumerate(policies):
            values = []
            for _topology, source, _selected_policy in sources:
                values.append(value(row_for(source, policy), metric))
            offset = (index - 1.5) * width
            axis.bar(x + offset, values, width, label=label, color=COLORS[label])

        axis.set_title(f"Synthetic held-out test: {ylabel}")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, topology_names)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        axis.text(
            0.01,
            -0.18,
            "Learned policy: current RDQN for Gridnet, BellCanada, Interoute and Cogentco; "
            "predictive GRU-RDQN for OS3E.",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
        figure.tight_layout(rect=(0.0, 0.10, 0.82, 1.0))
        save_figure(figure, OUTPUT_DIR / "synthetic", stem)


def real_final_rows() -> dict[str, list[dict[str, str]]]:
    current_paths = (
        PROJECT_ROOT / "results/real_case_study/final_current/fair_eval_20261114_155511.csv",
        PROJECT_ROOT / "results/real_case_study/replications/seed_43/final_current/fair_eval_20261114_175211.csv",
        PROJECT_ROOT / "results/real_case_study/replications/seed_44/final_current/fair_eval_20261114_193210.csv",
    )
    predictive_paths = (
        PROJECT_ROOT / "results/real_case_study/final_predictive/fair_eval_20261114_155848.csv",
        PROJECT_ROOT / "results/real_case_study/replications/seed_43/final_predictive/fair_eval_20261114_175504.csv",
        PROJECT_ROOT / "results/real_case_study/replications/seed_44/final_predictive/fair_eval_20261114_193318.csv",
    )
    return {
        "Current RDQN": [row_for(path, "rdqn") for path in current_paths],
        "Predictive GRU-RDQN": [row_for(path, "rdqn") for path in predictive_paths],
    }


def plot_real_case_study() -> None:
    rows = real_final_rows()
    labels = list(rows)
    x = np.arange(len(labels))

    aggregate_rows: list[dict[str, str | float]] = []
    for metric, ylabel, stem in METRICS:
        means = [np.mean([value(row, metric) for row in rows[label]]) for label in labels]
        stds = [np.std([value(row, metric) for row in rows[label]], ddof=1) for label in labels]

        figure, axis = plt.subplots(figsize=(7.2, 5.5))
        axis.bar(
            x,
            means,
            yerr=stds,
            capsize=6,
            color=[COLORS[label] for label in labels],
            width=0.55,
        )
        axis.set_title(f"Abilene-derived OS3E final test: {ylabel}")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.5,
            -0.15,
            "Bars show mean; error bars show sample standard deviation across training seeds 42, 43, and 44.",
            transform=axis.transAxes,
            ha="center",
            fontsize=8,
            va="top",
        )
        figure.tight_layout()
        save_figure(figure, OUTPUT_DIR / "real_case_study", stem)

        aggregate_rows.append(
            {
                "metric": stem,
                "current_mean": means[0],
                "current_sample_std": stds[0],
                "predictive_mean": means[1],
                "predictive_sample_std": stds[1],
            }
        )

    aggregate_path = OUTPUT_DIR / "real_case_study" / "three_seed_aggregate.csv"
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> None:
    plot_synthetic()
    plot_real_case_study()
    print(f"Saved final figures under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
