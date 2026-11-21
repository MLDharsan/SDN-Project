#!/usr/bin/env python3
"""Generate every data figure used in the final research report.

Read-only with respect to the project: it reads the frozen result CSVs,
TensorBoard logs, GRU checkpoints and traffic splits, and writes figures only
into report/latex/figures/.

Note: experiments/plot_final_results.py selects rows by policy only, so in the
multi-topology CSV it reuses the first (Gridnet) row for BellCanada and
Interoute. This script selects rows by (topology, policy) to avoid that.

Run from the project root:
    venv/bin/python report/scripts/make_report_figures.py
"""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sdn-drl-report")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical order (blue, orange, aqua, yellow).
C_LEARNED, C_THRESH, C_RANDOM, C_NOOP = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_CURRENT, C_PRED = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "savefig.dpi": 300,
})

TOPOS = ["gridnet", "bellcanada", "os3e", "interoute", "cogentco"]
NAMES = {"gridnet": "Gridnet", "bellcanada": "BellCanada", "os3e": "OS3E",
         "interoute": "Interoute", "cogentco": "Cogentco"}
POLICIES = [("rdqn", "Learned policy", C_LEARNED), ("threshold", "Threshold", C_THRESH),
            ("random_valid", "Random-valid", C_RANDOM), ("noop", "No-op", C_NOOP)]


def rows(path: str | Path) -> list[dict]:
    with open(ROOT / path, newline="") as handle:
        return list(csv.DictReader(handle))


def pick(table: list[dict], topology: str, policy: str) -> dict:
    return next(r for r in table if r["topology"] == topology and r["policy"] == policy)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- synthetic
SYN = {
    "gridnet": "results/final_test/current_selected/fair_eval_20260829_141042.csv",
    "bellcanada": "results/final_test/current_selected/fair_eval_20260829_141042.csv",
    "os3e": "results/final_test/predictive_os3e/fair_eval_20260829_142424.csv",
    "interoute": "results/final_test/current_selected/fair_eval_20260829_141042.csv",
    "cogentco": "results/final_test/current_cogentco_120k/fair_eval_20260829_095206.csv",
}


def syn_value(topology: str, policy: str, metric: str) -> float:
    return float(pick(rows(SYN[topology]), topology, policy)[metric])


SHORT = {"gridnet": "Grid.", "bellcanada": "Bell", "os3e": "OS3E",
         "interoute": "Inter.", "cogentco": "Cogent."}


def grouped(ax, metric: str, ylabel: str, title: str, short: bool = False) -> None:
    x = np.arange(len(TOPOS))
    w = 0.2
    for i, (pol, label, color) in enumerate(POLICIES):
        vals = [syn_value(t, pol, metric) for t in TOPOS]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, label=label, color=color)
    ax.set_xticks(x, [(SHORT if short else NAMES)[t] for t in TOPOS])
    ax.set_title(title, loc="left")
    style(ax, ylabel)


def fig_synthetic() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    grouped(ax, "avg_reward", "Cumulative reward per episode", "")
    ax.set_ylim(-560, -350)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "syn_reward")

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
    spec = [("avg_energy", "Energy (W)", "(a) Simulated energy, lower is better"),
            ("cs_avg_latency", "Latency (ms)", "(b) Average latency, lower is better"),
            ("cs_worst_latency", "Latency (ms)", "(c) Worst-case latency, lower is better"),
            ("load_balance_index", "Std. of switch counts", "(d) Load index, lower is better")]
    for ax, (m, yl, ti) in zip(axes.flat, spec):
        grouped(ax, m, yl, ti, short=True)
    axes[0, 0].set_ylim(300, 480)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "syn_metrics")


# ---------------------------------------------------------------- real data
REAL = {
    42: ("results/real_case_study/final_current/fair_eval_20261114_155511.csv",
         "results/real_case_study/final_predictive/fair_eval_20261114_155848.csv"),
    43: ("results/real_case_study/replications/seed_43/final_current/fair_eval_20261114_175211.csv",
         "results/real_case_study/replications/seed_43/final_predictive/fair_eval_20261114_175504.csv"),
    44: ("results/real_case_study/replications/seed_44/final_current/fair_eval_20261114_193210.csv",
         "results/real_case_study/replications/seed_44/final_predictive/fair_eval_20261114_193318.csv"),
}


def fig_real() -> None:
    seeds = list(REAL)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    x = np.arange(len(seeds))
    w = 0.36
    cur = [float(pick(rows(REAL[s][0]), "os3e", "rdqn")["avg_reward"]) for s in seeds]
    pre = [float(pick(rows(REAL[s][1]), "os3e", "rdqn")["avg_reward"]) for s in seeds]
    axes[0].bar(x - w / 2, cur, w * 0.92, color=C_CURRENT, label="Current RDQN")
    axes[0].bar(x + w / 2, pre, w * 0.92, color=C_PRED, label="Predictive GRU-RDQN")
    thr = float(pick(rows(REAL[42][0]), "os3e", "threshold")["avg_reward"])
    axes[0].axhline(thr, color=INK2, linestyle="--", linewidth=0.9)
    axes[0].text(2.45, thr + 3, "current-mode\nthreshold", ha="right", va="bottom", fontsize=7, color=INK2)
    axes[0].set_xticks(x, [f"seed {s}" for s in seeds])
    axes[0].set_ylim(-160, 0)
    axes[0].set_title("(a) Reward per seed", loc="left")
    style(axes[0], "Cumulative reward")

    for ax, metric, ylabel, title in [
        (axes[1], "avg_energy", "Energy (W)", "(b) Energy, mean ± s.d."),
        (axes[2], "cs_avg_latency", "Latency (ms)", "(c) Avg. latency, mean ± s.d."),
    ]:
        c = [float(pick(rows(REAL[s][0]), "os3e", "rdqn")[metric]) for s in seeds]
        p = [float(pick(rows(REAL[s][1]), "os3e", "rdqn")[metric]) for s in seeds]
        ax.bar([0, 1], [np.mean(c), np.mean(p)], 0.55,
               yerr=[np.std(c, ddof=1), np.std(p, ddof=1)], capsize=4,
               color=[C_CURRENT, C_PRED], error_kw={"elinewidth": 0.9, "ecolor": INK})
        ax.set_xticks([0, 1], ["Current", "Predictive"])
        ax.set_title(title, loc="left")
        style(ax, ylabel)
    axes[1].set_ylim(240, 285)
    axes[2].set_ylim(5.0, 6.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "real_three_seed")


def fig_zero_shot() -> None:
    cur = rows("results/real_case_study/zero_shot_os3e_current/fair_eval_20261107_111507.csv")
    pre = rows("results/real_case_study/zero_shot_os3e/fair_eval_20261028_212001.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    x = np.arange(len(POLICIES))
    w = 0.36
    for ax, metric, ylabel, title in [
        (axes[0], "avg_reward", "Cumulative reward", "(a) Reward, higher is better"),
        (axes[1], "avg_energy", "Energy (W)", "(b) Energy, lower is better"),
    ]:
        c = [float(pick(cur, "os3e", p)[metric]) for p, _, _ in POLICIES]
        q = [float(pick(pre, "os3e", p)[metric]) for p, _, _ in POLICIES]
        ax.bar(x - w / 2, c, w * 0.92, color=C_CURRENT, label="Current-state mode")
        ax.bar(x + w / 2, q, w * 0.92, color=C_PRED, label="Predictive mode")
        ax.set_xticks(x, ["RDQN", "Threshold", "Random-valid", "No-op"])
        ax.set_title(title, loc="left")
        style(ax, ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "zero_shot")


# ---------------------------------------------------------------- GRU
def fig_gru() -> None:
    import torch

    def load(pattern: str):
        path = sorted(glob.glob(str(ROOT / pattern)))[0]
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt["validation_metrics"], ckpt["validation_persistence_baseline"]

    labels, gru, pers = [], [], []
    for t in TOPOS:
        v, p = load(f"models/forecasting/gru_{t}_h12_f5_*.pth")
        labels.append(f"{NAMES[t]}\n(syn., h=5)")
        gru.append(v["mae"])
        pers.append(p["mae"])
    v, p = load("models/real_case_study/forecasting/gru_os3e_h12_f5_*.pth")
    labels.append("OS3E\n(Abilene, h=5)")
    gru.append(v["mae"])
    pers.append(p["mae"])

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, gru, w * 0.92, color=C_PRED, label="GRU forecaster")
    ax.bar(x + w / 2, pers, w * 0.92, color=C_CURRENT, label="Persistence baseline")
    for xi, g, q in zip(x, gru, pers):
        change = 100 * (q - g) / q
        ax.text(xi, max(g, q) + 0.0007, f"{change:+.1f}%", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, 0.046)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    style(ax, "Validation MAE")
    save(fig, "gru_vs_persistence")


# ---------------------------------------------------------------- training
RUNS = [
    ("rainbow_current_proactive_gridnet_20260719_122327", "Gridnet (current)"),
    ("rainbow_current_proactive_bellcanada_20260719_125454", "BellCanada (current)"),
    ("rainbow_current_proactive_os3e_20260719_135124", "OS3E (current)"),
    ("rainbow_predictive_proactive_os3e_20260719_170318", "OS3E (predictive)"),
    ("rainbow_current_proactive_interoute_20260808_132140", "Interoute (current)"),
    ("rainbow_current_proactive_cogentco_20260815_222043", "Cogentco (current, 120k)"),
]


def fig_training() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharey=False)
    for ax, (run, title) in zip(axes.flat, RUNS):
        acc = EventAccumulator(str(ROOT / "logs" / "proactive_dqn_tensorboard" / run),
                               size_guidance={"scalars": 0})
        acc.Reload()
        events = acc.Scalars("episode/reward")
        ep = np.array([e.step for e in events]) + 1
        val = np.array([e.value for e in events])
        ax.plot(ep, val, color=C_CURRENT, alpha=0.3, linewidth=0.8)
        k = 5
        if len(val) >= k:
            smooth = np.convolve(val, np.ones(k) / k, mode="valid")
            ax.plot(ep[k - 1:], smooth, color=C_CURRENT, linewidth=1.6)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.tick_params(labelsize=7)
        style(ax, "")
    for ax in axes[1]:
        ax.set_xlabel("Training episode (1,000 steps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward")
    fig.tight_layout()
    save(fig, "training_curves")


# ---------------------------------------------------------------- traffic
def fig_traffic() -> None:
    syn = np.load(ROOT / "data/traffic/processed/os3e/splits/os3e_train.npy")[:2016]
    real = np.load(ROOT / "data/traffic/real_case_study/os3e_real_trained_v1/splits/os3e_train.npy")[:2016]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True)
    for ax, data, title in [
        (axes[0], syn, "(a) Synthetic OS3E trace (training split)"),
        (axes[1], real, "(b) Abilene-derived OS3E trace (training chunks; 5-min intervals, one week)"),
    ]:
        t = np.arange(len(data))
        lo, hi = np.percentile(data, 10, axis=1), np.percentile(data, 90, axis=1)
        ax.fill_between(t, lo, hi, color=C_CURRENT, alpha=0.18, linewidth=0, label="10th to 90th percentile across switches")
        ax.plot(t, data.mean(axis=1), color=C_CURRENT, linewidth=0.9, label="Mean switch load")
        ax.axhline(0.70, color=C_PRED, linestyle="--", linewidth=0.9)
        ax.axhline(0.30, color=INK2, linestyle=":", linewidth=0.9)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=8.5)
        style(ax, "Normalised load")
    axes[0].text(len(syn) - 5, 0.72, "overload 0.70", ha="right", va="bottom", fontsize=7, color=C_PRED)
    axes[0].text(len(syn) - 5, 0.28, "underload 0.30", ha="right", va="top", fontsize=7, color=INK2)
    axes[1].set_xlabel("Decision interval (environment step)")
    axes[0].legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.42), fontsize=7.5)
    fig.tight_layout()
    save(fig, "traffic_traces")


if __name__ == "__main__":
    fig_synthetic()
    fig_real()
    fig_zero_shot()
    fig_gru()
    fig_training()
    fig_traffic()
    print(f"Figures written to {OUT}")
