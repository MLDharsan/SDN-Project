#!/usr/bin/env python3
"""Tables and figure for the Mininet emulation validation (E4).

Read-only with respect to the project: it reads the run folders in
mininet_results/ and writes only
  report/latex/chapters/07b_mininet_tables.tex   (appendix table, LaTeX)
  report/markdown/appendix_mininet_tables.md     (same table, Markdown)
  report/latex/figures/mininet_validation.{pdf,png}
  report/latex/figures/mininet_summary.txt       (numbers quoted in Chapter 5)

Run from the project root:
    venv/bin/python report/scripts/make_mininet_results.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sdn-drl-report")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MN = ROOT / "mininet_results"
FIG = ROOT / "report" / "latex" / "figures"
TEX_OUT = ROOT / "report" / "latex" / "chapters" / "07b_mininet_tables.tex"
MD_OUT = ROOT / "report" / "markdown" / "appendix_mininet_tables.md"
TXT_OUT = FIG / "mininet_summary.txt"

# (folder, short label, window label, mode)
RUNS = [
    ("20261121_131444_os3e_synthetic_current", "S72-cur", "Synthetic 72--221", "current"),
    ("20261121_142455_os3e_synthetic_predictive", "S72-pred", "Synthetic 72--221", "predictive"),
    ("syn_w600_current", "S600-cur", "Synthetic 600--749", "current"),
    ("syn_w600_predictive", "S600-pred", "Synthetic 600--749", "predictive"),
    ("real_w5250_current", "R5250-cur", "Abilene 5250--5399", "current"),
    ("real_w5250_predictive", "R5250-pred", "Abilene 5250--5399", "predictive"),
]
NOISE = [  # (folder, run-name prefix inside, window label)
    ("noise_robustness_syn_w72", "nc", "Synthetic 72--221"),
    ("noise_robustness_syn_w600", "nc", "Synthetic 600--749"),
    ("noise_robustness_real_w5250", "noisecheck", "Abilene 5250--5399"),
]
ORDER = ["rdqn", "threshold", "random_valid", "noop"]
POL = {"rdqn": "Learned", "threshold": "Threshold", "random_valid": "Random-valid", "noop": "No-op"}

C_LEARNED, C_THRESH, C_RANDOM, C_NOOP = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
COLORS = {"rdqn": C_LEARNED, "threshold": C_THRESH, "random_valid": C_RANDOM, "noop": C_NOOP}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 300,
})


def summary(folder: str, policy: str) -> dict:
    return json.loads((MN / folder / policy / "summary.json").read_text())


def steps(folder: str, policy: str) -> list[dict]:
    path = MN / folder / policy / "emulated_steps.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"n": int(arr.size), "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max())}


def neg(x: float, spec: str = "{:.2f}") -> str:
    text = spec.format(x)
    return text.replace("-", "$-$", 1) if text.startswith("-") else text


# ------------------------------------------------------------------ data
def collect() -> dict:
    table = []
    ops = {"migrate": [], "evoke": [], "park": []}
    handover = {}  # (mode, policy) -> list
    flow_install, cal, loss, rtt, jitter, cpu, rss, setup, fmae = [], [], [], [], [], [], [], [], []
    offered_loss = []  # (offered_mbps, loss) from no-op only (identical data plane for all)
    totals = {"ops": 0, "failed": 0, "inconsistent": 0, "steps": 0}
    for folder, short, window, mode in RUNS:
        for policy in ORDER:
            s = summary(folder, policy)
            emu, ref, em = s["simulator_style_metrics"], s["simulator_reference_metrics"], s["emulation_metrics"]
            table.append({"run": short, "window": window, "mode": mode, "policy": policy,
                          "emu": emu, "sim": ref, "agree": s["action_agreement_with_simulator"]})
            totals["ops"] += em["network_ops_total"]
            totals["failed"] += em["network_ops_failed"]
            totals["inconsistent"] += em["inconsistent_steps"]
            setup.append(s["setup"]["setup_s"])
            for row in steps(folder, policy):
                totals["steps"] += 1
                cal.append(row["calibration_mae"])
                tr = row["traffic"]
                loss.append(tr["loss_percent"])
                rtt.append(tr["rtt_avg_ms"])
                jitter.append(tr["jitter_ms"])
                if policy == "noop":
                    offered_loss.append((tr["offered_mbps"], tr["loss_percent"], "Abilene" if "Abilene" in window else "Synthetic"))
                for res in row["controller_resources"].values():
                    cpu.append(res.get("cpu_percent"))
                    rss.append(res.get("rss_mb"))
                for op in row["network_ops"]:
                    if not op.get("success"):
                        continue
                    if op["op"] == "migrate":
                        ops["migrate"].append(op["handover_ms"])
                        flow_install.append(op["flow_install_ms"])
                        handover.setdefault((mode, policy), []).append(op["handover_ms"])
                    elif op["op"] == "evoke":
                        ops["evoke"].append(op["startup_ms"])
                    elif op["op"] == "park":
                        ops["park"].append(op["stop_ms"])
            if mode == "predictive":
                f = em.get("forecast_mae_vs_measured") or {}
                if f.get("mean") is not None:
                    fmae.append(f["mean"])
    noise = []
    for folder, prefix, window in NOISE:
        for mode in ("current", "predictive"):
            vals = {"rdqn": [], "threshold": []}
            for seed in range(1, 9):
                run = MN / folder / f"{prefix}_{mode}_{seed}"
                for policy in vals:
                    vals[policy].append(json.loads((run / policy / "summary.json").read_text())
                                        ["simulator_style_metrics"]["total_reward"])
            noise.append({"window": window, "mode": mode, **vals})
    return {"table": table, "ops": ops, "handover": handover, "flow_install": flow_install, "cal": cal,
            "loss": loss, "rtt": rtt, "jitter": jitter, "cpu": cpu, "rss": rss, "setup": setup,
            "fmae": fmae, "offered_loss": offered_loss, "totals": totals, "noise": noise}


# ------------------------------------------------------------------ appendix table
COLS = [("Reward (Mn)", lambda r: neg(r["emu"]["total_reward"])),
        ("Reward (Sim)", lambda r: neg(r["sim"]["total_reward"])),
        ("Agree\\%", lambda r: f"{100 * r['agree']:.1f}"),
        ("Energy", lambda r: f"{r['emu']['avg_energy']:.1f}"),
        ("AvgLat", lambda r: f"{r['emu']['cs_avg_latency']:.2f}"),
        ("Park", lambda r: str(r["emu"]["parking_events"])),
        ("Evoke", lambda r: str(r["emu"]["evoking_events"])),
        ("Mig", lambda r: str(r["emu"]["migration_events"])),
        ("Fail", lambda r: str(r["emu"]["failed_actions"])),
        ("Over\\%", lambda r: f"{100 * r['emu']['overload_step_fraction']:.1f}"),
        ("Under\\%", lambda r: f"{100 * r['emu']['underload_step_fraction']:.1f}")]


def write_tables(data: dict) -> None:
    caption = ("E4 Mininet emulation, all runs (150 steps each, evaluation seed 1042). Mn = emulated on "
               "Mininet/OVS/os-ken; Sim = simulator on the same trace window. Agree = share of steps on which the "
               "emulated action equals the simulator action. Park/Evoke/Mig/Fail are counts per 150-step run. "
               "Learned = current RDQN in current-mode runs and GRU-RDQN in predictive-mode runs.")
    head = " & ".join(["\\textbf{Run}", "\\textbf{Policy}"] + [f"\\textbf{{{h}}}" for h, _ in COLS])
    tex = ["% AUTO-GENERATED by report/scripts/make_mininet_results.py -- do not edit by hand.",
           "\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", "\\label{tab:app-e4}",
           "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           "\\fitwidth{\\begin{tabular}{@{}ll" + "r" * len(COLS) + "@{}}", "\\toprule", head + "\\\\", "\\midrule"]
    md = [f"**{caption.replace('--', '-')}**", "",
          "| Run | Policy | " + " | ".join(h.replace("\\", "") for h, _ in COLS) + " |",
          "|---|---|" + "---:|" * len(COLS)]
    previous = None
    for row in data["table"]:
        label = f"{row['window']}, {row['mode'][:4]}."
        if previous is not None and label != previous:
            tex.append("\\midrule")
        cells = [label if label != previous else "", POL[row["policy"]]] + [f(row) for _, f in COLS]
        tex.append(" & ".join(cells) + "\\\\")
        md.append("| " + " | ".join([label.replace("--", "-"), POL[row["policy"]]] +
                                     [f(row).replace("$-$", "−") for _, f in COLS]) + " |")
        previous = label
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    TEX_OUT.write_text("\n".join(tex))
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(md) + "\n")


# ------------------------------------------------------------------ figure
def write_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.15, 1]})

    ax = axes[0]
    lo, hi = -205, -30
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.7, linestyle="--", zorder=1)
    for row in data["table"]:
        marker = "o" if row["mode"] == "current" else "^"
        ax.scatter(row["sim"]["total_reward"], row["emu"]["total_reward"], s=22, marker=marker,
                   color=COLORS[row["policy"]], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Simulator reward (same window)")
    ax.set_ylabel("Mininet reward")
    ax.set_title("(a) Emulated vs simulated", loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], linestyle="", marker="s", color=COLORS[p], label=POL[p]) for p in ORDER]
    handles += [plt.Line2D([], [], linestyle="", marker="o", color=INK2, label="current"),
                plt.Line2D([], [], linestyle="", marker="^", color=INK2, label="predictive")]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right", handletextpad=0.2, borderaxespad=0.1)

    ax = axes[1]
    labels = []
    for i, entry in enumerate(data["noise"]):
        x = i
        vals = np.asarray(entry["rdqn"])
        jitter = np.linspace(-0.18, 0.18, len(vals))
        color = C_LEARNED if entry["mode"] == "current" else C_THRESH
        ax.scatter(x + jitter, vals, s=12, color=color, edgecolor="white", linewidth=0.3, zorder=3)
        labels.append(("S72" if "72" in entry["window"] else "S600" if "600" in entry["window"] else "A5250")
                      + ("\ncur." if entry["mode"] == "current" else "\npred."))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Reward (8 noisy replays)")
    ax.set_title("(b) Robustness to load noise", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], linestyle="", marker="o", color=C_LEARNED, label="Current RDQN"),
                       plt.Line2D([], [], linestyle="", marker="o", color=C_THRESH, label="GRU-RDQN")],
              fontsize=6.5, loc="lower left")

    ax = axes[2]
    for profile, color in (("Synthetic", C_LEARNED), ("Abilene", C_THRESH)):
        pts = np.asarray([(o, l) for o, l, g in data["offered_loss"] if g == profile])
        ax.scatter(pts[:, 0], pts[:, 1], s=5, color=color, alpha=0.5, edgecolor="none", label=profile)
    ax.legend(fontsize=6.5, loc="upper left", markerscale=2.5)
    ax.set_xlabel("Offered load (Mb/s)")
    ax.set_ylabel("UDP loss (%)")
    ax.set_title("(c) Data plane", loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "mininet_validation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "mininet_validation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ text summary
def write_summary(data: dict) -> None:
    out = []
    p = out.append
    p("E4 Mininet summary (generated by make_mininet_results.py)\n")
    p(f"{'run':11s} {'policy':12s} {'emu':>9s} {'sim':>9s} {'agree':>6s} {'energy':>7s} ev pk mig over%")
    for r in data["table"]:
        e = r["emu"]
        p(f"{r['run']:11s} {r['policy']:12s} {e['total_reward']:9.2f} {r['sim']['total_reward']:9.2f} "
          f"{100 * r['agree']:6.1f} {e['avg_energy']:7.1f} {e['evoking_events']:2d} {e['parking_events']:2d} "
          f"{e['migration_events']:3d} {100 * e['overload_step_fraction']:5.1f}")
    agree = [r["agree"] for r in data["table"]]
    p(f"\nmean agreement all={100 * np.mean(agree):.1f}%  learned-predictive="
      f"{[round(100 * r['agree'], 1) for r in data['table'] if r['policy'] == 'rdqn' and r['mode'] == 'predictive']}")
    t = data["totals"]
    p(f"steps={t['steps']} network_ops={t['ops']} failed={t['failed']} inconsistent_steps={t['inconsistent']}")
    for name, values in [("handover_ms(all migrations)", data["ops"]["migrate"]),
                         ("flow_install_ms", data["flow_install"]),
                         ("controller_start_ms", data["ops"]["evoke"]),
                         ("controller_stop_ms", data["ops"]["park"]),
                         ("calibration_mae", data["cal"]), ("loss_percent", data["loss"]),
                         ("rtt_avg_ms", data["rtt"]), ("jitter_ms", data["jitter"]),
                         ("controller_cpu_percent", data["cpu"]), ("controller_rss_mb", data["rss"]),
                         ("setup_s", data["setup"]), ("gru_forecast_mae_vs_measured(per run)", data["fmae"])]:
        s = stats(values)
        p(f"{name:40s} n={s['n']:5d} mean={s['mean']:.4f} p95={s['p95']:.4f} max={s['max']:.4f}")
    p("\nhandover by (mode, policy):")
    for key in sorted(data["handover"]):
        s = stats(data["handover"][key])
        p(f"  {key}: n={s['n']} mean={s['mean']:.1f} p95={s['p95']:.1f}")
    for profile in ("Synthetic", "Abilene"):
        pts = [(o, l) for o, l, g in data["offered_loss"] if g == profile]
        free = [o for o, l in pts if l < 0.5]
        p(f"\n{profile}: steps={len(pts)} loss<0.5% up to offered {max(free):.0f} Mb/s; "
          f"loss>1% steps={sum(1 for _, l in pts if l > 1)}; mean loss={np.mean([l for _, l in pts]):.2f}%")
    p("\nnoise robustness (8 replays, load noise sd 0.001):")
    for n in data["noise"]:
        p(f"  {n['window']:20s} {n['mode']:10s} rdqn={sorted(set(round(v, 2) for v in n['rdqn']))} "
          f"threshold={sorted(set(round(v, 2) for v in n['threshold']))}")
    TXT_OUT.write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main() -> None:
    data = collect()
    write_tables(data)
    write_figure(data)
    write_summary(data)
    print(f"\nWrote {TEX_OUT}, {MD_OUT}, {FIG / 'mininet_validation.pdf'}")


if __name__ == "__main__":
    main()
