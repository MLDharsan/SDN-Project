#!/usr/bin/env python3
"""Generate the appendix result tables (LaTeX and Markdown) from frozen CSVs.

Read-only with respect to the project; writes into report/latex/chapters/ and
report/markdown/. Run from the project root:
    venv/bin/python report/scripts/make_appendix_tables.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX_OUT = ROOT / "report" / "latex" / "chapters" / "07a_generated_tables.tex"
MD_OUT = ROOT / "report" / "markdown" / "appendix_tables.md"

NAMES = {"gridnet": "Gridnet", "bellcanada": "BellCanada", "os3e": "OS3E",
         "interoute": "Interoute", "cogentco": "Cogentco"}
POL = {"rdqn": "RDQN", "threshold": "Threshold", "random_valid": "Random-valid", "noop": "No-op"}
ORDER = ["rdqn", "threshold", "random_valid", "noop"]

COLS = [  # key, header, format
    ("avg_reward", "Reward", "{:.2f}"),
    ("std_reward", "s.d.", "{:.1f}"),
    ("cs_avg_latency", "AvgLat", "{:.2f}"),
    ("cs_worst_latency", "WorstLat", "{:.2f}"),
    ("avg_energy", "Energy", "{:.2f}"),
    ("load_balance_index", "LoadIdx", "{:.3f}"),
    ("avg_parking_events", "Park", "{:.1f}"),
    ("avg_evoking_events", "Evoke", "{:.1f}"),
    ("avg_migration_events", "Mig", "{:.1f}"),
    ("avg_failed_actions", "Fail", "{:.1f}"),
    ("overload_step_fraction", "Over\\%", "pct"),
    ("underload_step_fraction", "Under\\%", "pct"),
]

TABLES = [
    ("E1 final test, current mode (Gridnet, BellCanada, OS3E, Interoute; 80k steps).",
     "tab:app-e1-current", ["results/final_test/current_selected/fair_eval_20260829_141042.csv"]),
    ("E1 final test, Cogentco current mode (120k steps).",
     "tab:app-e1-cogentco", ["results/final_test/current_cogentco_120k/fair_eval_20260829_095206.csv"]),
    ("E1 final test, OS3E predictive mode (GRU horizon 5).",
     "tab:app-e1-pred", ["results/final_test/predictive_os3e/fair_eval_20260829_142424.csv"]),
    ("Synthetic validation, current mode, all topologies (Cogentco at 80k steps).",
     "tab:app-val-current", ["results/validation/current_all/fair_eval_20260815_215014.csv"]),
    ("Synthetic validation, predictive mode (GRU horizon 5).",
     "tab:app-val-pred", ["results/validation/predictive/fair_eval_20260731_213635.csv"]),
    ("Synthetic validation, predictive mode (GRU horizon 3).",
     "tab:app-val-pred-h3", ["results/validation/predictive_h3/fair_eval_20260808_130945.csv"]),
    ("E2 zero-shot, current mode (Abilene-derived OS3E test trace).",
     "tab:app-e2-current", ["results/real_case_study/zero_shot_os3e_current/fair_eval_20261107_111507.csv"]),
    ("E2 zero-shot, predictive mode (Abilene-derived OS3E test trace).",
     "tab:app-e2-pred", ["results/real_case_study/zero_shot_os3e/fair_eval_20261028_212001.csv"]),
]

E3 = {
    42: "results/real_case_study",
    43: "results/real_case_study/replications/seed_43",
    44: "results/real_case_study/replications/seed_44",
}


def rows(path: str) -> list[dict]:
    with open(ROOT / path, newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(row: dict, key: str, spec: str) -> str:
    value = float(row[key])
    if spec == "pct":
        return f"{100 * value:.1f}"
    return spec.format(value)


def sort_rows(table: list[dict]) -> list[dict]:
    topo_order = list(NAMES)
    return sorted(table, key=lambda r: (topo_order.index(r["topology"]), ORDER.index(r["policy"])))


def latex_table(caption: str, label: str, data: list[tuple[str, dict]]) -> str:
    header = " & ".join(["\\textbf{Topology}", "\\textbf{Policy}"] + [f"\\textbf{{{h}}}" for _, h, _ in COLS])
    lines = [
        "\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\fitwidth{\\begin{tabular}{@{}ll" + "r" * len(COLS) + "@{}}", "\\toprule", header + "\\\\", "\\midrule",
    ]
    previous = None
    for group, row in data:
        if previous is not None and group != previous:
            lines.append("\\midrule")
        cells = [group if group != previous else "", POL[row["policy"]]]
        cells += [fmt(row, k, s).replace("-", "$-$", 1) if k == "avg_reward" else fmt(row, k, s) for k, _, s in COLS]
        lines.append(" & ".join(cells) + "\\\\")
        previous = group
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def md_table(caption: str, data: list[tuple[str, dict]]) -> str:
    heads = ["Group", "Policy"] + [h.replace("\\", "") for _, h, _ in COLS]
    out = [f"**{caption}**", "", "| " + " | ".join(heads) + " |",
           "|" + "---|" * 2 + "---:|" * len(COLS)]
    for group, row in data:
        out.append("| " + " | ".join([group, POL[row["policy"]]] + [fmt(row, k, s) for k, _, s in COLS]) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    tex, md = [], []
    for caption, label, paths in TABLES:
        data = []
        for path in paths:
            data += [(NAMES[r["topology"]], r) for r in sort_rows(rows(path))]
        tex.append(latex_table(caption, label, data))
        md.append(md_table(caption, data))

    for stage, label, caption in [
        ("validation", "tab:app-e3-val", "E3 validation (Abilene X15--X17) per training seed."),
        ("final", "tab:app-e3-final", "E3 final test (Abilene X21--X24) per training seed."),
    ]:
        data = []
        for seed, base in E3.items():
            for mode in ("current", "predictive"):
                folder = ROOT / base / f"{stage}_{mode}"
                path = sorted(folder.glob("fair_eval_*.csv"))[0].relative_to(ROOT)
                for r in sort_rows(rows(str(path))):
                    data.append((f"Seed {seed}, {mode[:4]}.", r))
        tex.append(latex_table(caption, label, data))
        md.append(md_table(caption.replace("--", "-"), data))

    TEX_OUT.write_text("% AUTO-GENERATED by report/scripts/make_appendix_tables.py -- do not edit by hand.\n" + "\n".join(tex))
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(md))
    print(f"Wrote {TEX_OUT} and {MD_OUT}")


if __name__ == "__main__":
    main()
