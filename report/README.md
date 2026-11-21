# Final Research Report (CSC406S6 / CSC406M6)

| File | What it is |
|---|---|
| `RESEARCH_REPORT.pdf` | Final report (64 pages, A4, IEEE references) — print this |
| `RESEARCH_REPORT_overleaf.zip` | Overleaf project: *New Project → Upload Project*, compiler **pdfLaTeX**, main file `main.tex` |
| `markdown/RESEARCH_REPORT.md` | Markdown edition (same numbering as the PDF; images from `latex/figures/`) |
| `latex/` | LaTeX sources: `main.tex`, `chapters/`, `references.bib`, `figures/` (TikZ sources in `figures/src/`) |
| `scripts/` | Read-only generators for figures, appendix tables and the Markdown edition |

Edit names, dates and supervisor in the metadata block at the top of `latex/main.tex`.

## Rebuild (from the project root)

```bash
venv/bin/python report/scripts/make_report_figures.py    # figures from frozen CSVs / logs / checkpoints
venv/bin/python report/scripts/make_appendix_tables.py   # Appendix A tables
cd report/latex && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd -
venv/bin/python report/scripts/build_markdown.py         # needs the LaTeX build (main.aux / main.bbl)
```

The scripts only read from `results/`, `models/`, `logs/` and `data/`; they write only inside `report/`.

## Note on the older figures

`experiments/plot_final_results.py` selects CSV rows by policy only, so the synthetic figures in
`results/final_figures/synthetic/` show Gridnet's values for BellCanada and Interoute (also embedded in
`FINAL_RESEARCH_PROJECT_REPORT_2026-11-16.pdf`). The figures in this report select rows by
(topology, policy) and are correct. The real-case-study figures are unaffected.
