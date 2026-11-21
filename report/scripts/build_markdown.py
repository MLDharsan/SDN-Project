#!/usr/bin/env python3
"""Build the Markdown edition of the report from the LaTeX sources.

Requires pandoc and a completed LaTeX build (main.aux / main.bbl), so that
citation, figure, table and section numbers match the PDF exactly.

Run from the project root:
    venv/bin/python report/scripts/build_markdown.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "report" / "latex"
OUT = ROOT / "report" / "markdown" / "RESEARCH_REPORT.md"
FIG_PREFIX = "../latex/figures/"

CHAPTERS = ["01_introduction", "02_related_work", "03_methodology",
            "04_experimental_setup", "05_results", "06_conclusion"]
APPENDIX = "07_appendices"


def matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def unwrap(text: str, macro: str) -> str:
    token = "\\" + macro + "{"
    while token in text:
        start = text.index(token)
        brace = start + len(token) - 1
        end = matching_brace(text, brace)
        text = text[:start] + text[brace + 1:end] + text[end + 1:]
    return text


def load_aux() -> tuple[dict, dict]:
    aux = (TEX / "main.aux").read_text()
    cites = dict(re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux))
    labels = {}
    for key, num in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux):
        labels[key] = num
    return cites, labels


def convert_algorithm(match: re.Match) -> str:
    body = match.group(0)
    caption = re.search(r"\\caption\{(.+?)\}\n", body)
    label = re.search(r"\\label\{([^}]+)\}", body)
    inner = re.search(r"\\kill\n(.*?)\\end\{tabbing\}", body, re.S).group(1)
    lines = []
    for raw in inner.strip().splitlines():
        line = raw.replace("\\>", "    ").replace("\\\\", "").replace("\\hfill", "  //")
        line = re.sub(r"\\textbf\{([^}]*)\}", r"\1", line)
        line = re.sub(r"\\emph\{([^}]*)\}", r"\1", line)
        line = line.replace("$", "")
        for a, b in [("\\hat{\\bm{\\ell}}", "\u2113\u0302"), ("\\bm{\\ell}", "\u2113"), ("\\hat\\ell", "\u2113\u0302"),
                     ("\\ell", "\u2113"), ("\\leftarrow", "\u2190"), ("\\rightarrow", "\u2192"),
                     ("\\mathcal{V}", "V"), ("\\mathcal{B}", "B"), ("\\in", " \u2208 "), ("\\cup", " \u222a "),
                     ("\\neq", " \u2260 "), ("\\emptyset", "\u2205"), ("\\le", " \u2264 "), ("\\ge", " \u2265 "),
                     ("\\setminus", " \\ "), ("\\arg\\min", "argmin"), ("\\arg\\max", "argmax"), ("\\max", "max"),
                     ("\\quad", "  "), ("\\varepsilon", "\u03b5"), ("\\cdot", "\u00b7"), ("\\bmod", "mod"),
                     ("\\ldots", "..."), ("\\dots", "..."), ("\\{", "{"), ("\\}", "}"), ("\\ ", " "), ("\\;", " "), ("\\,", " ")]:
            line = line.replace(a, b)
        line = re.sub(r"\^\{\\text\{train\}\}", "_train", line)
        line = re.sub(r"_\{([^}]*)\}", r"_\1", line)
        line = re.sub(r"\\text\{([^}]*)\}", r"\1", line)
        line = re.sub(r"Alg\.~", "Alg. ", line)
        lines.append(re.sub(r"  +", lambda m: m.group(0), line.rstrip()))
    title = caption.group(1) if caption else "Algorithm"
    return (f"\n\\paragraph{{ALGTITLE {label.group(1) if label else ''}|{title}}}\n"
            "\\begin{verbatim}\n" + "\n".join(lines) + "\n\\end{verbatim}\n")


def preprocess(text: str, cites: dict, labels: dict) -> str:
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"\\label\{eq:[^}]+\}", "", text)
    text = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", convert_algorithm, text, flags=re.S)
    text = unwrap(text, "fitwidth")
    text = re.sub(r"\\path\{([^}]*)\}", lambda m: "\\texttt{" + m.group(1).replace("_", "\\_") + "}", text)
    text = re.sub(r"\\cite\{([^}]+)\}",
                  lambda m: "[" + ", ".join(cites.get(k.strip(), "?") for k in m.group(1).split(",")) + "]", text)
    text = re.sub(r"\\(?:ref|eqref)\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), text)
    text = re.sub(r"\\includegraphics(\[[^]]*\])?\{([^}]+)\}",
                  lambda m: "\\includegraphics{" + FIG_PREFIX + m.group(2) + ".png}", text)
    text = text.replace("\\fitwidth", "")
    text = text.replace("\\rdqn", "RDQN").replace("\\grurdqn", "GRU-RDQN")
    text = re.sub(r"\\par\\smallskip\\footnotesize\s*", "\n\nNote: ", text)
    for name in ("small", "scriptsize", "footnotesize", "centering", "smallskip"):
        text = text.replace("\\" + name + "\n", "\n").replace("\\" + name + " ", " ")
    text = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", text)
    text = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", text)
    text = text.replace("\\phantom{+}", "")
    text = re.sub(r"\\paragraph\{(.+?)\}", r"\\textbf{\1}", text)

    def tabularx(m: re.Match) -> str:
        spec = re.sub(r"\{[^}]*\}|@|>|<", "", m.group(1))
        cols = sum(spec.count(ch) for ch in "pLXClrc")
        return "\\begin{tabular}{" + "l" * cols + "}"

    text = re.sub(r"\\begin\{tabularx\}\{\\textwidth\}\{((?:[^{}]|\{[^{}]*\})*)\}", tabularx, text)
    text = text.replace("\\end{tabularx}", "\\end{tabular}")
    return text


def pandoc(latex: str) -> str:
    macros = r"""
\newcommand{\reporttitle}{A Deep Reinforcement Learning Approach for Proactive Load-Balancing Controller Management in Software-Defined Networks}
\newcommand{\supervisor}{Dr. K. Thabotharan}
\newcommand{\pmsd}[2]{$#1 \pm #2$}
"""
    result = subprocess.run(
        ["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars+pipe_tables", "--wrap=none",
         "--shift-heading-level-by=1"],
        input=macros + latex, capture_output=True, text=True, check=True,
    )
    md = result.stdout
    md = re.sub(r"\*\*ALGTITLE ([^|\\]*)\\?\|(.+?)\*\*",
                lambda m: f"**Algorithm {LABELS.get(m.group(1).strip(), '')}: {m.group(2)}**", md)
    return postprocess(md)


def fix_table(block: str) -> str:
    lines = block.strip("\n").splitlines()
    if len(lines) >= 3 and set(lines[0].replace("|", "").strip()) <= {""}:
        lines = [lines[2], lines[1]] + lines[3:]
    return "\n".join(lines)


def postprocess(md: str) -> str:
    md = md.replace("$-$<!-- -->", "\u2212").replace("$-$", "\u2212").replace("\\(W\\)", "(W)")

    def table(m: re.Match) -> str:
        label, body = m.group(1), m.group(2).strip()
        parts = re.split(r"\n\n(?=[^|\n])", body, maxsplit=1)
        tbl, caption = (parts[0], parts[1]) if len(parts) == 2 else (body, "")
        notes = ""
        if "\n\nNote:" in caption:
            caption, notes = caption.split("\n\nNote:", 1)
            notes = "\n\n*Note:" + notes.strip() + "*"
        if not tbl.lstrip().startswith("|"):
            tbl, caption = caption, tbl
        number = LABELS.get(label, "")
        return f"**Table {number}:** {caption.strip()}\n\n{fix_table(tbl)}{notes}\n"

    md = re.sub(r'<div id="(tab:[^"]+)">\n(.*?)\n</div>', table, md, flags=re.S)

    def figure(m: re.Match) -> str:
        label, src, caption = m.group(1), m.group(2), m.group(3)
        number = LABELS.get(label, "")
        return f"![Figure {number}]({src})\n\n*Figure {number}: {caption.strip()}*\n"

    md = re.sub(r'<figure id="(fig:[^"]+)">\s*<img src="([^"]+)"[^>]*/>\s*<figcaption>(.*?)</figcaption>\s*</figure>',
                figure, md, flags=re.S)
    return md


def references(cites: dict) -> str:
    bbl = (TEX / "main.bbl").read_text()
    entries = re.split(r"\\bibitem\{", bbl)[1:]
    out = []
    for entry in entries:
        key, body = entry.split("}", 1)
        body = body.split("\\end{thebibliography}")[0]
        body = body.replace("\\newblock", " ").replace("\\BIBentryALTinterwordspacing", "")
        body = body.replace("\\BIBentrySTDinterwordspacing", "")
        md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=none"],
                            input=body, capture_output=True, text=True, check=True).stdout.strip()
        md = re.sub(r"\s+", " ", md)
        out.append(f"[{cites[key]}] {md}")
    return "\n\n".join(out)


LABELS: dict = {}


def main() -> None:
    global LABELS
    cites, LABELS = load_aux()
    abstract = (TEX / "chapters" / "00_abstract_body.tex").read_text()
    parts = [
        "# A Deep Reinforcement Learning Approach for Proactive Load-Balancing Controller Management in Software-Defined Networks",
        "",
        "**Final-Year Research Project Report -- CSC406S6 / CSC406M6**  ",
        "**Student:** M. Logadharsan (2021/SP/098)  ",
        "**Supervisor:** Dr. K. Thabotharan  ",
        "**Department of Computer Science, Faculty of Science, University of Jaffna -- December 2026**",
        "",
        "> Markdown edition generated from the LaTeX sources in `report/latex/`. The PDF (`report/RESEARCH_REPORT.pdf`) is the authoritative layout; numbering of chapters, figures, tables and references is identical.",
        "",
        "## Abstract",
        "",
        pandoc(preprocess(abstract, cites, LABELS)).strip(),
        "",
        "**Keywords:** software-defined networking; controller management; switch migration; energy efficiency; deep reinforcement learning; Rainbow DQN; GRU traffic forecasting.",
        "",
    ]
    for number, name in enumerate(CHAPTERS, start=1):
        latex = preprocess((TEX / "chapters" / f"{name}.tex").read_text(), cites, LABELS)
        md = pandoc(latex)
        md = re.sub(r"^## ", f"## Chapter {number}: ", md, count=1, flags=re.M)
        parts.append(number_sections(md, str(number)))
    parts += ["## References", "", references(cites), ""]

    appendix = (TEX / "chapters" / f"{APPENDIX}.tex").read_text()
    appendix = appendix.replace("\\input{chapters/07a_generated_tables}", "GENERATEDTABLESPLACEHOLDER")
    md = pandoc(preprocess(appendix, cites, LABELS))
    md = md.replace("GENERATEDTABLESPLACEHOLDER",
                    (ROOT / "report" / "markdown" / "appendix_tables.md").read_text())
    letters = iter("ABCDEFG")
    md = re.sub(r"(?m)^## (.+)$", lambda m: f"## Appendix {next(letters)}: {m.group(1)}", md)
    parts.append(md)

    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT}")


def number_sections(md: str, chapter: str) -> str:
    counters = [0, 0]

    def repl(m: re.Match) -> str:
        level = len(m.group(1))
        if level == 3:
            counters[0] += 1
            counters[1] = 0
            return f"### {chapter}.{counters[0]} {m.group(2)}"
        counters[1] += 1
        return f"#### {chapter}.{counters[0]}.{counters[1]} {m.group(2)}"

    return re.sub(r"(?m)^(###|####) (.+)$", repl, md)


if __name__ == "__main__":
    main()
