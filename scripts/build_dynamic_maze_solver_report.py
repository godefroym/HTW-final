#!/usr/bin/env python3
"""Build a compact dynamic-maze solver report from saved eval artifacts.

Outputs:
- normalized metrics CSV/TSV/JSON;
- one success/failure GIF per method when available;
- static GIF previews/contact sheets for LaTeX/PDF;
- a Markdown report with animated GIFs;
- a LaTeX report that can be compiled to PDF.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dynamic_maze_solver_report_20260620"
ORACLE_STRICT = (
    ROOT
    / "outputs"
    / "dynamic_maze_baseline_strict_20260620_031717"
    / "oracle_replan"
    / "metrics.json"
)
ORACLE_SEQ = (
    ROOT
    / "outputs"
    / "dynamic_maze_oracle_seq_bootstrap_ref_20260620"
    / "metrics.json"
)


METHODS = [
    {
        "slug": "oracle_replan",
        "name": "Oracle A* replanning",
        "family": "symbolic",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "oracle_replan"
        / "metrics.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "oracle_replan",
        "note": "Full current grid A* replanning. Strong myopic oracle, not stochastic-optimal.",
    },
    {
        "slug": "memory_optimistic",
        "name": "Memory optimistic A*",
        "family": "symbolic",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "memory_optimistic"
        / "metrics.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "memory_optimistic",
        "note": "A* on accumulated visible map, unknown cells treated as free.",
    },
    {
        "slug": "fog_optimistic",
        "name": "Fog optimistic A*",
        "family": "symbolic",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "fog_optimistic"
        / "metrics.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "fog_optimistic",
        "note": "A* on current local observation, unknown cells treated as free.",
    },
    {
        "slug": "fog_conservative",
        "name": "Fog conservative A*",
        "family": "symbolic",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "fog_conservative"
        / "metrics.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "fog_conservative",
        "note": "A* on current local observation, unknown cells treated as walls.",
    },
    {
        "slug": "memory_conservative",
        "name": "Memory conservative A*",
        "family": "symbolic",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "memory_conservative"
        / "metrics.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "memory_conservative",
        "note": "A* on accumulated map, never-seen cells treated as walls.",
    },
    {
        "slug": "random",
        "name": "Random",
        "family": "symbolic",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "random"
        / "metrics.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_baseline_strict_20260620_031717"
        / "random",
        "note": "Uniform random cardinal action baseline.",
    },
    {
        "slug": "jepa_subgoal_n2",
        "name": "JEPA subgoal N=2",
        "family": "astar_distilled",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_jepa_subgoal_starter_20260620_034609"
        / "02_eval_N2"
        / "subgoal_eval.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_jepa_subgoal_starter_20260620_034609"
        / "02_eval_N2",
        "note": "A*-distilled waypoint head; A* not queried at eval.",
    },
    {
        "slug": "jepa_subgoal_n4",
        "name": "JEPA subgoal N=4",
        "family": "astar_distilled",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_jepa_subgoal_starter_20260620_034609"
        / "02_eval_N4"
        / "subgoal_eval.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_jepa_subgoal_starter_20260620_034609"
        / "02_eval_N4",
        "note": "A*-distilled longer waypoint head; first saved GIFs are all successes.",
    },
    {
        "slug": "learned_value",
        "name": "Bootstrap learned value",
        "family": "bootstrap_wm",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_bootstrap_20260620_051011"
        / "01_eval_learned_value"
        / "learned_value_eval.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_bootstrap_20260620_051011"
        / "01_eval_learned_value",
        "note": "A*-free local exploration + hindsight TD value.",
    },
    {
        "slug": "probe_pos",
        "name": "Bootstrap probe-pos",
        "family": "bootstrap_wm",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_bootstrap_20260620_051011"
        / "01_eval_probe_pos"
        / "probe_pos_eval.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_bootstrap_20260620_051011"
        / "01_eval_probe_pos",
        "note": "World-model rollout scored by decoded position distance.",
    },
    {
        "slug": "repr_dist",
        "name": "Bootstrap repr-dist",
        "family": "bootstrap_wm",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_bootstrap_20260620_051011"
        / "01_eval_repr_dist"
        / "repr_dist_eval.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_bootstrap_20260620_051011"
        / "01_eval_repr_dist",
        "note": "World-model rollout scored by latent distance to goal.",
    },
    {
        "slug": "belief_value",
        "name": "BeliefLSTM value",
        "family": "bootstrap_wm",
        "metrics": ROOT
        / "outputs"
        / "dynamic_maze_belief_lstm_20260620_085948"
        / "01_eval_belief_value"
        / "belief_value_eval.json",
        "gifs": ROOT
        / "outputs"
        / "dynamic_maze_belief_lstm_20260620_085948"
        / "01_eval_belief_value",
        "note": "Frozen JEPA + recurrent belief state + learned value.",
    },
]


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100.0 * x:.1f}%"


def one_decimal(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.1f}"


def three_decimal(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.3f}"


def tex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def tex_cell(s: str) -> str:
    return s.replace("%", r"\%")


def find_gif(gif_dir: Path, status: str) -> Path | None:
    if not gif_dir.exists():
        return None
    pattern = "*succ*.gif" if status == "positive" else "*fail*.gif"
    matches = sorted(gif_dir.glob(pattern))
    return matches[0] if matches else None


def copy_gifs(row: dict, spec: dict):
    dst_dir = OUT / "gifs" / row["slug"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for status in ("positive", "negative"):
        src = find_gif(spec["gifs"], status)
        row[f"{status}_src"] = str(src) if src else ""
        if src:
            dst = dst_dir / f"{status}.gif"
            shutil.copy2(src, dst)
            row[f"{status}_gif"] = str(dst.relative_to(OUT))
        else:
            row[f"{status}_gif"] = ""


def gif_frame(path: Path, last: bool = True) -> Image.Image:
    im = Image.open(path)
    frame = None
    if last:
        try:
            while True:
                frame = im.copy().convert("RGB")
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    if frame is None:
        im.seek(0)
        frame = im.copy().convert("RGB")
    return frame


def make_placeholder(path: Path, title: str, subtitle: str):
    img = Image.new("RGB", (360, 220), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 359, 219], outline=(190, 190, 190), width=2)
    draw.text((18, 70), title, fill=(40, 40, 40))
    draw.text((18, 110), subtitle, fill=(90, 90, 90))
    img.save(path)


def make_preview(row: dict, status: str):
    preview_dir = OUT / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    out = preview_dir / f"{row['slug']}_{status}.png"
    gif_rel = row[f"{status}_gif"]
    if not gif_rel:
        success = row["success_rate"]
        if status == "positive" and success == 0:
            reason = "No success in run"
        elif status == "negative" and success == 1:
            reason = "No failure in run"
        else:
            reason = "GIF not saved"
        make_placeholder(out, row["name"], reason)
        row[f"{status}_preview"] = str(out.relative_to(OUT))
        return

    gif_path = OUT / gif_rel
    first = gif_frame(gif_path, last=False)
    last = gif_frame(gif_path, last=True)
    scale = 3
    first = first.resize((first.width * scale, first.height * scale), Image.Resampling.NEAREST)
    last = last.resize((last.width * scale, last.height * scale), Image.Resampling.NEAREST)
    w = first.width + last.width + 28
    h = max(first.height, last.height) + 54
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"{row['name']} - {status}", fill=(0, 0, 0))
    canvas.paste(first, (8, 34))
    canvas.paste(last, (first.width + 20, 34))
    draw.text((8, h - 16), "start", fill=(70, 70, 70))
    draw.text((first.width + 20, h - 16), "end", fill=(70, 70, 70))
    canvas.save(out)
    row[f"{status}_preview"] = str(out.relative_to(OUT))


def make_contact(rows: list[dict], status: str):
    cols = 3
    cell_w, cell_h = 420, 285
    rows_n = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        prev = Image.open(OUT / row[f"{status}_preview"]).convert("RGB")
        prev.thumbnail((cell_w - 24, cell_h - 54), Image.Resampling.LANCZOS)
        draw.text((x + 10, y + 8), row["name"], fill=(0, 0, 0))
        draw.text(
            (x + 10, y + 24),
            f"success={pct(row['success_rate'])}, steps={one_decimal(row['mean_steps'])}",
            fill=(70, 70, 70),
        )
        sheet.paste(prev, (x + 10, y + 48))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(220, 220, 220))
    out = OUT / "figures" / f"contact_{status}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out.relative_to(OUT)


def row_from_spec(spec: dict, oracle_strict_steps: float, oracle_seq_steps: float) -> dict:
    data = read_json(spec["metrics"])
    success = float(data.get("success_rate", 0.0))
    steps = float(data.get("mean_steps", 0.0))
    blocked = data.get("mean_blocked_moves")
    blocked = float(blocked) if blocked is not None else None
    final_manhattan = data.get("mean_final_manhattan")
    final_manhattan = float(final_manhattan) if final_manhattan is not None else None
    reported_eff = data.get("mean_efficiency")
    reported_eff = float(reported_eff) if reported_eff is not None else None
    episodes = data.get("episodes", data.get("num_episodes"))
    episodes = int(episodes) if episodes is not None else None
    oracle_steps = oracle_seq_steps if spec["family"] == "bootstrap_wm" else oracle_strict_steps
    step_ratio = oracle_steps / steps if steps else None
    combined = success * step_ratio if step_ratio is not None else None
    row = {
        "slug": spec["slug"],
        "name": spec["name"],
        "family": spec["family"],
        "success_rate": success,
        "mean_steps": steps,
        "mean_blocked_moves": blocked,
        "mean_final_manhattan": final_manhattan,
        "episodes": episodes,
        "reported_astar_efficiency": reported_eff,
        "oracle_reference_steps": oracle_steps,
        "step_eff_vs_oracle_mean": step_ratio,
        "combined_success_step_ratio": combined,
        "note": spec["note"],
    }
    copy_gifs(row, spec)
    return row


def write_metrics(rows: list[dict]):
    fields = [
        "slug",
        "name",
        "family",
        "success_rate",
        "mean_steps",
        "mean_blocked_moves",
        "mean_final_manhattan",
        "episodes",
        "reported_astar_efficiency",
        "oracle_reference_steps",
        "step_eff_vs_oracle_mean",
        "combined_success_step_ratio",
        "positive_gif",
        "negative_gif",
        "note",
    ]
    for suffix, delimiter in [("csv", ","), ("tsv", "\t")]:
        with (OUT / f"summary_metrics.{suffix}").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
    with (OUT / "summary_metrics.json").open("w") as f:
        json.dump(rows, f, indent=2)


def write_markdown(rows: list[dict], success_contact: Path, failure_contact: Path):
    total_runs = sum(r["episodes"] or 0 for r in rows)
    lines = [
        "# Dynamic Maze Solver Comparison",
        "",
        "This report compares symbolic baselines, A*-distilled JEPA controllers, and A*-free world-model controllers.",
        "",
        "The PDF report uses static previews; this Markdown file embeds the animated GIFs directly.",
        "",
        f"Closed-loop evaluation episodes represented in the table: **{total_runs}** "
        f"({len(rows)} methods, usually 64 episodes per method).",
        "",
        "The symbolic/A*-distilled and bootstrap runs were produced by different launchers; "
        "oracle-normalized ratios are therefore operational comparisons, not a single perfectly matched benchmark.",
        "",
        "## Metrics",
        "",
        "| Method | Runs | Success | Steps | Blocked | A* eff. | Step/oracle | Combined |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['episodes'] or '--'} | {pct(r['success_rate'])} | {one_decimal(r['mean_steps'])} | "
            f"{one_decimal(r['mean_blocked_moves'])} | {three_decimal(r['reported_astar_efficiency'])} | "
            f"{pct(r['step_eff_vs_oracle_mean'])} | {pct(r['combined_success_step_ratio'])} |"
        )
    lines += [
        "",
        "## Contact Sheets",
        "",
        f"![Success previews]({success_contact})",
        "",
        f"![Failure previews]({failure_contact})",
        "",
        "## Animated GIFs",
        "",
        "| Method | Success example | Failure example |",
        "|---|---|---|",
    ]
    for r in rows:
        pos = f"![success]({r['positive_gif']})" if r["positive_gif"] else "_not available_"
        neg = f"![failure]({r['negative_gif']})" if r["negative_gif"] else "_not available_"
        lines.append(f"| {r['name']} | {pos} | {neg} |")
    (OUT / "DYNAMIC_MAZE_SOLVER_REPORT.md").write_text("\n".join(lines) + "\n")


def write_latex(rows: list[dict], success_contact: Path, failure_contact: Path):
    total_runs = sum(r["episodes"] or 0 for r in rows)
    table_rows = []
    for r in rows:
        table_rows.append(
            " & ".join(
                [
                    tex_escape(r["name"]),
                    str(r["episodes"] or "--"),
                    tex_cell(pct(r["success_rate"])),
                    one_decimal(r["mean_steps"]),
                    one_decimal(r["mean_blocked_moves"]),
                    three_decimal(r["reported_astar_efficiency"]),
                    tex_cell(pct(r["step_eff_vs_oracle_mean"])),
                    tex_cell(pct(r["combined_success_step_ratio"])),
                ]
            )
            + r" \\"
        )
    notes = "\n".join(
        rf"\item \textbf{{{tex_escape(r['name'])}}}: {tex_escape(r['note'])}"
        for r in rows
    )
    gif_rows = []
    for r in rows:
        pos = (
            rf"\href{{run:gifs/{r['slug']}/positive.gif}}{{success GIF}}"
            if r["positive_gif"]
            else "--"
        )
        neg = (
            rf"\href{{run:gifs/{r['slug']}/negative.gif}}{{failure GIF}}"
            if r["negative_gif"]
            else "--"
        )
        gif_rows.append(rf"{tex_escape(r['name'])} & {pos} & {neg} \\")

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.55in]{{geometry}}
\usepackage{{booktabs,longtable,graphicx,hyperref,xcolor}}
\usepackage{{array}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}

\title{{Dynamic Maze Solver Comparison}}
\author{{HTW EB-JEPA}}
\date{{20 June 2026}}

\begin{{document}}
\maketitle

\section*{{Scope}}
This document summarizes the dynamic-maze experiments: symbolic A* baselines,
A*-distilled JEPA subgoal controllers, and A*-free world-model controllers.
Animated GIFs are saved next to this report; the PDF contains static start/end
previews and clickable links to the GIF files.

The table covers {total_runs} closed-loop evaluation episodes across
{len(rows)} methods. Most rows use 64 episodes. The symbolic/A*-distilled and
bootstrap runs come from different launchers, so oracle-normalized ratios should
be read as operational comparisons rather than as one perfectly matched
benchmark.

\section*{{Metric Definitions}}
\begin{{itemize}}
  \item Success is the empirical probability of reaching the goal over 64 episodes.
  \item Runs is the number of closed-loop evaluation episodes available for that method.
  \item Steps is the mean number of executed moves.
  \item Blocked moves counts attempted moves into walls/closed doors.
  \item A* efficiency is the evaluator's per-episode \(A_0/\max(\mathrm{{steps}}, A_0)\), when available.
  \item Step/oracle is \(\mathrm{{mean\ oracle\ steps}}/\mathrm{{mean\ solver\ steps}}\).
  \item Combined is success times Step/oracle; it rewards both reaching the goal and doing so efficiently.
\end{{itemize}}

\section*{{Results}}
\small
\begin{{longtable}}{{p{{0.24\linewidth}}rrrrrrr}}
\toprule
Method & Runs & Success & Steps & Blocked & A* eff. & Step/oracle & Combined \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Success Examples}}
\begin{{center}}
\includegraphics[width=\linewidth]{{{success_contact}}}
\end{{center}}

\section*{{Failure Examples}}
\begin{{center}}
\includegraphics[width=\linewidth]{{{failure_contact}}}
\end{{center}}

\section*{{GIF Links}}
\begin{{longtable}}{{p{{0.42\linewidth}}ll}}
\toprule
Method & Success & Failure \\
\midrule
{chr(10).join(gif_rows)}
\bottomrule
\end{{longtable}}

\section*{{Notes}}
\begin{{itemize}}
{notes}
\end{{itemize}}

\section*{{Main Reading}}
The strongest non-oracle controller in this batch is still representation
distance in JEPA latent space: it reaches 75.0\% success with 244.9 mean steps.
The BeliefLSTM value improves the first learned value head, reaching 73.4\%
success, but it still makes more blocked moves. The next useful Track 7
iteration should make the learned value explicitly penalize blocked transitions
and use memory signals that distinguish seen-passable, seen-blocked, and
unseen cells.

\end{{document}}
"""
    (OUT / "DYNAMIC_MAZE_SOLVER_REPORT.tex").write_text(tex)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oracle_strict = read_json(ORACLE_STRICT)
    oracle_seq = read_json(ORACLE_SEQ)
    oracle_strict_steps = float(oracle_strict["mean_steps"])
    oracle_seq_steps = float(oracle_seq["mean_steps"])

    rows = [row_from_spec(m, oracle_strict_steps, oracle_seq_steps) for m in METHODS]
    for row in rows:
        make_preview(row, "positive")
        make_preview(row, "negative")
    success_contact = make_contact(rows, "positive")
    failure_contact = make_contact(rows, "negative")
    write_metrics(rows)
    write_markdown(rows, success_contact, failure_contact)
    write_latex(rows, success_contact, failure_contact)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
