"""Summarize Track 7 Two Rooms evals.

Expected run layout, produced by ``run_two_rooms_track7.sh``:

    <out_dir>/repr_s50/plan_eval/step-*_eval_only/eval.csv
    <out_dir>/value_s50/plan_eval/step-*_eval_only/eval.csv
    <out_dir>/repr_s200/plan_eval/step-*_eval_only/eval.csv
    <out_dir>/value_s200/plan_eval/step-*_eval_only/eval.csv

The script writes:

    <out_dir>/track7_summary.csv
    <out_dir>/track7_summary.tex
    <out_dir>/track7_success_vs_time.png  (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RUNS = [
    ("repr_s50", "repr_dist", 50),
    ("value_s50", "learned_value", 50),
    ("repr_s200", "repr_dist", 200),
    ("value_s200", "learned_value", 200),
]


def _read_last_eval(run_dir: Path) -> dict[str, float]:
    matches = sorted((run_dir / "plan_eval").glob("step-*_eval_only/eval.csv"))
    if not matches:
        raise FileNotFoundError(f"no eval.csv found under {run_dir / 'plan_eval'}")
    with matches[-1].open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty eval csv: {matches[-1]}")
    row = rows[-1]
    return {
        "success_rate": float(row["success_rate"]),
        "mean_state_dist": float(row["mean_state_dist"]),
        "avg_episode_time": float(row["avg_episode_time"]),
    }


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "objective",
                "num_samples",
                "success_rate",
                "mean_state_dist",
                "avg_episode_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict[str, object]], out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Run & Objective & Samples & Success & Time / ep. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['run']} & {row['objective']} & {row['num_samples']} & "
            f"{100.0 * float(row['success_rate']):.1f}\\% & "
            f"{float(row['avg_episode_time']):.1f}s \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines))


def _try_plot(rows: list[dict[str, object]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional local dependency
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return

    colors = {"repr_dist": "#777777", "learned_value": "#1d6fb8"}
    markers = {50: "o", 200: "s"}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for row in rows:
        obj = str(row["objective"])
        samples = int(row["num_samples"])
        ax.scatter(
            float(row["avg_episode_time"]),
            100.0 * float(row["success_rate"]),
            s=90,
            marker=markers.get(samples, "o"),
            color=colors.get(obj, "#333333"),
            label=f"{obj}, S={samples}",
        )
        ax.annotate(
            str(row["run"]),
            (float(row["avg_episode_time"]), 100.0 * float(row["success_rate"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Average episode time (s)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Track 7: learned value vs latent distance")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for run, objective, num_samples in RUNS:
        values = _read_last_eval(args.out_dir / run)
        rows.append(
            {
                "run": run,
                "objective": objective,
                "num_samples": num_samples,
                **values,
            }
        )

    _write_csv(rows, args.out_dir / "track7_summary.csv")
    _write_latex(rows, args.out_dir / "track7_summary.tex")
    _try_plot(rows, args.out_dir / "track7_success_vs_time.png")
    print(f"wrote Track 7 summary under {args.out_dir}")


if __name__ == "__main__":
    main()
