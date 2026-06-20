"""Plot dynamic-maze layouts without fog and summarize stochastic gates."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from eb_jepa.datasets.dynamic_maze.dynamic_maze import (
    DynamicMazeDatasetConfig,
    _sample_layout,
)


def gate_open_probability(p0: float, toggle_prob: float, step: np.ndarray) -> np.ndarray:
    """Probability that a two-state door is open after `step` toggles."""
    return 0.5 + (p0 - 0.5) * ((1.0 - 2.0 * toggle_prob) ** step)


def draw_layout(ax, cfg: DynamicMazeDatasetConfig, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base, doors, door_open, start, goal = _sample_layout(cfg, rng)

    # 0 = wall, 1 = free corridor. Doors are overlaid separately.
    ax.imshow(base, cmap=ListedColormap(["#242424", "#f7f7f2"]), vmin=0, vmax=1)
    if len(doors) > 0:
        closed = doors[~door_open]
        opened = doors[door_open]
        if len(closed) > 0:
            ax.scatter(
                closed[:, 1],
                closed[:, 0],
                marker="s",
                s=42,
                c="#d94b4b",
                edgecolors="black",
                linewidths=0.35,
                label="gate closed at t=0",
            )
        if len(opened) > 0:
            ax.scatter(
                opened[:, 1],
                opened[:, 0],
                marker="s",
                s=42,
                c="#31a66a",
                edgecolors="black",
                linewidths=0.35,
                label="gate open at t=0",
            )
        for i, (r, c) in enumerate(doors):
            ax.text(
                int(c),
                int(r),
                str(i),
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
                fontweight="bold",
            )

    ax.scatter([start[1]], [start[0]], marker="o", s=58, c="#3274d9", edgecolors="black")
    ax.scatter([goal[1]], [goal[0]], marker="*", s=92, c="#f0cf3a", edgecolors="black")
    ax.set_title(
        f"seed={seed} | gates={len(doors)} | open@t0={int(door_open.sum())}/{len(doors)}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return {
        "seed": seed,
        "num_gates": int(len(doors)),
        "open_at_t0": int(door_open.sum()),
        "closed_at_t0": int(len(doors) - door_open.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/dynamic_maze_visualizations")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-layouts", type=int, default=6)
    parser.add_argument("--num-doors", type=int, default=8)
    parser.add_argument("--door-open-prob", type=float, default=0.35)
    parser.add_argument("--door-toggle-prob", type=float, default=0.04)
    parser.add_argument("--n-steps", type=int, default=65)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DynamicMazeDatasetConfig(
        num_doors=args.num_doors,
        door_initial_open_prob=args.door_open_prob,
        door_toggle_prob=args.door_toggle_prob,
        n_steps=args.n_steps,
        normalize=False,
    )

    ncols = 3
    nrows = int(np.ceil(args.num_layouts / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.35 * nrows), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    rows = []
    for i in range(args.num_layouts):
        rows.append(draw_layout(axes[i], cfg, args.seed + i))
    for ax in axes[args.num_layouts :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Dynamic maze layouts without fog | "
        f"p(open at reset)={args.door_open_prob:.2f}, "
        f"p(toggle per move)={args.door_toggle_prob:.2f}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    layout_path = out_dir / "gate_layouts_no_fog.png"
    fig.savefig(layout_path)
    plt.close(fig)

    steps = np.arange(args.n_steps + 1)
    probs = gate_open_probability(args.door_open_prob, args.door_toggle_prob, steps)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    ax.plot(steps, probs, color="#2d6cdf", linewidth=2.2)
    ax.axhline(0.5, color="#444444", linewidth=1, linestyle="--", alpha=0.7)
    for step in (0, 2, 4, 17, args.n_steps):
        ax.scatter([step], [probs[step]], color="#d94b4b", s=28, zorder=3)
        ax.text(step, probs[step] + 0.012, f"{100 * probs[step]:.1f}%", ha="center", fontsize=8)
    ax.set_title("Gate open probability over a rollout")
    ax.set_xlabel("move")
    ax.set_ylabel("P(gate open)")
    ax.set_ylim(0.30, 0.53)
    ax.grid(alpha=0.22)
    prob_path = out_dir / "gate_open_probability_curve.png"
    fig.tight_layout()
    fig.savefig(prob_path)
    plt.close(fig)

    with (out_dir / "gate_layout_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "num_gates", "open_at_t0", "closed_at_t0"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "gate_open_probability.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["step", "p_open"])
        writer.writerows((int(step), float(prob)) for step, prob in zip(steps, probs))

    print(layout_path)
    print(prob_path)


if __name__ == "__main__":
    main()
