"""Generate quick visual diagnostics for the Two Rooms and Maze environments.

Examples:
  uv run python -m examples.ac_video_jepa.visualize_envs --env both --out-dir ../tmp/env_views
  uv run python -m examples.ac_video_jepa.visualize_envs --env maze --n-samples 6 --seed 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.utils import create_env, init_data, load_env_data_config


def _base_overrides(env_name: str, n_samples: int) -> dict:
    cfg = {
        "batch_size": max(1, n_samples),
        "size": max(16, n_samples * 4),
        "val_size": max(8, n_samples * 2),
        "num_workers": 0,
        "pin_mem": False,
        "persistent_workers": False,
        "device": "cpu",
        "pipeline": {"mode": "online"},
    }
    if env_name == "maze":
        # Keep the default 21x21 task but make generation reliable for tiny samples.
        cfg.update({"max_gen_retries": 128})
    return cfg


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unnormalize_batch(loader, states, locations):
    """Return states as [B,T,C,H,W] in display scale and locations as [B,T,2]."""
    normalizer = loader.dataset.normalizer
    b, c, t, h, w = states.shape
    frames = states.permute(0, 2, 1, 3, 4).contiguous()
    flat = frames.view(b * t, c, h, w)
    frames = normalizer.unnormalize_state(flat).view(b, t, c, h, w)

    loc = locations.permute(0, 2, 1).contiguous()
    loc = normalizer.unnormalize_location(loc)
    return frames.detach().cpu(), loc.detach().cpu()


def _frame_to_rgb(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    """Map a 2-channel observation (dot, walls) to a readable RGB image."""
    arr = _to_numpy(frame).astype(np.float32)
    if arr.shape[0] != 2:
        raise ValueError(f"expected [2,H,W], got {arr.shape}")
    if arr.max() > 2.0:
        arr = arr / 255.0
    dot = np.clip(arr[0], 0.0, 1.0)
    wall = np.clip(arr[1], 0.0, 1.0)

    h, w = dot.shape
    rgb = np.full((h, w, 3), 246, dtype=np.float32)
    wall_alpha = np.clip(wall, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - 0.82 * wall_alpha) + np.array([42, 46, 54]) * (0.82 * wall_alpha)

    if dot.max() > 0:
        dot = dot / (dot.max() + 1e-8)
    dot_alpha = np.clip(dot * 1.4, 0.0, 1.0)[..., None]
    dot_color = np.array([30, 110, 235], dtype=np.float32)
    rgb = rgb * (1.0 - dot_alpha) + dot_color * dot_alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _xy_for_plot(env_name: str, positions: np.ndarray):
    """Convert stored coordinate conventions to imshow x/y coordinates."""
    if env_name == "maze":
        return positions[:, 1], positions[:, 0]  # row,col -> x=col,y=row
    return positions[:, 0], positions[:, 1]      # x,y -> x,y


def _plot_traj(ax, env_name: str, frame, positions, title: str, wall_x=None, door_y=None):
    ax.imshow(_frame_to_rgb(frame), interpolation="nearest")
    x, y = _xy_for_plot(env_name, positions)
    ax.plot(x, y, color="#ffb000", linewidth=2.0, marker="o", markersize=2.5)
    ax.scatter([x[0]], [y[0]], s=55, color="#1aa260", edgecolor="white", linewidth=0.8, label="start")
    ax.scatter([x[-1]], [y[-1]], s=55, color="#d93025", edgecolor="white", linewidth=0.8, label="end")
    if env_name == "two_rooms" and wall_x is not None and door_y is not None:
        ax.axvline(float(np.asarray(wall_x).reshape(-1)[0]), color="white", linewidth=0.8, alpha=0.7)
        ax.scatter(
            [float(np.asarray(wall_x).reshape(-1)[0])],
            [float(np.asarray(door_y).reshape(-1)[0])],
            s=35,
            color="#00acc1",
            edgecolor="white",
            linewidth=0.6,
            label="door",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_dataset_views(env_name: str, out_dir: Path, n_samples: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader, _, _, _ = init_data(
        env_name,
        _base_overrides(env_name, n_samples),
        device=torch.device("cpu"),
    )
    states, actions, locations, wall_x, door_y = next(iter(loader))
    frames, loc = _unnormalize_batch(loader, states, locations)
    n = min(n_samples, frames.shape[0])
    t = frames.shape[1]
    picks = [0, t // 2, t - 1]

    fig, axes = plt.subplots(n, 4, figsize=(9.8, 2.35 * n), squeeze=False)
    for i in range(n):
        for j, idx in enumerate(picks):
            axes[i, j].imshow(_frame_to_rgb(frames[i, idx]), interpolation="nearest")
            axes[i, j].set_title(f"{env_name} sample {i} | t={idx}", fontsize=9)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        _plot_traj(
            axes[i, 3],
            env_name,
            frames[i, 0].clone().mul(torch.tensor([0.0, 1.0]).view(2, 1, 1)),
            loc[i].numpy(),
            "trajectory overlay",
            wall_x=wall_x[i] if env_name == "two_rooms" else None,
            door_y=door_y[i] if env_name == "two_rooms" else None,
        )
    fig.suptitle(f"{env_name}: generated dataset windows", y=0.995, fontsize=12)
    fig.tight_layout()
    grid_path = out_dir / f"{env_name}_dataset_grid.png"
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)

    for i in range(n):
        overlay_path = out_dir / f"{env_name}_trajectory_{i:02d}.png"
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        _plot_traj(
            ax,
            env_name,
            frames[i, 0].clone().mul(torch.tensor([0.0, 1.0]).view(2, 1, 1)),
            loc[i].numpy(),
            f"{env_name} trajectory {i}",
            wall_x=wall_x[i] if env_name == "two_rooms" else None,
            door_y=door_y[i] if env_name == "two_rooms" else None,
        )
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=180)
        plt.close(fig)

        gif_frames = [_frame_to_rgb(frames[i, k]) for k in range(t)]
        imageio.mimsave(out_dir / f"{env_name}_trajectory_{i:02d}.gif", gif_frames, duration=0.12, loop=0)
    return [grid_path]


def _env_config(env_name: str, overrides: dict):
    merged = load_env_data_config(env_name, overrides)
    if env_name == "two_rooms":
        from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig

        return update_config_from_yaml(WallDatasetConfig, merged)
    if env_name == "maze":
        from eb_jepa.datasets.maze.maze_dataset import MazeDatasetConfig

        return update_config_from_yaml(MazeDatasetConfig, merged)
    raise ValueError(env_name)


def _save_planning_scene(env_name: str, out_dir: Path, seed: int):
    cfg = _env_config(env_name, _base_overrides(env_name, 1))
    env = create_env(
        env_name,
        cfg,
        rng=np.random.default_rng(seed),
        n_allowed_steps=200,
        normalize=False,
    )
    obs, info = env.reset()
    target = info["target_obs"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    axes[0].imshow(_frame_to_rgb(obs), interpolation="nearest")
    axes[0].set_title("start observation", fontsize=9)
    axes[1].imshow(_frame_to_rgb(target), interpolation="nearest")
    axes[1].set_title("goal observation", fontsize=9)

    wall_only = obs.detach().clone() if isinstance(obs, torch.Tensor) else torch.tensor(obs)
    wall_only[0].zero_()
    axes[2].imshow(_frame_to_rgb(wall_only), interpolation="nearest")
    if env_name == "maze":
        points = [env.dot_position.detach().cpu().numpy()]
        points += [w.detach().cpu().numpy() for w in env.compute_waypoints(spacing_cells=2)]
        points = np.asarray(points, dtype=np.float32)
        x, y = _xy_for_plot(env_name, points)
        axes[2].plot(x, y, color="#ffb000", linewidth=2.0, marker="o", markersize=3)
        axes[2].set_title("A* waypoint route", fontsize=9)
    else:
        start = env.dot_position.detach().cpu().numpy()
        goal = env.target_position.detach().cpu().numpy()
        door = np.asarray([float(env.wall_x.detach().cpu()), float(env.hole_y.detach().cpu())], dtype=np.float32)
        route = np.vstack([start, door, goal])
        x, y = _xy_for_plot(env_name, route)
        axes[2].plot(x, y, color="#ffb000", linewidth=2.0, marker="o", markersize=3)
        axes[2].scatter([door[0]], [door[1]], s=40, color="#00acc1", edgecolor="white", linewidth=0.6)
        axes[2].set_title("door-aware route sketch", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{env_name}: planning scene", y=0.99, fontsize=12)
    fig.tight_layout()
    out = out_dir / f"{env_name}_planning_scene.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["two_rooms", "maze", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/env_views"))
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    envs = ["two_rooms", "maze"] if args.env == "both" else [args.env]
    written = []
    for env_name in envs:
        written.extend(_save_dataset_views(env_name, args.out_dir, args.n_samples, args.seed))
        written.append(_save_planning_scene(env_name, args.out_dir, args.seed + 17))
    for path in sorted(args.out_dir.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
