"""Render a lightweight 3D-vision POC for Dynamic Maze.

This script is a demo artifact, not a full training pipeline. It shows how a
DynamicMazeEnv state can be converted on the fly into a 4-channel perceptual
observation: up, down, left, right grayscale perspective views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from eb_jepa.datasets.dynamic_maze.dynamic_maze import (
    DynamicMazeDatasetConfig,
    DynamicMazeEnv,
)
from eb_jepa.datasets.dynamic_maze.vision_renderer import (
    VIEW_NAMES,
    VisionRenderConfig,
    render_four_views,
)


def _scale_nearest(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.Resampling.NEAREST)


def render_topdown(env: DynamicMazeEnv, scale: int = 12) -> Image.Image:
    grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[grid == 0] = np.array([58, 62, 68], dtype=np.uint8)
    rgb[grid == 1] = np.array([224, 226, 220], dtype=np.uint8)

    for idx, (r, c) in enumerate(env.door_cells):
        color = np.array([61, 163, 109], dtype=np.uint8)
        if not bool(env.door_open[idx]):
            color = np.array([70, 105, 210], dtype=np.uint8)
        rgb[int(r), int(c)] = color

    sr, sc = 1, 1
    gr, gc = [int(v) for v in env.goal_cell.tolist()]
    ar, ac = [int(v) for v in env.agent_cell.tolist()]
    rgb[sr, sc] = np.array([230, 80, 65], dtype=np.uint8)
    rgb[gr, gc] = np.array([245, 220, 55], dtype=np.uint8)
    rgb[ar, ac] = np.array([20, 20, 20], dtype=np.uint8)

    img = Image.fromarray(rgb).resize((w * scale, h * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    for r in range(h + 1):
        y = r * scale
        draw.line([(0, y), (w * scale, y)], fill=(120, 120, 120), width=1)
    for c in range(w + 1):
        x = c * scale
        draw.line([(x, 0), (x, h * scale)], fill=(120, 120, 120), width=1)
    return img


def make_views_mosaic(views: np.ndarray, tile_size: int = 160) -> Image.Image:
    canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    positions = {
        "up": (0, 0),
        "down": (tile_size, 0),
        "left": (0, tile_size),
        "right": (tile_size, tile_size),
    }
    for idx, name in enumerate(VIEW_NAMES):
        tile = Image.fromarray(views[idx]).convert("RGB")
        tile = _scale_nearest(tile, tile_size)
        x, y = positions[name]
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_size - 1, y + 17), fill=(0, 0, 0))
        draw.text((x + 6, y + 3), name, fill=(245, 245, 245))
    return canvas


def make_demo_frame(env: DynamicMazeEnv, render_cfg: VisionRenderConfig) -> Image.Image:
    grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
    views = render_four_views(
        grid,
        env.agent_cell,
        goal_cell=env.goal_cell,
        door_cells=env.door_cells,
        config=render_cfg,
    )
    topdown = render_topdown(env, scale=12)
    mosaic = make_views_mosaic(views, tile_size=160)

    height = max(topdown.height, mosaic.height)
    canvas = Image.new("RGB", (topdown.width + mosaic.width + 18, height + 28), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), "dynamic maze top-down", fill=(235, 235, 235))
    draw.text((topdown.width + 18, 5), "egocentric 4-channel vision", fill=(235, 235, 235))
    canvas.paste(topdown, (0, 28))
    canvas.paste(mosaic, (topdown.width + 18, 28))
    return canvas


def write_readme(out_dir: Path, args: argparse.Namespace, n_frames: int) -> None:
    text = f"""# Dynamic Maze Vision POC

This folder is a lightweight proof of concept for replacing symbolic 2D maze
observations with egocentric grayscale vision.

- renderer: `eb_jepa.datasets.dynamic_maze.vision_renderer`
- observation shape: `[4, {args.image_size}, {args.image_size}]`
- channel order: `up`, `down`, `left`, `right`
- implementation: analytic grid raycaster, no Chromium, no OpenGL, no Three.js
- episode seed: `{args.seed}`
- rendered frames: `{n_frames}`

The renderer is designed to be called at dataset time from the current dynamic
occupancy grid, agent cell, goal cell, and door cells. It can therefore render
stochastic doors on the fly without storing images for every maze state.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/dynamic_maze_vision_poc_20260620")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-doors", type=int, default=8)
    parser.add_argument("--door-toggle-prob", type=float, default=0.04)
    parser.add_argument("--door-open-prob", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--gif-fps", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DynamicMazeDatasetConfig(
        seed=args.seed,
        num_doors=args.num_doors,
        door_toggle_prob=args.door_toggle_prob,
        door_initial_open_prob=args.door_open_prob,
        normalize=False,
    )
    render_cfg = VisionRenderConfig(image_size=args.image_size)
    env = DynamicMazeEnv(cfg, normalize=False, n_allowed_steps=args.max_steps + 1)
    env.reset()

    grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
    start_views = render_four_views(
        grid,
        env.agent_cell,
        goal_cell=env.goal_cell,
        door_cells=env.door_cells,
        config=render_cfg,
    )
    render_topdown(env).save(out_dir / "topdown_start.png")
    make_views_mosaic(start_views, tile_size=192).save(out_dir / "four_views_start.png")
    make_demo_frame(env, render_cfg).save(out_dir / "demo_frame_start.png")
    np.save(out_dir / "four_views_start.npy", start_views)

    frames = []
    mid_frame = None
    success = False
    for step in range(args.max_steps):
        frame = make_demo_frame(env, render_cfg)
        if step == min(10, args.max_steps - 1):
            mid_frame = frame.copy()
        frames.append(np.asarray(frame))
        if np.array_equal(env.agent_cell, env.goal_cell):
            success = True
            break
        action = env.teacher_action("oracle_replan")
        _, _, done, truncated, _ = env.step(action)
        if done:
            frames.append(np.asarray(make_demo_frame(env, render_cfg)))
            success = True
            break
        if truncated:
            break

    imageio.mimsave(out_dir / "vision_episode.gif", frames, fps=args.gif_fps, loop=0)
    if mid_frame is not None:
        mid_frame.save(out_dir / "demo_frame_mid.png")
    metadata = {
        "seed": args.seed,
        "image_size": args.image_size,
        "num_doors": args.num_doors,
        "door_toggle_prob": args.door_toggle_prob,
        "door_initial_open_prob": args.door_open_prob,
        "frames": len(frames),
        "success": success,
        "blocked_moves": int(env.blocked_moves),
        "final_cell": [int(v) for v in env.agent_cell.tolist()],
        "goal_cell": [int(v) for v in env.goal_cell.tolist()],
        "channel_order": list(VIEW_NAMES),
        "observation_shape": [4, args.image_size, args.image_size],
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    write_readme(out_dir, args, len(frames))
    print(f"[vision-poc] wrote {out_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
