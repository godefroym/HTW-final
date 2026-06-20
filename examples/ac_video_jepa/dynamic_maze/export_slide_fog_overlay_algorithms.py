"""Export slide-ready full-maze/fog GIFs for Dynamic Maze algorithms.

This is a visualization-only runner. It replays already trained controllers and
renders the whole maze, tinting cells outside the current fog-of-war window.
It does not retrain models or change checkpoints.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.dynamic_maze.dynamic_maze import (
    DynamicMazeDatasetConfig,
    DynamicMazeEnv,
    _fog_mask,
)
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.hierarchical import CARDINALS, SubgoalPredictor, fine_kstep_target
from eb_jepa.state_decoder import (
    BeliefLSTM,
    BeliefValueHead,
    GoalValueHead,
    MLPXYHead,
)
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


DEFAULT_BOOTSTRAP_CKPT = (
    "/lustre/work/vivatech-goma/gmeynard/runs/"
    "dynamic_maze_bootstrap_20260620_051011/00_bootstrap_wm_value/latest.pth.tar"
)
DEFAULT_BELIEF_CKPT = (
    "/lustre/work/vivatech-goma/gmeynard/runs/"
    "dynamic_maze_belief_lstm_20260620_085948/00_belief_value/latest.pth.tar"
)
DEFAULT_SUBGOAL_FINE_CKPT = (
    "/lustre/work/vivatech-goma/gmeynard/runs/"
    "dynamic_maze_jepa_subgoal_starter_20260620_034609/00_wm_astar/latest.pth.tar"
)
DEFAULT_SUBGOAL_N2_CKPT = (
    "/lustre/work/vivatech-goma/gmeynard/runs/"
    "dynamic_maze_jepa_subgoal_starter_20260620_034609/01_subgoal_N2/"
    "subgoal.pth.tar"
)
DEFAULT_SUBGOAL_N4_CKPT = (
    "/lustre/work/vivatech-goma/gmeynard/runs/"
    "dynamic_maze_jepa_subgoal_starter_20260620_034609/01_subgoal_N4/"
    "subgoal.pth.tar"
)


BOOTSTRAP_METHODS = {"learned_value", "probe_pos", "repr_dist"}
BELIEF_METHODS = {"belief_value"}
SUBGOAL_METHODS = {"jepa_subgoal_n2", "jepa_subgoal_n4"}
ALL_METHODS = (
    "repr_dist",
    "learned_value",
    "probe_pos",
    "belief_value",
    "jepa_subgoal_n2",
    "jepa_subgoal_n4",
)


def obs_tensor(obs, norm, device):
    return norm.normalize_state(
        obs.to(dtype=torch.float32, device=device)
    ).unsqueeze(0).unsqueeze(2)


def cfg_env_from_checkpoint(ckpt):
    cfg = OmegaConf.load(Path(ckpt).parent / "config.yaml")
    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    cfg_data.pop("pipeline", None)
    _, _, env_config, _ = init_data(cfg.data.env_name, cfg_data=cfg_data)
    return cfg, env_config


def make_env(cfg, env_config, max_steps):
    return create_env(
        cfg.data.env_name,
        config=env_config,
        n_allowed_steps=max_steps + 1,
        n_steps=max_steps + 1,
        max_step_norm=1.5,
    )


def cell_from_pixel(env, pixel):
    if isinstance(pixel, torch.Tensor):
        pixel = pixel.detach().cpu().numpy()
    off = (env.config.cell_size - 1) / 2.0
    cell = np.rint((np.asarray(pixel, dtype=np.float32) - off) / env.config.cell_size)
    return np.clip(
        cell,
        [0, 0],
        [env.config.maze_height - 1, env.config.maze_width - 1],
    ).astype(np.int32)


def render_fullmaze_fog(env, scale=18):
    current = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
    visible = _fog_mask(
        current.shape,
        env.agent_cell,
        env.config.fog_radius,
        env.config.fog_metric,
    )
    h, w = current.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    free = np.array([232, 235, 226], dtype=np.float32)
    wall = np.array([68, 72, 84], dtype=np.float32)
    closed_door = np.array([55, 102, 210], dtype=np.float32)
    open_door = np.array([44, 178, 122], dtype=np.float32)
    fog_tint = np.array([72, 90, 112], dtype=np.float32)
    trail = np.array([255, 150, 55], dtype=np.float32)
    start = np.array([145, 86, 235], dtype=np.float32)
    goal = np.array([246, 214, 60], dtype=np.float32)
    agent = np.array([232, 55, 55], dtype=np.float32)

    rgb[current == 1] = free
    rgb[current == 0] = wall
    for cell, is_open in zip(env.door_cells, env.door_open):
        rgb[int(cell[0]), int(cell[1])] = open_door if bool(is_open) else closed_door

    rgb[~visible] = 0.45 * rgb[~visible] + 0.55 * fog_tint

    for pixel in env.position_history:
        r, c = cell_from_pixel(env, pixel)
        rgb[int(r), int(c)] = 0.45 * rgb[int(r), int(c)] + 0.55 * trail

    rgb[1, 1] = start
    rgb[int(env.goal_cell[0]), int(env.goal_cell[1])] = goal
    rgb[int(env.agent_cell[0]), int(env.agent_cell[1])] = agent

    frame = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    line = np.array([38, 42, 50], dtype=np.float32)
    frame[::scale, :, :] = line
    frame[:, ::scale, :] = line
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_gif(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)


def episode_result(method, attempt, success, moves, env, frames, extra=None):
    final_manhattan = int(np.abs(env.agent_cell - env.goal_cell).sum())
    row = {
        "method": method,
        "attempt": int(attempt),
        "success": bool(success),
        "moves": int(moves),
        "blocked_moves": int(env.blocked_moves),
        "final_manhattan": final_manhattan,
        "frames": int(len(frames)),
    }
    if extra:
        row.update(extra)
    return row


def choose_representative(method, episodes, out_dir, fps):
    if not episodes:
        raise RuntimeError(f"{method}: no episode generated")
    chosen = next((item for item in episodes if item["row"]["success"]), episodes[0])
    label = "positive" if chosen["row"]["success"] else "negative"
    gif_name = f"{method}_{label}_fullmaze_fog_overlay.gif"
    gif_path = out_dir / gif_name
    save_gif(gif_path, chosen["frames"], fps=fps)
    chosen["row"]["gif"] = gif_name
    return chosen["row"]


def run_bootstrap_method(args, method, device, out_dir):
    cfg, env_config = cfg_env_from_checkpoint(args.bootstrap_ckpt)
    cell_size = float(env_config.cell_size)
    jepa, fdim = build_fine(cfg, env_config, device)
    ckpt_info = load_checkpoint(
        Path(args.bootstrap_ckpt),
        jepa,
        optimizer=None,
        scheduler=None,
        device=device,
        strict=False,
    )
    jepa.eval()
    for param in jepa.parameters():
        param.requires_grad_(False)

    xy_head = MLPXYHead(input_shape=fdim, normalizer=None).to(device)
    if "xy_head_state_dict" in ckpt_info:
        xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])
    xy_head.eval()

    value_head = GoalValueHead(fdim).to(device)
    if method == "learned_value" and "value_head_state_dict" not in ckpt_info:
        raise ValueError(f"{args.bootstrap_ckpt} has no value_head_state_dict")
    if "value_head_state_dict" in ckpt_info:
        value_head.load_state_dict(ckpt_info["value_head_state_dict"])
    value_head.eval()

    env = make_env(cfg, env_config, args.max_steps)
    norm = env.normalizer
    xy_head.normalizer = norm
    off = (cell_size - 1) / 2.0
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}

    def pred_cell(z):
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (
            int(round((float(xy[0]) - off) / cell_size)),
            int(round((float(xy[1]) - off) / cell_size)),
        )

    attempts = []
    for attempt in range(args.search_episodes):
        obs, info = env.reset()
        goal_img = info["target_obs"]
        goal_enc = jepa.encode(obs_tensor(goal_img.detach().clone(), norm, device))
        goal_xy = norm.normalize_location(
            info["target_position"].to(dtype=torch.float32, device=device).unsqueeze(0)
        )[0]
        frames = [render_fullmaze_fog(env, args.scale)]
        blocked = {}
        visits = {}
        last_reverse = -1
        success = False
        moves = 0

        for _ in range(args.max_steps):
            ot = obs_tensor(obs, norm, device)
            ot4 = ot.repeat(4, 1, 1, 1, 1)
            zf = fine_kstep_target(
                jepa,
                ot4,
                torch.arange(4, device=device),
                args.lookahead,
                cell_size,
            )

            if method == "learned_value":
                values = value_head(zf.float(), goal_enc.float()).squeeze(1)
                scores = (-values).detach().cpu().numpy()
            elif method == "probe_pos":
                xy = xy_head(zf.float()).permute(0, 2, 1)[:, 0]
                scores = ((xy - goal_xy.unsqueeze(0)) ** 2).mean(dim=-1)
                scores = scores.detach().cpu().numpy()
            elif method == "repr_dist":
                target = goal_enc.expand(4, -1, zf.shape[2], -1, -1)
                scores = ((zf.float() - target.float()) ** 2).mean(dim=(1, 2, 3, 4))
                scores = scores.detach().cpu().numpy()
            else:
                raise ValueError(f"unknown bootstrap method: {method}")

            cell = tuple(int(c) for c in env.agent_cell)
            visits[cell] = visits.get(cell, 0) + 1
            if args.revisit_penalty > 0:
                for direction in range(4):
                    scores[direction] += args.revisit_penalty * visits.get(
                        pred_cell(zf[direction : direction + 1]), 0
                    )

            order = sorted(range(4), key=lambda direction: float(scores[direction]))
            candidates = [
                direction
                for direction in order
                if direction not in blocked.get(cell, set()) and direction != last_reverse
            ]
            candidates += [direction for direction in order if direction not in candidates]

            moved = False
            done = False
            trunc = False
            for direction in candidates:
                prev = env.agent_cell.copy()
                action = (CARDINALS[direction] * cell_size).cpu().numpy()
                obs, _, done, trunc, _ = env.step(action)
                if not np.array_equal(env.agent_cell, prev):
                    moved = True
                    last_reverse = opposite[direction]
                    moves += 1
                    frames.append(render_fullmaze_fog(env, args.scale))
                    break
                blocked.setdefault(cell, set()).add(direction)
                if done or trunc:
                    break
            if done or np.array_equal(env.agent_cell, env.goal_cell):
                success = True
                break
            if trunc or not moved:
                break

        row = episode_result(
            method,
            attempt,
            success,
            moves,
            env,
            frames,
            extra={"lookahead": int(args.lookahead), "astar_free": True},
        )
        attempts.append({"row": row, "frames": frames})
        print(
            f"[slide-gifs] {method} attempt={attempt} "
            f"{'SUCCESS' if success else 'fail'} moves={moves} "
            f"blocked={env.blocked_moves}",
            flush=True,
        )
        if success and args.prefer_success:
            break

    return choose_representative(method, attempts, out_dir, args.fps)


def repeat_hidden(hidden, batch_size):
    if hidden is None:
        return None
    return tuple(h.repeat(1, batch_size, 1).contiguous() for h in hidden)


def run_belief_method(args, device, out_dir):
    method = "belief_value"
    ckpt = Path(args.belief_ckpt)
    cfg, env_config = cfg_env_from_checkpoint(ckpt)
    cell_size = float(env_config.cell_size)
    jepa, fdim = build_fine(cfg, env_config, device)
    ckpt_info = load_checkpoint(
        ckpt,
        jepa,
        optimizer=None,
        scheduler=None,
        device=device,
        strict=False,
    )
    jepa.eval()
    for param in jepa.parameters():
        param.requires_grad_(False)

    hidden_dim = int(ckpt_info.get("belief_hidden_dim", 256))
    num_layers = int(ckpt_info.get("belief_num_layers", 1))
    belief = BeliefLSTM(
        fdim,
        action_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    belief.load_state_dict(ckpt_info["belief_lstm_state_dict"])
    belief.eval()

    value_head = BeliefValueHead(hidden_dim, fdim).to(device)
    value_head.load_state_dict(ckpt_info["belief_value_head_state_dict"])
    value_head.eval()

    xy_head = MLPXYHead(input_shape=fdim, normalizer=None).to(device)
    if "xy_head_state_dict" in ckpt_info:
        xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])
    xy_head.eval()

    env = make_env(cfg, env_config, args.max_steps)
    norm = env.normalizer
    xy_head.normalizer = norm
    off = (cell_size - 1) / 2.0
    action_vecs = CARDINALS.to(device) * cell_size
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}

    def encode_obs(obs):
        return jepa.encode(obs_tensor(obs, norm, device)).float()

    def pred_cell(z):
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (
            int(round((float(xy[0]) - off) / cell_size)),
            int(round((float(xy[1]) - off) / cell_size)),
        )

    attempts = []
    for attempt in range(args.search_episodes):
        obs, info = env.reset()
        goal_enc = encode_obs(info["target_obs"].detach().clone())
        z0 = encode_obs(obs)
        _, hidden = belief.step(z0, torch.zeros(1, 2, device=device))

        frames = [render_fullmaze_fog(env, args.scale)]
        blocked = {}
        visits = {}
        last_reverse = -1
        success = False
        moves = 0

        for _ in range(args.max_steps):
            ot = obs_tensor(obs, norm, device)
            ot4 = ot.repeat(4, 1, 1, 1, 1)
            zf = fine_kstep_target(
                jepa,
                ot4,
                torch.arange(4, device=device),
                args.lookahead,
                cell_size,
            ).float()
            h_rep = repeat_hidden(hidden, 4)
            h_cand, _ = belief.step(zf, action_vecs, h_rep)
            values = value_head(h_cand.float(), goal_enc.float()).squeeze(1)
            scores = (-values).detach().cpu().numpy()

            cell = tuple(int(c) for c in env.agent_cell)
            visits[cell] = visits.get(cell, 0) + 1
            if args.revisit_penalty > 0:
                for direction in range(4):
                    scores[direction] += args.revisit_penalty * visits.get(
                        pred_cell(zf[direction : direction + 1]), 0
                    )

            order = sorted(range(4), key=lambda direction: float(scores[direction]))
            candidates = [
                direction
                for direction in order
                if direction not in blocked.get(cell, set()) and direction != last_reverse
            ]
            candidates += [direction for direction in order if direction not in candidates]

            moved = False
            done = False
            trunc = False
            for direction in candidates:
                prev = env.agent_cell.copy()
                action = action_vecs[direction].detach().cpu().numpy()
                obs, _, done, trunc, _ = env.step(action)
                z_obs = encode_obs(obs)
                _, hidden = belief.step(
                    z_obs,
                    action_vecs[direction : direction + 1],
                    hidden,
                )
                if not np.array_equal(env.agent_cell, prev):
                    moved = True
                    last_reverse = opposite[direction]
                    moves += 1
                    frames.append(render_fullmaze_fog(env, args.scale))
                    break
                blocked.setdefault(cell, set()).add(direction)
                if done or trunc:
                    break
            if done or np.array_equal(env.agent_cell, env.goal_cell):
                success = True
                break
            if trunc or not moved:
                break

        row = episode_result(
            method,
            attempt,
            success,
            moves,
            env,
            frames,
            extra={
                "lookahead": int(args.lookahead),
                "belief_hidden_dim": int(hidden_dim),
                "astar_free": True,
            },
        )
        attempts.append({"row": row, "frames": frames})
        print(
            f"[slide-gifs] {method} attempt={attempt} "
            f"{'SUCCESS' if success else 'fail'} moves={moves} "
            f"blocked={env.blocked_moves}",
            flush=True,
        )
        if success and args.prefer_success:
            break

    return choose_representative(method, attempts, out_dir, args.fps)


def run_subgoal_method(args, method, device, out_dir):
    fine_ckpt = Path(args.subgoal_fine_ckpt)
    sg_ckpt = Path(args.subgoal_n2_ckpt if method.endswith("n2") else args.subgoal_n4_ckpt)
    cfg, env_config = cfg_env_from_checkpoint(fine_ckpt)
    cell_size = float(env_config.cell_size)
    jepa, fdim = build_fine(cfg, env_config, device)
    info = load_checkpoint(
        fine_ckpt,
        jepa,
        optimizer=None,
        scheduler=None,
        device=device,
        strict=False,
    )
    jepa.eval()
    for param in jepa.parameters():
        param.requires_grad_(False)

    sck = torch.load(sg_ckpt, map_location=device, weights_only=False)
    subgoal = SubgoalPredictor(fdim).to(device)
    subgoal.load_state_dict(sck["subgoal"])
    subgoal.eval()
    n_value = int(sck["N"])
    lookahead = n_value if args.subgoal_lookahead == "auto" else int(args.subgoal_lookahead)

    env = make_env(cfg, env_config, 10_000)
    norm = env.normalizer
    xy_head = MLPXYHead(input_shape=fdim, normalizer=norm).to(device)
    if "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    xy_head.eval()
    off = (cell_size - 1) / 2.0
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}

    def probe_xy(z):
        return xy_head(z.float()).permute(0, 2, 1)[0, 0]

    def pred_cell(z):
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (
            int(round((float(xy[0]) - off) / cell_size)),
            int(round((float(xy[1]) - off) / cell_size)),
        )

    attempts = []
    for attempt in range(args.search_episodes):
        obs, info_e = env.reset()
        goal_xy = norm.normalize_location(
            info_e["target_position"].to(dtype=torch.float32, device=device).unsqueeze(0)
        )[0]
        solved = solve_a_star(
            info_e["current_grid"],
            tuple(int(c) for c in env.agent_cell),
            tuple(int(c) for c in env.goal_cell),
        )
        initial_len = len(solved[0]) - 1 if solved is not None else 100
        max_steps = min(
            int(args.budget_factor * initial_len + args.budget_margin),
            args.max_steps,
        )

        frames = [render_fullmaze_fog(env, args.scale)]
        blocked = {}
        visits = {}
        last_reverse = -1
        success = False
        moves = 0

        for _ in range(max_steps):
            ot = obs_tensor(obs, norm, device)
            z = jepa.encode(ot)
            sg = subgoal(z, goal_xy.unsqueeze(0))[0]
            cell = tuple(int(c) for c in env.agent_cell)
            visits[cell] = visits.get(cell, 0) + 1
            distances = []
            for direction in range(4):
                zf = fine_kstep_target(
                    jepa,
                    ot,
                    torch.tensor([direction], device=device),
                    lookahead,
                    cell_size,
                )
                score = float(torch.norm(probe_xy(zf) - sg).item())
                if args.revisit_penalty > 0:
                    score += args.revisit_penalty * visits.get(pred_cell(zf), 0)
                distances.append(score)

            order = sorted(range(4), key=lambda direction: distances[direction])
            candidates = [
                direction
                for direction in order
                if direction not in blocked.get(cell, set()) and direction != last_reverse
            ]
            candidates += [direction for direction in order if direction not in candidates]

            moved = False
            done = False
            trunc = False
            for direction in candidates:
                prev = env.agent_cell.copy()
                action = (CARDINALS[direction] * cell_size).cpu().numpy()
                obs, _, done, trunc, _ = env.step(action)
                if not np.array_equal(env.agent_cell, prev):
                    moved = True
                    last_reverse = opposite[direction]
                    moves += 1
                    frames.append(render_fullmaze_fog(env, args.scale))
                    break
                blocked.setdefault(cell, set()).add(direction)
                if done or trunc:
                    break
            if done or np.array_equal(env.agent_cell, env.goal_cell):
                success = True
                break
            if trunc or not moved:
                break

        row = episode_result(
            method,
            attempt,
            success,
            moves,
            env,
            frames,
            extra={
                "N": int(n_value),
                "lookahead": int(lookahead),
                "initial_astar_len": int(initial_len),
                "efficiency": float(initial_len / max(moves, initial_len))
                if success
                else 0.0,
                "astar_free": True,
            },
        )
        attempts.append({"row": row, "frames": frames})
        print(
            f"[slide-gifs] {method} attempt={attempt} "
            f"{'SUCCESS' if success else 'fail'} moves={moves} "
            f"A0={initial_len} blocked={env.blocked_moves}",
            flush=True,
        )
        if success and args.prefer_success:
            break

    return choose_representative(method, attempts, out_dir, args.fps)


def write_readme(out_dir, rows):
    lines = [
        "# Slide Assets: Dynamic Maze Algorithm Fog Overlay GIFs",
        "",
        "These GIFs replay trained JEPA/WM controllers and render the full maze for",
        "presentation. Cells outside the current observation radius are tinted",
        "blue-grey so the hidden fog-of-war region stays visible on slides.",
        "",
        "Color key:",
        "",
        "- red: agent",
        "- yellow: goal",
        "- orange: visited trail",
        "- green/blue: open/closed stochastic gates",
        "- blue-grey tint: currently outside the fog-of-war observation",
        "",
        "Generated algorithms:",
        "",
    ]
    for row in rows:
        label = "success" if row["success"] else "failure"
        lines.append(
            f"- `{row['method']}`: `{row['gif']}` ({label}, "
            f"moves={row['moves']}, blocked={row['blocked_moves']}, "
            f"final_manhattan={row['final_manhattan']})"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- The controllers are replayed from existing checkpoints; this is not a",
            "  training run.",
            "- A* is not called in the action loop for these JEPA/WM controllers. For",
            "  subgoal GIFs, A* is used only to report the initial budget reference,",
            "  matching the existing evaluation script.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--bootstrap-ckpt", default=DEFAULT_BOOTSTRAP_CKPT)
    parser.add_argument("--belief-ckpt", default=DEFAULT_BELIEF_CKPT)
    parser.add_argument("--subgoal-fine-ckpt", default=DEFAULT_SUBGOAL_FINE_CKPT)
    parser.add_argument("--subgoal-n2-ckpt", default=DEFAULT_SUBGOAL_N2_CKPT)
    parser.add_argument("--subgoal-n4-ckpt", default=DEFAULT_SUBGOAL_N4_CKPT)
    parser.add_argument("--search-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--subgoal-lookahead", default="auto")
    parser.add_argument("--revisit-penalty", type=float, default=0.02)
    parser.add_argument("--budget-factor", type=float, default=6.0)
    parser.add_argument("--budget-margin", type=int, default=20)
    parser.add_argument("--scale", type=int, default=18)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--prefer-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    bad = sorted(set(methods) - set(ALL_METHODS))
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Valid methods: {ALL_METHODS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for method in methods:
        print(f"[slide-gifs] starting {method} on {device}", flush=True)
        if method in BOOTSTRAP_METHODS:
            row = run_bootstrap_method(args, method, device, out_dir)
        elif method in BELIEF_METHODS:
            row = run_belief_method(args, device, out_dir)
        elif method in SUBGOAL_METHODS:
            row = run_subgoal_method(args, method, device, out_dir)
        else:
            raise ValueError(f"unsupported method: {method}")
        rows.append(row)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "methods": rows,
                "fps": float(args.fps),
                "scale": int(args.scale),
                "search_episodes": int(args.search_episodes),
                "max_steps": int(args.max_steps),
                "bootstrap_ckpt": str(args.bootstrap_ckpt),
                "belief_ckpt": str(args.belief_ckpt),
                "subgoal_fine_ckpt": str(args.subgoal_fine_ckpt),
                "subgoal_n2_ckpt": str(args.subgoal_n2_ckpt),
                "subgoal_n4_ckpt": str(args.subgoal_n4_ckpt),
            },
            f,
            indent=2,
        )
    write_readme(out_dir, rows)
    print(f"[slide-gifs] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
