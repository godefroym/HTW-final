"""Evaluate a Dynamic Maze vision JEPA with representation-distance planning.

The policy is A*-free at evaluation time. For each cardinal action, it rolls the
image-trained world model K steps and chooses the action whose predicted latent
is closest to the latent of the goal-view observation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.dynamic_maze.vision_renderer import VisionRenderConfig
from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.hierarchical import CARDINALS, fine_kstep_target
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.dynamic_maze.render_vision_poc import make_demo_frame
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


@torch.no_grad()
def main() -> None:
    ckpt = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    num_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    lookahead = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    n_gifs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 800
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(ckpt.parent / "config.yaml")
    if cfg.data.get("observation_mode") != "vision":
        raise ValueError(
            f"{ckpt} was not trained with data.observation_mode=vision "
            f"(got {cfg.data.get('observation_mode')!r})"
        )

    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    cfg_data.pop("pipeline", None)
    _, _, env_config, _ = init_data(cfg.data.env_name, cfg_data=cfg_data)
    cell_size = float(env_config.cell_size)

    jepa, _ = build_fine(cfg, env_config, device)
    load_checkpoint(ckpt, jepa, optimizer=None, scheduler=None, device=device, strict=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    env = create_env(
        cfg.data.env_name,
        config=env_config,
        n_allowed_steps=max_steps + 1,
        n_steps=max_steps + 1,
        max_step_norm=1.5,
    )
    norm = env.normalizer
    render_cfg = VisionRenderConfig(
        image_size=int(env_config.img_size),
        fov_degrees=float(env_config.vision_fov_degrees),
        vertical_fov_degrees=float(env_config.vision_vertical_fov_degrees),
        max_depth=float(env_config.vision_max_depth),
        goal_marker=bool(env_config.vision_goal_marker),
    )
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}

    def obs_tensor(obs: torch.Tensor) -> torch.Tensor:
        return norm.normalize_state(
            obs.to(dtype=torch.float32, device=device)
        ).unsqueeze(0).unsqueeze(2)

    successes, steps_all, blocked_all, manhattan_all = [], [], [], []
    print(
        f"[vision-reprdist] episodes={num_ep} lookahead={lookahead} "
        f"max_steps={max_steps} image_size={env_config.img_size}",
        flush=True,
    )

    for ep in range(num_ep):
        obs, info = env.reset()
        goal_enc = jepa.encode(obs_tensor(info["target_obs"].detach().clone()))
        frames = [np.asarray(make_demo_frame(env, render_cfg))]
        blocked = {}
        visits = {}
        last_reverse = -1
        success = False
        n_moves = 0

        for _ in range(max_steps):
            ot = obs_tensor(obs)
            zf = fine_kstep_target(
                jepa,
                ot.repeat(4, 1, 1, 1, 1),
                torch.arange(4, device=device),
                lookahead,
                cell_size,
            )
            target = goal_enc.expand(4, -1, zf.shape[2], -1, -1)
            scores = ((zf.float() - target.float()) ** 2).mean(dim=(1, 2, 3, 4))
            scores = scores.detach().cpu().numpy()

            cell = tuple(int(c) for c in env.agent_cell)
            visits[cell] = visits.get(cell, 0) + 1
            for d in range(4):
                dr, dc = CARDINALS[d].detach().cpu().numpy().astype(np.int32)
                nxt = (cell[0] + int(dr), cell[1] + int(dc))
                scores[d] += 0.015 * visits.get(nxt, 0)

            order = sorted(range(4), key=lambda d: float(scores[d]))
            candidates = [
                d for d in order
                if d not in blocked.get(cell, set()) and d != last_reverse
            ]
            candidates += [d for d in order if d not in candidates]

            moved = False
            done = False
            truncated = False
            for d in candidates:
                prev = env.agent_cell.copy()
                action = (CARDINALS[d] * cell_size).cpu().numpy()
                obs, _, done, truncated, _ = env.step(action)
                if not np.array_equal(env.agent_cell, prev):
                    moved = True
                    last_reverse = opposite[d]
                    n_moves += 1
                    if ep < n_gifs:
                        frames.append(np.asarray(make_demo_frame(env, render_cfg)))
                    break
                blocked.setdefault(cell, set()).add(d)
                if done or truncated:
                    break

            if done or np.array_equal(env.agent_cell, env.goal_cell):
                success = True
                break
            if truncated or not moved:
                break

        successes.append(float(success))
        steps_all.append(n_moves)
        blocked_all.append(int(env.blocked_moves))
        manhattan = int(np.abs(env.agent_cell - env.goal_cell).sum())
        manhattan_all.append(manhattan)
        if ep < n_gifs:
            label = "succ" if success else "fail"
            imageio.mimsave(
                out_dir / f"ep{ep}_vision_reprdist_{label}.gif",
                frames,
                fps=8,
                loop=0,
            )
        print(
            f"[vision-reprdist] ep={ep} {'SUCCESS' if success else 'fail'} "
            f"moves={n_moves} blocked={env.blocked_moves} manhattan={manhattan}",
            flush=True,
        )

    result = {
        "objective": "vision_repr_dist",
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps_all)),
        "mean_blocked_moves": float(np.mean(blocked_all)),
        "mean_final_manhattan": float(np.mean(manhattan_all)),
        "num_episodes": num_ep,
        "lookahead": lookahead,
        "max_steps": max_steps,
        "image_size": int(env_config.img_size),
        "observation_mode": "vision",
        "astar_free": True,
    }
    with (out_dir / "vision_reprdist_eval.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[vision-reprdist] success={100 * result['success_rate']:.1f}% "
        f"steps={result['mean_steps']:.1f} blocked={result['mean_blocked_moves']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
