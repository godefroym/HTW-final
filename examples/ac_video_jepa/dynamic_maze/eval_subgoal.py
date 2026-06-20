"""A*-free dynamic-maze eval for a subgoal head distilled from A* trajectories."""

import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.dynamic_maze.dynamic_maze import observation_to_rgb
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.hierarchical import CARDINALS, SubgoalPredictor, fine_kstep_target
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def overlay_goal(frame, env):
    out = frame.copy()
    r, c = env.goal_cell
    cs = env.config.cell_size
    out[int(r * cs) : int((r + 1) * cs), int(c * cs) : int((c + 1) * cs)] = np.array(
        [245, 215, 50], dtype=np.uint8
    )
    return out


@torch.no_grad()
def main():
    fine_ckpt, sg_ckpt, rdir = sys.argv[1], sys.argv[2], sys.argv[3]
    num_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    lookahead = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    revisit_pen = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
    n_gifs = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    budget_factor = float(sys.argv[8]) if len(sys.argv) > 8 else 6.0
    budget_margin = int(sys.argv[9]) if len(sys.argv) > 9 else 20
    os.makedirs(rdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    cfg_data.pop("pipeline", None)
    _, _, env_config, _ = init_data(cfg.data.env_name, cfg_data=cfg_data)
    cell_size = float(env_config.cell_size)
    n_allowed = 10_000

    jepa, f = build_fine(cfg, env_config, device)
    info = load_checkpoint(
        Path(fine_ckpt), jepa, optimizer=None, scheduler=None, device=device,
        strict=False,
    )
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    sck = torch.load(sg_ckpt, map_location=device, weights_only=False)
    subgoal = SubgoalPredictor(f).to(device)
    subgoal.load_state_dict(sck["subgoal"])
    subgoal.eval()

    env = create_env(
        cfg.data.env_name,
        config=env_config,
        n_allowed_steps=n_allowed,
        n_steps=n_allowed,
        max_step_norm=1.5,
    )
    norm = env.normalizer
    xy_head = MLPXYHead(input_shape=f, normalizer=norm).to(device)
    if "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    xy_head.eval()

    def obs_tensor(obs):
        return norm.normalize_state(
            obs.to(dtype=torch.float32, device=device)
        ).unsqueeze(0).unsqueeze(2)

    off = (cell_size - 1) / 2.0

    def probe_xy(z):
        return xy_head(z.float()).permute(0, 2, 1)[0, 0]

    def pred_cell(z):
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (
            int(round((float(xy[0]) - off) / cell_size)),
            int(round((float(xy[1]) - off) / cell_size)),
        )

    print(
        f"[dyn-subgoal-eval] N={sck['N']} episodes={num_ep} "
        f"lookahead={lookahead} budget={budget_factor}xA0+{budget_margin}",
        flush=True,
    )

    successes, efficiencies, steps_all, blocked_all = [], [], [], []
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    for ep in range(num_ep):
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
        max_steps = min(int(budget_factor * initial_len + budget_margin), n_allowed)

        frames = [overlay_goal(observation_to_rgb(obs), env)]
        blocked = {}
        visits = {}
        last_reverse = -1
        success = False
        n_moves = 0

        for _ in range(max_steps):
            ot = obs_tensor(obs)
            z = jepa.encode(ot)
            sg = subgoal(z, goal_xy.unsqueeze(0))[0]
            cell = tuple(int(c) for c in env.agent_cell)
            visits[cell] = visits.get(cell, 0) + 1
            dist = []
            for d in range(4):
                zf = fine_kstep_target(
                    jepa, ot, torch.tensor([d], device=device), lookahead, cell_size
                )
                score = float(torch.norm(probe_xy(zf) - sg).item())
                if revisit_pen > 0:
                    score += revisit_pen * visits.get(pred_cell(zf), 0)
                dist.append(score)
            order = sorted(range(4), key=lambda d: dist[d])
            candidates = [
                d for d in order
                if d not in blocked.get(cell, set()) and d != last_reverse
            ]
            candidates += [d for d in order if d not in candidates]

            moved = False
            done = False
            for d in candidates:
                prev = env.agent_cell.copy()
                action = (CARDINALS[d] * cell_size).cpu().numpy()
                obs, _, done, trunc, _ = env.step(action)
                if not np.array_equal(env.agent_cell, prev):
                    moved = True
                    last_reverse = opposite[d]
                    n_moves += 1
                    if ep < n_gifs:
                        frames.append(overlay_goal(observation_to_rgb(obs), env))
                    break
                blocked.setdefault(cell, set()).add(d)
                if done or trunc:
                    break
            if done:
                success = True
                break
            if not moved:
                break

        successes.append(float(success))
        efficiencies.append((initial_len / max(n_moves, initial_len)) if success else 0.0)
        steps_all.append(n_moves)
        blocked_all.append(int(env.blocked_moves))
        if ep < n_gifs:
            label = "succ" if success else "fail"
            imageio.mimsave(os.path.join(rdir, f"ep{ep}_{label}.gif"), frames, fps=8, loop=0)
        print(
            f"[dyn-subgoal-eval] ep={ep} {'SUCCESS' if success else 'fail'} "
            f"moves={n_moves} A0={initial_len} blocked={env.blocked_moves}",
            flush=True,
        )

    result = {
        "success_rate": float(np.mean(successes)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "mean_steps": float(np.mean(steps_all)),
        "mean_blocked_moves": float(np.mean(blocked_all)),
        "num_episodes": num_ep,
        "N": int(sck["N"]),
        "block_penalty": float(sck.get("block_penalty", 0.0)),
        "block_radius_px": int(sck.get("block_radius_px", 0)),
        "lookahead": lookahead,
        "revisit_pen": revisit_pen,
        "budget": f"{budget_factor}xA*_initial+{budget_margin}",
        "astar_free": True,
    }
    with open(os.path.join(rdir, "subgoal_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[dyn-subgoal-eval] N={sck['N']} success={100*result['success_rate']:.1f}% "
        f"eff={result['mean_efficiency']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
