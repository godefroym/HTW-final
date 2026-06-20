"""A*-free dynamic-maze eval for a bootstrap WM + hindsight value head.

The evaluated agent does not query A* for actions, subgoals, masks, or budgets.
At each step it:

1. encodes the current observation;
2. rolls the world model K steps for each cardinal action;
3. scores each imagined latent with a learned goal-conditioned value head;
4. executes the best action, with a small revisit penalty to reduce loops.

Run:
  python -m examples.ac_video_jepa.dynamic_maze.eval_bootstrap_value \
    <ckpt> <out_dir> [objective=learned_value] [num_ep=64] [lookahead=4]
"""

import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.dynamic_maze.dynamic_maze import observation_to_rgb
from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.hierarchical import CARDINALS, fine_kstep_target
from eb_jepa.state_decoder import GoalValueHead, MLPXYHead
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
    ckpt = sys.argv[1]
    rdir = sys.argv[2]
    objective = sys.argv[3] if len(sys.argv) > 3 else "learned_value"
    num_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    lookahead = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    revisit_pen = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
    n_gifs = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    max_steps = int(sys.argv[8]) if len(sys.argv) > 8 else 800
    if objective not in {"learned_value", "probe_pos", "repr_dist"}:
        raise ValueError("objective must be learned_value, probe_pos, or repr_dist")
    os.makedirs(rdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(ckpt).parent / "config.yaml")
    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    cfg_data.pop("pipeline", None)
    _, _, env_config, _ = init_data(cfg.data.env_name, cfg_data=cfg_data)
    cell_size = float(env_config.cell_size)

    jepa, fdim = build_fine(cfg, env_config, device)
    ckpt_info = load_checkpoint(
        Path(ckpt), jepa, optimizer=None, scheduler=None, device=device, strict=False
    )
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    xy_head = MLPXYHead(input_shape=fdim, normalizer=None).to(device)
    if "xy_head_state_dict" in ckpt_info:
        xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])
    xy_head.eval()

    value_head = GoalValueHead(fdim).to(device)
    if objective == "learned_value" and "value_head_state_dict" not in ckpt_info:
        raise ValueError(f"{ckpt} has no value_head_state_dict")
    if "value_head_state_dict" in ckpt_info:
        value_head.load_state_dict(ckpt_info["value_head_state_dict"])
    value_head.eval()

    env = create_env(
        cfg.data.env_name,
        config=env_config,
        n_allowed_steps=max_steps + 1,
        n_steps=max_steps + 1,
        max_step_norm=1.5,
    )
    norm = env.normalizer
    xy_head.normalizer = norm
    off = (cell_size - 1) / 2.0
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}

    def obs_tensor(obs):
        return norm.normalize_state(
            obs.to(dtype=torch.float32, device=device)
        ).unsqueeze(0).unsqueeze(2)

    def pred_cell(z):
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (
            int(round((float(xy[0]) - off) / cell_size)),
            int(round((float(xy[1]) - off) / cell_size)),
        )

    print(
        f"[dyn-bootstrap-value] objective={objective} episodes={num_ep} "
        f"lookahead={lookahead} max_steps={max_steps}",
        flush=True,
    )

    successes, steps_all, blocked_all, manhattan_all = [], [], [], []
    for ep in range(num_ep):
        obs, info = env.reset()
        goal_img = info["target_obs"]
        goal_enc = jepa.encode(obs_tensor(goal_img.detach().clone().to(torch.float32)))
        goal_xy = norm.normalize_location(
            info["target_position"].to(dtype=torch.float32, device=device).unsqueeze(0)
        )[0]

        frames = [overlay_goal(observation_to_rgb(obs), env)]
        blocked = {}
        visits = {}
        last_reverse = -1
        success = False
        n_moves = 0

        for _ in range(max_steps):
            ot = obs_tensor(obs)
            ot4 = ot.repeat(4, 1, 1, 1, 1)
            zf = fine_kstep_target(
                jepa,
                ot4,
                torch.arange(4, device=device),
                lookahead,
                cell_size,
            )

            if objective == "learned_value":
                values = value_head(zf.float(), goal_enc.float()).squeeze(1)
                scores = (-values).detach().cpu().numpy()
            elif objective == "probe_pos":
                xy = xy_head(zf.float()).permute(0, 2, 1)[:, 0]
                scores = ((xy - goal_xy.unsqueeze(0)) ** 2).mean(dim=-1)
                scores = scores.detach().cpu().numpy()
            else:
                target = goal_enc.expand(4, -1, zf.shape[2], -1, -1)
                scores = ((zf.float() - target.float()) ** 2).mean(dim=(1, 2, 3, 4))
                scores = scores.detach().cpu().numpy()

            cell = tuple(int(c) for c in env.agent_cell)
            visits[cell] = visits.get(cell, 0) + 1
            if revisit_pen > 0:
                for d in range(4):
                    scores[d] += revisit_pen * visits.get(pred_cell(zf[d : d + 1]), 0)

            order = sorted(range(4), key=lambda d: float(scores[d]))
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
            if done or np.array_equal(env.agent_cell, env.goal_cell):
                success = True
                break
            if not moved:
                break

        successes.append(float(success))
        steps_all.append(n_moves)
        blocked_all.append(int(env.blocked_moves))
        manhattan = int(np.abs(env.agent_cell - env.goal_cell).sum())
        manhattan_all.append(manhattan)
        if ep < n_gifs:
            label = "succ" if success else "fail"
            imageio.mimsave(
                os.path.join(rdir, f"ep{ep}_{objective}_{label}.gif"),
                frames,
                fps=8,
                loop=0,
            )
        print(
            f"[dyn-bootstrap-value] ep={ep} {'SUCCESS' if success else 'fail'} "
            f"moves={n_moves} blocked={env.blocked_moves} manhattan={manhattan}",
            flush=True,
        )

    result = {
        "objective": objective,
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps_all)),
        "mean_blocked_moves": float(np.mean(blocked_all)),
        "mean_final_manhattan": float(np.mean(manhattan_all)),
        "num_episodes": num_ep,
        "lookahead": lookahead,
        "revisit_pen": revisit_pen,
        "budget": f"fixed_{max_steps}_moves",
        "astar_free": True,
    }
    with open(os.path.join(rdir, f"{objective}_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[dyn-bootstrap-value] {objective}: success={100*result['success_rate']:.1f}% "
        f"steps={result['mean_steps']:.1f} manhattan={result['mean_final_manhattan']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
