"""Evaluate naive A* baselines on Dynamic Maze.

The metrics here are deliberately operational, not claims of stochastic
optimality. They tell us whether the pivot environment is solvable, how hard
fog-of-war is, and how much door stochasticity hurts myopic replanning.
"""

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from eb_jepa.datasets.dynamic_maze.dynamic_maze import (
    DynamicMazeDatasetConfig,
    DynamicMazeEnv,
    _fog_mask,
    observation_to_rgb,
)
from eb_jepa.datasets.maze.maze_solver import DIRECTIONS, solve_a_star


POLICIES = (
    "oracle_replan",
    "fog_optimistic",
    "fog_conservative",
    "memory_optimistic",
    "memory_conservative",
    "random",
)


def overlay_goal(frame, env):
    out = frame.copy()
    r, c = env.goal_cell
    cs = env.config.cell_size
    rr = slice(int(r * cs), int((r + 1) * cs))
    cc = slice(int(c * cs), int((c + 1) * cs))
    out[rr, cc] = np.array([245, 215, 50], dtype=np.uint8)
    return out


def random_action(env):
    idx = int(env.rng.integers(0, 4))
    dr, dc = DIRECTIONS[idx]
    return np.array([dr, dc], dtype=np.float32) * env.config.cell_size


def astar_action_on_grid(env, grid):
    solved = solve_a_star(
        grid,
        tuple(int(v) for v in env.agent_cell.tolist()),
        tuple(int(v) for v in env.goal_cell.tolist()),
    )
    if solved is None or not solved[1]:
        return np.zeros(2, dtype=np.float32)
    dr, dc = DIRECTIONS[solved[1][0]]
    return np.array([dr, dc], dtype=np.float32) * env.config.cell_size


def strict_fog_action(env, policy):
    """Fog A* without oracle fallback, for honest baseline evaluation."""
    return astar_action_on_grid(env, env._teacher_grid(policy))


class FogMemoryPlanner:
    """A* planner with persistent discovered map under fog-of-war."""

    def __init__(self, env, optimistic: bool):
        self.optimistic = optimistic
        fill = 1 if optimistic else 0
        self.grid = np.full_like(env.base_grid, fill_value=fill, dtype=np.uint8)
        self.known = np.zeros_like(env.base_grid, dtype=bool)
        self.update(env)

    def update(self, env):
        current = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        visible = _fog_mask(
            current.shape,
            env.agent_cell,
            env.config.fog_radius,
            env.config.fog_metric,
        )
        self.known[visible] = True
        self.grid[visible] = current[visible]
        self.grid[tuple(env.agent_cell)] = 1
        self.grid[tuple(env.goal_cell)] = 1

    def action(self, env):
        return astar_action_on_grid(env, self.grid)


def current_astar_len(env):
    grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
    solved = solve_a_star(
        grid,
        tuple(int(v) for v in env.agent_cell.tolist()),
        tuple(int(v) for v in env.goal_cell.tolist()),
    )
    return len(solved[0]) - 1 if solved is not None else None


def run_policy(policy, cfg, episodes, budget_factor, budget_margin, n_gifs, out_dir):
    out_dir = Path(out_dir) / policy
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in range(episodes):
        ep_cfg = DynamicMazeDatasetConfig(**cfg.__dict__)
        ep_cfg.seed = int(cfg.seed or 0) + 100_003 * ep
        env = DynamicMazeEnv(ep_cfg, normalize=False, n_allowed_steps=10_000)
        obs, _ = env.reset()
        memory = None
        if policy == "memory_optimistic":
            memory = FogMemoryPlanner(env, optimistic=True)
        elif policy == "memory_conservative":
            memory = FogMemoryPlanner(env, optimistic=False)
        initial_len = current_astar_len(env) or 100
        max_steps = min(int(budget_factor * initial_len + budget_margin), 10_000)
        frames = [overlay_goal(observation_to_rgb(obs), env)]
        success = False

        for step in range(max_steps):
            if policy == "random":
                action = random_action(env)
            elif memory is not None:
                memory.update(env)
                action = memory.action(env)
            elif policy.startswith("fog_"):
                action = strict_fog_action(env, policy)
            else:
                action = env.teacher_action(policy)
            obs, _, done, trunc, _ = env.step(action)
            if ep < n_gifs:
                frames.append(overlay_goal(observation_to_rgb(obs), env))
            if done:
                success = True
                break
            if trunc:
                break

        steps = len(env.position_history) - 1
        rows.append(
            {
                "episode": ep,
                "success": bool(success),
                "steps": steps,
                "initial_astar_len": initial_len,
                "blocked_moves": int(env.blocked_moves),
                "efficiency": float(initial_len / max(steps, initial_len))
                if success
                else 0.0,
            }
        )
        if ep < n_gifs:
            label = "succ" if success else "fail"
            imageio.mimsave(out_dir / f"ep{ep}_{label}.gif", frames, fps=8, loop=0)
        print(
            f"[dynamic-maze] {policy} ep={ep} "
            f"{'SUCCESS' if success else 'fail'} steps={steps} "
            f"A0={initial_len} blocked={env.blocked_moves}",
            flush=True,
        )

    success_rate = float(np.mean([r["success"] for r in rows]))
    result = {
        "policy": policy,
        "success_rate": success_rate,
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_initial_astar_len": float(np.mean([r["initial_astar_len"] for r in rows])),
        "mean_blocked_moves": float(np.mean([r["blocked_moves"] for r in rows])),
        "mean_efficiency": float(np.mean([r["efficiency"] for r in rows])),
        "episodes": episodes,
        "budget": f"{budget_factor}xA*_initial+{budget_margin}",
        "config": cfg.__dict__,
        "episodes_detail": rows,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-doors", type=int, default=8)
    parser.add_argument("--door-toggle-prob", type=float, default=0.04)
    parser.add_argument("--door-open-prob", type=float, default=0.35)
    parser.add_argument("--fog-radius", type=int, default=4)
    parser.add_argument("--n-gifs", type=int, default=4)
    parser.add_argument("--budget-factor", type=float, default=6.0)
    parser.add_argument("--budget-margin", type=int, default=20)
    parser.add_argument(
        "--policies",
        default="oracle_replan,fog_optimistic,fog_conservative,random",
        help=f"Comma-separated subset of {POLICIES}",
    )
    args = parser.parse_args()

    cfg = DynamicMazeDatasetConfig(
        num_doors=args.num_doors,
        door_toggle_prob=args.door_toggle_prob,
        door_initial_open_prob=args.door_open_prob,
        fog_radius=args.fog_radius,
        seed=args.seed,
        normalize=False,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    bad = sorted(set(policies) - set(POLICIES))
    if bad:
        raise ValueError(f"Unknown policies: {bad}")

    results = [
        run_policy(
            policy,
            cfg,
            args.episodes,
            args.budget_factor,
            args.budget_margin,
            args.n_gifs,
            out,
        )
        for policy in policies
    ]
    with (out / "summary.tsv").open("w") as f:
        f.write("policy\tsuccess_rate\tmean_efficiency\tmean_steps\tmean_blocked_moves\n")
        for r in results:
            f.write(
                f"{r['policy']}\t{r['success_rate']:.6f}\t"
                f"{r['mean_efficiency']:.6f}\t{r['mean_steps']:.3f}\t"
                f"{r['mean_blocked_moves']:.3f}\n"
            )
    print(f"[dynamic-maze] wrote {out / 'summary.tsv'}")


if __name__ == "__main__":
    main()
