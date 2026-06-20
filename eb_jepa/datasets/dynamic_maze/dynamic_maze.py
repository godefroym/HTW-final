"""Dynamic maze with stochastic doors and fog-of-war.

This environment is intentionally separate from the static Maze benchmark. It
lets us test a setting where a static shortest path is no longer the right
planning objective: doors open/close stochastically, the agent only observes a
local fog-of-war window, and A* baselines replan myopically on the current map.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from eb_jepa.datasets.dynamic_maze.normalizer import DynamicMazeNormalizer
from eb_jepa.datasets.dynamic_maze.vision_renderer import (
    VisionRenderConfig,
    render_four_views,
)
from eb_jepa.datasets.maze.maze_dataset import cell_to_pixel, render_dot
from eb_jepa.datasets.maze.maze_generator import generate_maze
from eb_jepa.datasets.maze.maze_solver import DIRECTIONS, solve_a_star

InfoType = Dict[str, Any]
ObsType = torch.Tensor


class DynamicMazeSample(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    locations: torch.Tensor
    wall_x: torch.Tensor
    door_y: torch.Tensor


@dataclass
class DynamicMazeDatasetConfig:
    maze_height: int = 21
    maze_width: int = 21
    cell_size: int = 3
    img_size: int = 63
    observation_mode: str = "grid"
    vision_fov_degrees: float = 82.0
    vision_vertical_fov_degrees: float = 68.0
    vision_max_depth: float = 24.0
    vision_goal_marker: bool = True

    n_steps: int = 129
    sample_length: int = 17
    min_path_length: int = 50
    max_gen_retries: int = 64
    layout_use_astar_filter: bool = True

    num_doors: int = 8
    door_initial_open_prob: float = 0.35
    door_toggle_prob: float = 0.04
    min_door_candidates: int = 8

    fog_radius: int = 4
    fog_metric: str = "manhattan"
    teacher_policy: str = "oracle_replan"
    explore_random_prob: float = 0.12
    explore_repeat_prob: float = 0.65
    explore_novelty_bonus: float = 1.0
    explore_goal_bias: float = 0.08
    explore_visit_penalty: float = 0.15

    agent_std: float = 1.2
    seed: Optional[int] = 1

    size: int = 100000
    val_size: int = 10000
    batch_size: int = 64
    train: bool = True
    device: str = "cpu"
    normalize: bool = True
    num_workers: int = 0
    pin_mem: bool = False
    persistent_workers: bool = False


def _door_candidates(maze: np.ndarray) -> np.ndarray:
    """Return wall cells that can become doors between two corridor cells."""
    h, w = maze.shape
    candidates: List[Tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if maze[r, c] != 0:
                continue
            vertical = maze[r - 1, c] == 1 and maze[r + 1, c] == 1
            horizontal = maze[r, c - 1] == 1 and maze[r, c + 1] == 1
            if vertical or horizontal:
                candidates.append((r, c))
    if not candidates:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(candidates, dtype=np.int32)


def _sample_layout(config: DynamicMazeDatasetConfig, rng: np.random.Generator):
    h, w = config.maze_height, config.maze_width
    start = np.array([1, 1], dtype=np.int32)
    goal = np.array([h - 2, w - 2], dtype=np.int32)
    last = None
    for _ in range(config.max_gen_retries):
        maze = generate_maze(h, w, rng=rng)
        maze[start[0], start[1]] = 1
        maze[goal[0], goal[1]] = 1
        sol = (
            solve_a_star(maze, tuple(start), tuple(goal))
            if config.layout_use_astar_filter
            else None
        )
        candidates = _door_candidates(maze)
        last = (maze, candidates, sol)
        enough_doors = len(candidates) >= min(config.min_door_candidates, config.num_doors)
        if config.layout_use_astar_filter:
            valid_layout = (
                sol is not None
                and len(sol[0]) >= config.min_path_length
                and enough_doors
            )
        else:
            valid_layout = enough_doors
        if valid_layout:
            break
    else:
        maze, candidates, sol = last

    if len(candidates) == 0 or config.num_doors <= 0:
        doors = np.zeros((0, 2), dtype=np.int32)
        door_open = np.zeros((0,), dtype=bool)
    else:
        n = min(config.num_doors, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        doors = candidates[idx].astype(np.int32)
        door_open = rng.random(n) < config.door_initial_open_prob
    return maze.astype(np.uint8), doors, door_open.astype(bool), start, goal


def _current_grid(
    base_grid: np.ndarray,
    door_cells: np.ndarray,
    door_open: np.ndarray,
    force_open: Tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    grid = base_grid.copy()
    for cell, is_open in zip(door_cells, door_open):
        if is_open:
            grid[int(cell[0]), int(cell[1])] = 1
    for cell in force_open:
        grid[int(cell[0]), int(cell[1])] = 1
    return grid


def _fog_mask(shape: Tuple[int, int], center: np.ndarray, radius: int, metric: str):
    rr, cc = np.indices(shape)
    if metric == "chebyshev":
        dist = np.maximum(np.abs(rr - center[0]), np.abs(cc - center[1]))
    else:
        dist = np.abs(rr - center[0]) + np.abs(cc - center[1])
    return dist <= radius


def _cell_mask_to_img(mask: np.ndarray, cell_size: int) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.uint8) * 255)
    return t.repeat_interleave(cell_size, dim=0).repeat_interleave(cell_size, dim=1)


def render_observation(
    agent_cell: np.ndarray,
    goal_cell: np.ndarray,
    base_grid: np.ndarray,
    door_cells: np.ndarray,
    door_open: np.ndarray,
    config: DynamicMazeDatasetConfig,
    device: torch.device,
) -> torch.Tensor:
    """Render 4-channel uint8 observation under fog-of-war."""
    grid = _current_grid(base_grid, door_cells, door_open, (agent_cell, goal_cell))
    known = _fog_mask(grid.shape, agent_cell, config.fog_radius, config.fog_metric)
    door_mask = np.zeros_like(grid, dtype=bool)
    for cell in door_cells:
        door_mask[int(cell[0]), int(cell[1])] = True

    wall_obs = (grid == 0) & known
    unknown = ~known
    door_obs = door_mask & known

    dot_position = torch.tensor(
        cell_to_pixel(agent_cell, config.cell_size),
        dtype=torch.float32,
        device=device,
    )
    dot = render_dot(dot_position, config.img_size, config.agent_std, device=device)
    wall_img = _cell_mask_to_img(wall_obs, config.cell_size).to(device)
    unknown_img = _cell_mask_to_img(unknown, config.cell_size).to(device)
    door_img = _cell_mask_to_img(door_obs, config.cell_size).to(device)
    return torch.stack([dot, wall_img, unknown_img, door_img], dim=0)


def render_vision_observation(
    agent_cell: np.ndarray,
    goal_cell: np.ndarray,
    base_grid: np.ndarray,
    door_cells: np.ndarray,
    door_open: np.ndarray,
    config: DynamicMazeDatasetConfig,
    device: torch.device,
) -> torch.Tensor:
    """Render four egocentric grayscale views as uint8 [4, H, W]."""
    grid = _current_grid(base_grid, door_cells, door_open, (agent_cell, goal_cell))
    render_cfg = VisionRenderConfig(
        image_size=int(config.img_size),
        fov_degrees=float(config.vision_fov_degrees),
        vertical_fov_degrees=float(config.vision_vertical_fov_degrees),
        max_depth=float(config.vision_max_depth),
        goal_marker=bool(config.vision_goal_marker),
    )
    views = render_four_views(
        grid,
        agent_cell,
        goal_cell=goal_cell,
        door_cells=door_cells,
        config=render_cfg,
    )
    return torch.from_numpy(views).to(device)


def observation_to_rgb(obs: torch.Tensor) -> np.ndarray:
    """Convert a 4-channel dynamic-maze observation to an RGB debug frame."""
    arr = obs.detach().cpu().numpy()
    if arr.max() <= 2.0:
        arr = np.clip(arr, 0, 1) * 255.0
    arr = arr.astype(np.uint8)
    h, w = arr.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unknown = arr[2] > 0
    walls = arr[1] > 0
    doors = arr[3] > 0
    agent = arr[0] > 12

    rgb[unknown] = np.array([25, 25, 32], dtype=np.uint8)
    rgb[walls] = np.array([150, 150, 150], dtype=np.uint8)
    rgb[doors & walls] = np.array([45, 110, 230], dtype=np.uint8)
    rgb[doors & ~walls] = np.array([60, 190, 130], dtype=np.uint8)
    rgb[agent] = np.array([235, 55, 55], dtype=np.uint8)
    return rgb


class DynamicMazeEnv(gym.Env):
    def __init__(
        self,
        config: DynamicMazeDatasetConfig,
        rng: Optional[np.random.Generator] = None,
        n_steps: int = 300,
        n_allowed_steps: int = 300,
        max_step_norm: float = 1.5,
        normalize: bool = True,
        **_unused,
    ):
        super().__init__()
        self.config = config
        self.rng = rng or np.random.default_rng(config.seed)
        self.n_steps = n_steps
        self.n_allowed_steps = n_allowed_steps
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.normalize = normalize
        self.normalizer = DynamicMazeNormalizer(config.img_size) if normalize else None
        self.action_space = gym.spaces.Box(
            low=-max_step_norm * config.cell_size,
            high=max_step_norm * config.cell_size,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, config.img_size, config.img_size),
            dtype=np.float32,
        )
        self.base_grid = None
        self.maze_grid = None
        self.door_cells = None
        self.door_open = None
        self.agent_cell = None
        self.goal_cell = None
        self.dot_position = None
        self.target_position = None
        self.position_history = []
        self.blocked_moves = 0
        self.known_cells = None
        self.visit_counts = None
        self._last_explore_dir = None

    def reset(self, location=None) -> Tuple[ObsType, InfoType]:
        base, doors, open_, start, goal = _sample_layout(self.config, self.rng)
        self.base_grid = base
        self.door_cells = doors
        self.door_open = open_
        self.agent_cell = start.copy()
        self.goal_cell = goal.copy()
        if location is not None:
            self.agent_cell = self._pixel_to_cell(location)
        self.blocked_moves = 0
        self.known_cells = np.zeros_like(self.base_grid, dtype=bool)
        self.visit_counts = np.zeros_like(self.base_grid, dtype=np.int32)
        self._last_explore_dir = None
        self._sync_positions()
        self._sync_grid()
        self._update_known()
        self.position_history = [self.dot_position]
        self.visit_counts[tuple(self.agent_cell)] += 1
        return self._render(), self._build_info()

    def _sync_grid(self):
        grid = _current_grid(
            self.base_grid, self.door_cells, self.door_open,
            (self.agent_cell, self.goal_cell),
        )
        self.maze_grid = torch.from_numpy(grid.astype(np.int64)).to(self.device)

    def _sync_positions(self):
        self.dot_position = torch.tensor(
            cell_to_pixel(self.agent_cell, self.config.cell_size),
            device=self.device,
            dtype=torch.float32,
        )
        self.target_position = torch.tensor(
            cell_to_pixel(self.goal_cell, self.config.cell_size),
            device=self.device,
            dtype=torch.float32,
        )

    def _toggle_doors(self):
        if len(self.door_open) == 0 or self.config.door_toggle_prob <= 0:
            return
        flip = self.rng.random(len(self.door_open)) < self.config.door_toggle_prob
        self.door_open = np.logical_xor(self.door_open, flip)

    def _update_known(self):
        if self.known_cells is None:
            return
        self.known_cells |= _fog_mask(
            self.base_grid.shape,
            self.agent_cell,
            self.config.fog_radius,
            self.config.fog_metric,
        )

    def step(
        self,
        action,
        render_observation: bool = True,
    ) -> Tuple[ObsType, float, bool, bool, InfoType]:
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)

        dr_cont, dc_cont = float(action_np[0]), float(action_np[1])
        if abs(dr_cont) >= abs(dc_cont):
            dr, dc = ((1 if dr_cont > 0 else -1) if dr_cont != 0 else 0), 0
        else:
            dr, dc = 0, ((1 if dc_cont > 0 else -1) if dc_cont != 0 else 0)

        grid = _current_grid(
            self.base_grid, self.door_cells, self.door_open,
            (self.agent_cell, self.goal_cell),
        )
        nr, nc = int(self.agent_cell[0] + dr), int(self.agent_cell[1] + dc)
        moved = False
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 1:
            self.agent_cell = np.array([nr, nc], dtype=np.int32)
            moved = bool(dr != 0 or dc != 0)
        elif dr != 0 or dc != 0:
            self.blocked_moves += 1

        done = bool(np.array_equal(self.agent_cell, self.goal_cell))
        self._toggle_doors()
        self._sync_positions()
        self._sync_grid()
        self._update_known()
        if self.visit_counts is not None:
            self.visit_counts[tuple(self.agent_cell)] += 1
        self.position_history.append(self.dot_position)

        truncated = len(self.position_history) >= self.n_allowed_steps
        reward = 1.0 if done else 0.0
        info = self._build_info()
        info["moved"] = moved
        obs = self._render() if render_observation else None
        return obs, reward, done, truncated, info

    def _pixel_to_cell(self, pixel):
        if isinstance(pixel, torch.Tensor):
            pixel = pixel.detach().cpu().numpy()
        offset = (self.config.cell_size - 1) / 2.0
        cell = np.rint((np.asarray(pixel, dtype=np.float32) - offset) / self.config.cell_size)
        cell = np.clip(cell, [0, 0], [self.config.maze_height - 1, self.config.maze_width - 1])
        return cell.astype(np.int32)

    def _render(self):
        if self.config.observation_mode == "vision":
            return render_vision_observation(
                self.agent_cell,
                self.goal_cell,
                self.base_grid,
                self.door_cells,
                self.door_open,
                self.config,
                self.device,
            )
        if self.config.observation_mode != "grid":
            raise ValueError(
                "DynamicMazeDatasetConfig.observation_mode must be 'grid' or 'vision'"
            )
        return render_observation(
            self.agent_cell,
            self.goal_cell,
            self.base_grid,
            self.door_cells,
            self.door_open,
            self.config,
            self.device,
        )

    def _build_info(self) -> InfoType:
        return {
            "dot_position": self.dot_position,
            "target_position": self.target_position,
            "target_obs": self.get_target_obs(),
            "base_grid": self.base_grid.copy(),
            "current_grid": self.maze_grid.detach().cpu().numpy().astype(np.uint8),
            "door_cells": self.door_cells.copy(),
            "door_open": self.door_open.copy(),
            "known_cells": None if self.known_cells is None else self.known_cells.copy(),
            "blocked_moves": self.blocked_moves,
        }

    def get_target_obs(self):
        old = self.agent_cell.copy()
        self.agent_cell = self.goal_cell.copy()
        obs = self._render()
        self.agent_cell = old
        return obs

    def _teacher_grid(self, policy: str):
        current = _current_grid(
            self.base_grid, self.door_cells, self.door_open,
            (self.agent_cell, self.goal_cell),
        )
        if policy == "oracle_replan":
            return current

        known = _fog_mask(
            current.shape, self.agent_cell, self.config.fog_radius, self.config.fog_metric
        )
        if policy == "fog_conservative":
            grid = np.zeros_like(current, dtype=np.uint8)
        elif policy == "fog_optimistic":
            grid = np.ones_like(current, dtype=np.uint8)
        else:
            raise ValueError(
                "teacher policy must be oracle_replan, fog_optimistic, or fog_conservative"
            )
        grid[known] = current[known]
        grid[tuple(self.agent_cell)] = 1
        grid[tuple(self.goal_cell)] = 1
        return grid

    def _cardinal_action(self, direction_idx: int) -> np.ndarray:
        dr, dc = DIRECTIONS[int(direction_idx)]
        return np.array([dr, dc], dtype=np.float32) * self.config.cell_size

    def _random_cardinal_action(self) -> np.ndarray:
        return self._cardinal_action(int(self.rng.integers(0, len(DIRECTIONS))))

    def _local_frontier_scores(self, goal_biased: bool) -> np.ndarray:
        """Score one-step local moves without any graph search.

        The policy only uses the current grid under the agent, the local fog mask
        and visit counts. It is intentionally myopic: it bootstraps world-model
        data from exploration rather than from a shortest-path oracle.
        """
        current = _current_grid(
            self.base_grid, self.door_cells, self.door_open,
            (self.agent_cell, self.goal_cell),
        )
        known = self.known_cells
        visits = self.visit_counts
        scores = np.full((len(DIRECTIONS),), -1.0e9, dtype=np.float32)
        for i, (dr, dc) in enumerate(DIRECTIONS):
            nxt = self.agent_cell + np.array([dr, dc], dtype=np.int32)
            r, c = int(nxt[0]), int(nxt[1])
            if not (0 <= r < current.shape[0] and 0 <= c < current.shape[1]):
                continue
            if current[r, c] != 1:
                continue
            visible_after = _fog_mask(
                current.shape, nxt, self.config.fog_radius, self.config.fog_metric
            )
            novelty = float((visible_after & ~known).sum()) if known is not None else 0.0
            revisit = float(visits[r, c]) if visits is not None else 0.0
            score = (
                self.config.explore_novelty_bonus * novelty
                - self.config.explore_visit_penalty * revisit
            )
            if goal_biased:
                manhattan = float(np.abs(nxt - self.goal_cell).sum())
                score -= self.config.explore_goal_bias * manhattan
            score += float(self.rng.normal(0.0, 1.0e-3))
            scores[i] = score
        return scores

    def exploration_action(self, policy: str) -> np.ndarray:
        """Action policies that do not call A* or any graph solver."""
        if policy in {"random", "random_cardinal"}:
            return self._random_cardinal_action()

        if policy == "persistent_random":
            if (
                self._last_explore_dir is None
                or self.rng.random() > self.config.explore_repeat_prob
            ):
                self._last_explore_dir = int(self.rng.integers(0, len(DIRECTIONS)))
            return self._cardinal_action(self._last_explore_dir)

        if policy in {"local_frontier", "local_goal_frontier"}:
            if self.rng.random() < self.config.explore_random_prob:
                return self._random_cardinal_action()
            scores = self._local_frontier_scores(goal_biased=policy == "local_goal_frontier")
            if not np.isfinite(scores).any() or float(scores.max()) < -1.0e8:
                return self._random_cardinal_action()
            self._last_explore_dir = int(np.argmax(scores))
            return self._cardinal_action(self._last_explore_dir)

        raise ValueError(
            "exploration policy must be random, persistent_random, "
            "local_frontier, or local_goal_frontier"
        )

    def teacher_action(self, policy: Optional[str] = None) -> np.ndarray:
        policy = policy or self.config.teacher_policy
        if policy in {
            "random",
            "random_cardinal",
            "persistent_random",
            "local_frontier",
            "local_goal_frontier",
        }:
            return self.exploration_action(policy)
        if np.array_equal(self.agent_cell, self.goal_cell):
            return np.zeros(2, dtype=np.float32)
        grid = self._teacher_grid(policy)
        solved = solve_a_star(
            grid,
            tuple(int(v) for v in self.agent_cell.tolist()),
            tuple(int(v) for v in self.goal_cell.tolist()),
        )
        if solved is None and policy != "oracle_replan":
            solved = solve_a_star(
                self._teacher_grid("oracle_replan"),
                tuple(int(v) for v in self.agent_cell.tolist()),
                tuple(int(v) for v in self.goal_cell.tolist()),
            )
        if solved is None or not solved[1]:
            return np.zeros(2, dtype=np.float32)
        dr, dc = DIRECTIONS[solved[1][0]]
        return np.array([dr, dc], dtype=np.float32) * self.config.cell_size

    def policy_action(self, policy: Optional[str] = None) -> np.ndarray:
        return self.teacher_action(policy)

    def eval_state(self, goal_dot_position, curr_dot_position, succes_treshold=None):
        curr_cell = self._pixel_to_cell(curr_dot_position)
        goal_cell = self._pixel_to_cell(goal_dot_position)
        grid = _current_grid(
            self.base_grid, self.door_cells, self.door_open,
            (curr_cell, goal_cell),
        )
        solved = solve_a_star(grid, tuple(curr_cell.tolist()), tuple(goal_cell.tolist()))
        if solved is None:
            return {"success": False, "state_dist": float("inf")}
        state_dist = float((len(solved[0]) - 1) * self.config.cell_size)
        threshold = self.config.cell_size + 0.5 if succes_treshold is None else succes_treshold
        return {"success": state_dist < threshold, "state_dist": state_dist}


class DynamicMazeDataset(torch.utils.data.Dataset):
    def __init__(self, config: DynamicMazeDatasetConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.normalizer = (
            DynamicMazeNormalizer(config.img_size) if config.normalize else None
        )
        self._rng = np.random.default_rng(config.seed)

    def __len__(self):
        return self.config.size if self.config.train else self.config.val_size

    def __getitem__(self, idx):
        rng = self._rng
        if self.config.seed is not None:
            split_offset = 0 if self.config.train else 1_000_000_000
            rng = np.random.default_rng(int(self.config.seed) + split_offset + int(idx))
        sample = self.generate_multistep_sample(rng=rng)
        return sample._replace(
            states=sample.states.squeeze(0),
            actions=sample.actions.squeeze(0),
            locations=sample.locations.squeeze(0),
        )

    def generate_multistep_sample(self, rng: Optional[np.random.Generator] = None):
        cfg = self.config
        rng = rng or self._rng
        env = DynamicMazeEnv(
            cfg,
            rng=rng,
            n_steps=cfg.n_steps + 1,
            n_allowed_steps=cfg.n_steps + 1,
            normalize=False,
        )
        obs, _ = env.reset()
        if cfg.observation_mode == "vision":
            return self._generate_multistep_sample_sparse_render(env, rng)

        states, actions, locations = [], [], []
        for _ in range(cfg.n_steps):
            states.append(obs.detach().cpu())
            locations.append(env.dot_position.detach().cpu())
            action = env.policy_action(cfg.teacher_policy)
            actions.append(torch.from_numpy(action.astype(np.float32)))
            obs, _, _, _, _ = env.step(action)

        states = torch.stack(states, dim=0).float()
        actions_t = torch.stack(actions, dim=0).float()
        locations_t = torch.stack(locations, dim=0).float()

        sl = cfg.sample_length
        max_start = max(0, cfg.n_steps - sl)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        states = states[start : start + sl]
        actions_t = actions_t[start : start + sl]
        locations_t = locations_t[start : start + sl]

        if cfg.normalize and self.normalizer is not None:
            states = self.normalizer.normalize_state(states)
            locations_t = self.normalizer.normalize_location(locations_t)

        states = states.permute(1, 0, 2, 3).unsqueeze(0)
        actions_t = actions_t.permute(1, 0).unsqueeze(0)
        locations_t = locations_t.permute(1, 0).unsqueeze(0)
        return DynamicMazeSample(
            states=states,
            actions=actions_t,
            locations=locations_t,
            wall_x=torch.zeros(1),
            door_y=torch.zeros(1),
        )

    def _generate_multistep_sample_sparse_render(
        self,
        env: DynamicMazeEnv,
        rng: np.random.Generator,
    ):
        """Generate a vision sample while rendering only the sampled time window."""
        cfg = self.config
        sl = cfg.sample_length
        max_start = max(0, cfg.n_steps - sl)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        stop = start + sl

        states, actions, locations = [], [], []
        for t in range(cfg.n_steps):
            in_window = start <= t < stop
            if in_window:
                states.append(env._render().detach().cpu())
                locations.append(env.dot_position.detach().cpu())
            action = env.policy_action(cfg.teacher_policy)
            if in_window:
                actions.append(torch.from_numpy(action.astype(np.float32)))
            env.step(action, render_observation=False)

        states = torch.stack(states, dim=0).float()
        actions_t = torch.stack(actions, dim=0).float()
        locations_t = torch.stack(locations, dim=0).float()

        if cfg.normalize and self.normalizer is not None:
            states = self.normalizer.normalize_state(states)
            locations_t = self.normalizer.normalize_location(locations_t)

        states = states.permute(1, 0, 2, 3).unsqueeze(0)
        actions_t = actions_t.permute(1, 0).unsqueeze(0)
        locations_t = locations_t.permute(1, 0).unsqueeze(0)
        return DynamicMazeSample(
            states=states,
            actions=actions_t,
            locations=locations_t,
            wall_x=torch.zeros(1),
            door_y=torch.zeros(1),
        )
