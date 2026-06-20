"""Lightweight egocentric vision renderer for Dynamic Maze.

The renderer is intentionally analytic: it raycasts the 21x21 grid directly and
does not require Three.js, Chromium, OpenGL, or EGL. A state is rendered as four
grayscale views ordered as up, down, left, right.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, pi, radians, sin, tan
from typing import Iterable, NamedTuple

import numpy as np


class RayHit(NamedTuple):
    distance: float
    side: int
    cell: tuple[int, int]
    hit: bool


@dataclass(frozen=True)
class VisionRenderConfig:
    image_size: int = 128
    fov_degrees: float = 82.0
    vertical_fov_degrees: float = 68.0
    max_depth: float = 24.0
    wall_height_m: float = 2.0
    eye_height_m: float = 1.8
    horizon_fraction: float = 0.47
    goal_marker: bool = True
    door_boost: int = 34


VIEW_NAMES = ("up", "down", "left", "right")
VIEW_ANGLES = {
    "up": -pi / 2.0,
    "down": pi / 2.0,
    "left": pi,
    "right": 0.0,
}


def door_mask_from_cells(shape: tuple[int, int], door_cells: np.ndarray | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if door_cells is None:
        return mask
    for r, c in np.asarray(door_cells, dtype=np.int32):
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
            mask[int(r), int(c)] = True
    return mask


def _angle_diff(a: float, b: float) -> float:
    return (a - b + pi) % (2.0 * pi) - pi


def _raycast_grid(grid: np.ndarray, origin_xy: tuple[float, float], angle: float, max_depth: float) -> RayHit:
    """Return first wall hit by a ray in a grid where 0 is blocked and 1 is free."""
    height, width = grid.shape
    ox, oy = origin_xy
    dx, dy = cos(angle), sin(angle)
    map_x, map_y = int(ox), int(oy)

    if abs(dx) < 1.0e-9:
        delta_x = 1.0e9
        step_x = 0
        side_x = 1.0e9
    elif dx > 0.0:
        delta_x = abs(1.0 / dx)
        step_x = 1
        side_x = (map_x + 1.0 - ox) * delta_x
    else:
        delta_x = abs(1.0 / dx)
        step_x = -1
        side_x = (ox - map_x) * delta_x

    if abs(dy) < 1.0e-9:
        delta_y = 1.0e9
        step_y = 0
        side_y = 1.0e9
    elif dy > 0.0:
        delta_y = abs(1.0 / dy)
        step_y = 1
        side_y = (map_y + 1.0 - oy) * delta_y
    else:
        delta_y = abs(1.0 / dy)
        step_y = -1
        side_y = (oy - map_y) * delta_y

    dist = 0.0
    side = 0
    for _ in range(height + width + 8):
        if side_x < side_y:
            map_x += step_x
            dist = side_x
            side_x += delta_x
            side = 0
        else:
            map_y += step_y
            dist = side_y
            side_y += delta_y
            side = 1
        if dist > max_depth:
            break
        if not (0 <= map_x < width and 0 <= map_y < height):
            return RayHit(min(dist, max_depth), side, (map_y, map_x), True)
        if grid[map_y, map_x] == 0:
            return RayHit(max(dist, 1.0e-3), side, (map_y, map_x), True)
    return RayHit(max_depth, side, (map_y, map_x), False)


def _background(size: int) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    horizon = int(size * 0.47)
    if horizon > 0:
        ceiling = np.linspace(48, 28, horizon, dtype=np.uint8)[:, None]
        img[:horizon, :] = ceiling
    if horizon < size:
        floor = np.linspace(44, 92, size - horizon, dtype=np.uint8)[:, None]
        img[horizon:, :] = floor
    return img


def _render_goal_marker(
    image: np.ndarray,
    zbuffer: np.ndarray,
    origin_xy: tuple[float, float],
    goal_cell: np.ndarray,
    center_angle: float,
    config: VisionRenderConfig,
) -> None:
    gx = float(goal_cell[1]) + 0.5
    gy = float(goal_cell[0]) + 0.5
    ox, oy = origin_xy
    vec_x, vec_y = gx - ox, gy - oy
    distance = float(np.hypot(vec_x, vec_y))
    if distance < 0.35:
        return
    delta = _angle_diff(np.arctan2(vec_y, vec_x), center_angle)
    half_fov = radians(config.fov_degrees) / 2.0
    if abs(delta) > half_fov:
        return
    camera_x = tan(delta) / tan(half_fov)
    col = int(round((camera_x + 1.0) * 0.5 * (config.image_size - 1)))
    if not (0 <= col < config.image_size):
        return
    span = max(1, int(2 + 8 / max(distance, 1.0)))
    lo, hi = max(0, col - span), min(config.image_size, col + span + 1)
    if distance >= float(np.min(zbuffer[lo:hi])) - 0.2:
        return

    focal_y = (config.image_size / 2.0) / tan(radians(config.vertical_fov_degrees) / 2.0)
    horizon = config.image_size * config.horizon_fraction
    bottom = int(horizon + (config.eye_height_m / distance) * focal_y)
    marker_h = int(max(8, min(config.image_size * 0.7, 20.0 / max(distance, 0.8))))
    top = max(0, bottom - marker_h)
    bottom = min(config.image_size, bottom)
    if top < bottom:
        image[top:bottom, lo:hi] = 255


def render_view(
    current_grid: np.ndarray,
    agent_cell: np.ndarray,
    view_name: str,
    goal_cell: np.ndarray | None = None,
    door_cells: np.ndarray | None = None,
    config: VisionRenderConfig | None = None,
) -> np.ndarray:
    """Render one cardinal egocentric view as a grayscale uint8 image."""
    if view_name not in VIEW_ANGLES:
        raise ValueError(f"view_name must be one of {VIEW_NAMES}, got {view_name!r}")
    config = config or VisionRenderConfig()
    grid = np.asarray(current_grid, dtype=np.uint8)
    door_mask = door_mask_from_cells(grid.shape, door_cells)
    size = int(config.image_size)
    fov = radians(config.fov_degrees)
    focal_y = (size / 2.0) / tan(radians(config.vertical_fov_degrees) / 2.0)
    horizon = size * config.horizon_fraction
    origin_xy = (float(agent_cell[1]) + 0.5, float(agent_cell[0]) + 0.5)
    center_angle = VIEW_ANGLES[view_name]

    image = _background(size)
    zbuffer = np.full((size,), config.max_depth, dtype=np.float32)
    half_tan = tan(fov / 2.0)

    for col in range(size):
        camera_x = (2.0 * (col + 0.5) / size) - 1.0
        ray_angle = center_angle + atan(camera_x * half_tan)
        hit = _raycast_grid(grid, origin_xy, ray_angle, config.max_depth)
        corrected = max(0.05, hit.distance * cos(_angle_diff(ray_angle, center_angle)))
        zbuffer[col] = corrected
        if not hit.hit:
            continue

        top = int(horizon - ((config.wall_height_m - config.eye_height_m) / corrected) * focal_y)
        bottom = int(horizon + (config.eye_height_m / corrected) * focal_y)
        top = max(0, min(size, top))
        bottom = max(0, min(size, bottom))
        if top >= bottom:
            continue

        attenuation = 1.0 / (1.0 + 0.055 * corrected * corrected)
        shade = int(74 + 176 * attenuation)
        if hit.side == 1:
            shade = int(shade * 0.82)
        r, c = hit.cell
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and door_mask[r, c]:
            shade = min(255, shade + config.door_boost)
        image[top:bottom, col] = np.uint8(np.clip(shade, 0, 255))

    if config.goal_marker and goal_cell is not None:
        _render_goal_marker(image, zbuffer, origin_xy, np.asarray(goal_cell), center_angle, config)
    return image


def render_four_views(
    current_grid: np.ndarray,
    agent_cell: np.ndarray,
    goal_cell: np.ndarray | None = None,
    door_cells: np.ndarray | None = None,
    config: VisionRenderConfig | None = None,
    view_names: Iterable[str] = VIEW_NAMES,
) -> np.ndarray:
    """Render cardinal egocentric views as uint8 tensor [4, H, W]."""
    config = config or VisionRenderConfig()
    return np.stack(
        [
            render_view(current_grid, agent_cell, name, goal_cell, door_cells, config)
            for name in view_names
        ],
        axis=0,
    )
