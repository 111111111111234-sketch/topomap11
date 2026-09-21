"""Observed-space planning on a SLAM occupancy grid, reusing topomap's A*."""

import numpy as np
from scipy import ndimage

from src.route_guidance import grid_path, visible


def traversable(grid, radius):
    # Unknown, ambiguous occupancy, and map boundaries are all blocked. Padding
    # matters: scipy otherwise treats all-free array boundaries asymmetrically.
    free = grid.cells == 0
    clearance = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]
    return free & (clearance * grid.resolution > radius + grid.resolution * 0.71)


def connected_free(grid, xy, radius):
    mask = traversable(grid, radius)
    start = grid.world_to_cell(xy)
    cell = np.rint(start).astype(int)
    if (np.any(cell < 0) or np.any(cell >= mask.shape) or not mask[tuple(cell)]
            or not visible(mask, start, cell)):
        return np.zeros_like(mask)
    labels, _ = ndimage.label(mask)
    return labels == labels[tuple(cell)]


def route(grid, reachable, start_xy, target_xy):
    start, end = grid.world_to_cell(start_xy), grid.world_to_cell(target_xy)
    path = grid_path(reachable, start, end)
    if path is None or not visible(reachable, start, path[0]):
        return None
    return np.vstack([start_xy, grid.cell_to_world(path)])


def frontiers(grid, reachable, minimum_cells=4, approach_distance=1.2):
    boundary = (grid.cells == 0) & ndimage.binary_dilation(grid.cells == -1)
    labels, count = ndimage.label(boundary, structure=np.ones((3, 3)))
    candidates = np.argwhere(reachable)
    if not len(candidates):
        return []
    result = []
    for label in range(1, count + 1):
        cluster = np.argwhere(labels == label)
        if len(cluster) < minimum_cells:
            continue
        center = cluster.mean(axis=0)
        distance = np.linalg.norm(candidates - center, axis=1) * grid.resolution
        closest = int(distance.argmin())
        if distance[closest] <= approach_distance:
            result.append((grid.cell_to_world(center), grid.cell_to_world(candidates[closest]), len(cluster)))
    return result


def object_approach(grid, reachable, start_xy, object_xy, distance=0.9, max_distance=None):
    cells = np.argwhere(reachable)
    if not len(cells):
        return None
    world = grid.cell_to_world(cells)
    distances = np.linalg.norm(world - object_xy, axis=1)
    upper = distance + .5 if max_distance is None else min(distance + .5, max_distance)
    valid = (distances >= distance) & (distances <= upper)
    if not valid.any():
        return None
    world = world[valid]
    best = np.argmin(np.linalg.norm(world - start_xy, axis=1))
    return route(grid, reachable, start_xy, world[best])


def depth_point(depth, intrinsics, xyxy, map_from_camera):
    """Robust central-box surface estimate, not an instance segmentation mask."""
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=float)
    h, w = depth.shape
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = max(2., (x2 - x1) * .2), max(2., (y2 - y1) * .2)
    left, right = max(0, int(cx - rx)), min(w, int(cx + rx) + 1)
    top, bottom = max(0, int(cy - ry)), min(h, int(cy + ry) + 1)
    if right <= left or bottom <= top:
        return None
    patch = depth[top:bottom, left:right]
    valid = np.isfinite(patch) & (patch > .15) & (patch < 8.)
    if valid.sum() < max(8, patch.size * .3):
        return None
    z = float(np.median(patch[valid]))
    optical = np.linalg.solve(intrinsics, [cx, cy, 1.]) * z
    return (map_from_camera @ np.r_[optical, 1.])[:3]
