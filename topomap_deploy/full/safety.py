"""Immutable known-space witness and independent motion-side route checks."""

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometrySnapshot:
    mask: np.ndarray
    origin_xy: np.ndarray
    voxel_size: float
    map_epoch: str
    revision: int
    sequence: int
    stamp_ns: int

    @classmethod
    def from_planner(cls, planner, frame, revision):
        from src.dual_dynamic_navigation.executor import known_mask
        mask, origin = known_mask(planner).copy(), planner._vol_origin[:2].copy()
        mask.setflags(write=False)
        origin.setflags(write=False)
        return cls(mask, origin, planner._voxel_size, frame.map_epoch, revision, frame.sequence, frame.stamp_ns)

    @property
    def digest(self):
        return hashlib.sha256(np.packbits(self.mask).tobytes() + str(self.mask.shape).encode()).hexdigest()

    def validate_route(self, route, frame, *, now_ns, sensor_timeout_s, map_timeout_s, start_tolerance_m):
        from src.route_guidance import visible
        route.validate()
        frame.validate()
        if (route.map_epoch != self.map_epoch or frame.map_epoch != self.map_epoch
                or route.map_revision != self.revision or route.observation_sequence != self.sequence
                or route.certificate["geometry_sha256"] != self.digest):
            raise ValueError("stale or mismatched geometry certificate")
        if not 0 <= now_ns - frame.stamp_ns <= sensor_timeout_s * 1e9:
            raise ValueError("stale sensor")
        if not 0 <= now_ns - self.stamp_ns <= map_timeout_s * 1e9:
            raise ValueError("stale planning map")
        if np.linalg.norm(frame.map_from_base[:2, 3] - route.path_map_xy[0]) > start_tolerance_m:
            raise ValueError("robot moved since route planning")
        points = (route.path_map_xy - self.origin_xy) / self.voxel_size
        if not visible(self.mask, points[0], points[0]) or not all(
                visible(self.mask, a, b) for a, b in zip(points, points[1:])):
            raise ValueError("uncertified motion segment")

    def allows_body_segment(self, start_xy, end_xy, radius_m):
        """Conservative swept circular body; unknown/out-of-bounds are blocked."""
        from src.route_guidance import visible
        # A square enclosing the circular body is intentionally conservative.
        radius = int(np.ceil(radius_m / self.voxel_size))
        a = (np.asarray(start_xy) - self.origin_xy) / self.voxel_size
        b = (np.asarray(end_xy) - self.origin_xy) / self.voxel_size
        return all(visible(self.mask, a + (x, y), b + (x, y))
                   for x in range(-radius, radius + 1) for y in range(-radius, radius + 1))
