"""Read-only frontier views from an explicit measured observation cycle."""

import numpy as np

from src.coordinates import pos_habitat_to_normal


class MissingCapturedView(RuntimeError):
    pass


def capture_name(frame):
    return f"capture-{frame.map_epoch}-{frame.sequence}.png"


class CapturedViews:
    def __init__(self, *, position_tolerance_m=.1, yaw_tolerance_deg=25.):
        self.position_tolerance_m = position_tolerance_m
        self.yaw_tolerance_rad = np.deg2rad(yaw_tolerance_deg)
        self.cycle = None
        self.used = []

    def bind(self, cycle):
        cycle.validate()
        self.cycle, self.used = cycle, []

    def frontier_observation(self, pts, view_dir, camera_tilt=0.):
        if self.cycle is None:
            raise MissingCapturedView("No captured observation cycle")
        origin = pos_habitat_to_normal(np.asarray(pts))[:2]
        direction = pos_habitat_to_normal(np.asarray(view_dir))[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise MissingCapturedView("Undefined frontier bearing")
        candidates = []
        for frame in self.cycle.frames:
            if np.linalg.norm(frame.map_from_base[:2, 3] - origin) > self.position_tolerance_m:
                continue
            forward = frame.map_from_camera_optical[:2, 2]
            if np.linalg.norm(forward) < 1e-8:
                continue
            error = float(np.arccos(np.clip(forward @ direction / (norm * np.linalg.norm(forward)), -1., 1.)))
            # Requested bearing must also be inside the actual camera image.
            half_fov = min(np.arctan(frame.intrinsics[0, 2] / frame.intrinsics[0, 0]),
                           np.arctan((frame.rgb.shape[1] - frame.intrinsics[0, 2]) / frame.intrinsics[0, 0]))
            if error <= min(self.yaw_tolerance_rad, half_fov * .8):
                candidates.append((error, -frame.sequence, frame))
        if not candidates:
            raise MissingCapturedView("Frontier requires an uncaptured bearing; collect a new sweep")
        frame = min(candidates, key=lambda item: item[:2])[2]
        self.used.append(capture_name(frame))
        return {"color_sensor": frame.rgb, "depth_sensor": frame.depth_m,
                "capture_sequence": frame.sequence}
