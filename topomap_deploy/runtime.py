"""Category ObjectNav: RGB-D detections + HGR frontier memory + known-space A*."""

from dataclasses import dataclass

import numpy as np

from .planning import connected_free, depth_point, frontiers, object_approach, route
from .protocol import body_waypoints


@dataclass
class Detection:
    label: str
    confidence: float
    box: list


class YoloDetector:
    def __init__(self, weights, classes, device):
        from pathlib import Path
        import torch
        from ultralytics import YOLOWorld
        if not Path(weights).is_file():
            raise ValueError(f"YOLO-World checkpoint not found: {weights}; download it explicitly first")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; configure device=cpu explicitly for slow diagnostics")
        self.model = YOLOWorld(weights)
        self.classes = classes
        self.model.set_classes(classes)
        self.device = device

    def detect(self, rgb):
        # Ultralytics ndarray inputs are BGR, while the wire protocol is RGB.
        result = self.model.predict(rgb[:, :, ::-1].copy(), device=self.device,
                                    conf=.45, verbose=False)[0]
        return [Detection(self.classes[int(label)], float(confidence), box.tolist())
                for label, confidence, box in zip(result.boxes.cls.cpu().numpy(),
                                                 result.boxes.conf.cpu().numpy(),
                                                 result.boxes.xyxy.cpu().numpy())]


class Navigator:
    def __init__(self, detector, memory, goal, cfg):
        self.detector, self.memory, self.goal, self.cfg = detector, memory, goal, cfg
        self.last_seq = self.last_stamp = 0
        self.active_frontier = None
        self.confirmations = 0
        self.target = None
        self.missing_target_frames = 0
        self.done = False

    def step(self, observation):
        if observation.seq <= self.last_seq or observation.stamp_ns <= self.last_stamp:
            raise ValueError("observation sequence and capture timestamp must both increase")
        self.last_seq, self.last_stamp = observation.seq, observation.stamp_ns
        base = observation.map_from_base[:2, 3]
        result = {"seq": observation.seq, "capture_stamp_ns": observation.stamp_ns,
                  "waypoints": [], "stop": False, "state": "BLOCKED", "reason": "no_known_path"}
        if self.done:
            return dict(result, state="ARRIVED", stop=True, reason="goal_confirmed")
        reachable = connected_free(observation.grid, base, self.cfg["robot_radius_m"])
        if not reachable.any():
            return dict(result, reason="robot_footprint_not_in_known_free_space")
        detections = self.detector.detect(observation.rgb)
        labels = sorted({d.label for d in detections})
        self.memory.observe(observation, labels)
        if self.active_frontier is not None:
            xy, node, stamp = self.active_frontier
            if np.linalg.norm(base - xy) < .3 and observation.stamp_ns > stamp:
                self.memory.arrived(node, observation, labels)
                self.active_frontier = None

        # Only fresh detections support arrival. A cached target can guide a
        # route, but cannot declare success without repeated current evidence.
        fresh = []
        for detection in detections:
            if detection.label != self.goal or detection.confidence < self.cfg["target_confidence"]:
                continue
            point = depth_point(observation.depth, observation.intrinsics,
                                detection.box, observation.map_from_camera)
            if point is not None:
                fresh.append(point[:2])
        if fresh:
            self.missing_target_frames = 0
            chosen = min(fresh, key=lambda p: np.linalg.norm(p - base))
            stable = self.target is not None and np.linalg.norm(chosen - self.target) < .6
            self.confirmations = self.confirmations + 1 if stable else 1
            self.target = chosen
            if (self.confirmations >= self.cfg["confirmation_frames"]
                    and np.linalg.norm(chosen - base) <= self.cfg["arrival_distance_m"]):
                self.done = True
                return dict(result, state="ARRIVED", reason="goal_confirmed", stop=True)
        else:
            self.confirmations = 0
            self.missing_target_frames += 1
            if self.missing_target_frames >= 5:
                self.target = None
        path = None
        if self.target is not None:
            path = object_approach(observation.grid, reachable, base, self.target,
                                   self.cfg["approach_distance_m"],
                                   max_distance=(self.cfg["approach_distance_m"]
                                                 + self.cfg["arrival_distance_m"]) / 2.)
        if path is not None:
            reason = "object_approach"
        else:
            candidates = frontiers(observation.grid, reachable)
            ranked = self.memory.rank(candidates, observation, labels, self.goal)
            reason = "hgr_frontier"
            if self.active_frontier is not None:
                xy, node, stamp = self.active_frontier
                path = route(observation.grid, reachable, base, xy)
                if node not in self.memory.graph.nodes:
                    path = None
            if path is None:
                self.active_frontier = None
                for _, xy, node in ranked:
                    candidate = route(observation.grid, reachable, base, xy)
                    if candidate is not None and np.linalg.norm(xy - base) > .3:
                        path = candidate
                        self.active_frontier = (xy, node, observation.stamp_ns)
                        break
        if path is None:
            return dict(result, reason="no_reachable_target_or_frontier")
        waypoints = body_waypoints(path, observation.map_from_base,
                                   spacing=.05, max_distance=self.cfg["path_horizon_m"])
        return dict(result, state="RUNNING", reason=reason, waypoints=waypoints)
