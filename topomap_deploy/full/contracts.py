"""Contracts for measured input, original algorithm stages, and physical output.

No simulator object IDs, ground-truth goal positions, or full-world navmesh enter
these contracts. All map/base/camera transforms refer to the capture timestamp.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np


class State(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    MAPPING = "MAPPING"
    HYPOTHESIZING = "HYPOTHESIZING"
    SELECTING = "SELECTING"
    ROUTING = "ROUTING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    ARRIVED = "ARRIVED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class IntegrationPending(RuntimeError):
    """An explicitly missing adapter; never substitute a diagnostic backend."""


@dataclass(frozen=True)
class Goal:
    goal_id: str
    kind: str  # object / description / image; same task types as GOAT
    question: str
    category: str
    image: Optional[str] = None  # original selector's encoded image representation

    def metadata(self) -> dict:
        if self.kind not in ("object", "description", "image"):
            raise ValueError("unsupported goal kind")
        if not self.goal_id or not self.question:
            raise ValueError("goal_id and question are required")
        if self.kind == "object" and not self.category:
            raise ValueError("object goal needs a category")
        if self.kind == "image" and not self.image:
            raise ValueError("image goal needs image evidence")
        return {"question": self.question, "task_type": self.kind,
                "class": self.category, "image": self.image}


@dataclass(frozen=True)
class Frame:
    sequence: int
    stamp_ns: int
    map_epoch: str  # change on localization reset / discontinuous map correction
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    map_from_camera_optical: np.ndarray  # map z-up; optical x-right/y-down/z-forward
    map_from_base: np.ndarray  # base x-forward/y-left/z-up

    def validate(self):
        if self.sequence < 0 or self.stamp_ns <= 0 or not self.map_epoch:
            raise ValueError("invalid capture identity")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("RGB must be HWC uint8")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("depth must be registered to RGB")
        if not (np.isfinite(self.depth_m) & (self.depth_m > 0)).any():
            raise ValueError("depth has no usable samples")
        k = np.asarray(self.intrinsics)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
                or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
                or not np.allclose([k[0, 1], k[1, 0]], 0)):
            raise ValueError("invalid camera intrinsics")
        for transform in (self.map_from_camera_optical, self.map_from_base):
            value = np.asarray(transform)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError("invalid capture-time transform")
            rotation = value[:3, :3]
            if (not np.allclose(value[3], [0, 0, 0, 1])
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
                raise ValueError("capture transform must be rigid")


@dataclass(frozen=True)
class ObservationCycle:
    frames: Tuple[Frame, ...]  # actual captures in temporal order; main view last

    @property
    def latest(self) -> Frame:
        if not self.frames:
            raise ValueError("empty observation cycle")
        return self.frames[-1]

    def validate(self):
        epoch = self.latest.map_epoch
        previous = None
        for frame in self.frames:
            frame.validate()
            if frame.map_epoch != epoch:
                raise ValueError("map changed within observation cycle")
            if previous and (frame.sequence <= previous.sequence or frame.stamp_ns <= previous.stamp_ns):
                raise ValueError("captures must advance monotonically")
            previous = frame


@dataclass(frozen=True)
class Selection:
    goal_id: str
    source_id: str
    kind: str  # snapshot or frontier
    original_choice: Any  # actual src.tsdf_planner.SnapShot / Frontier, not a copy by class label
    intent_id: str = ""  # original HypothesisAwareNavigator's source-bound intent
    requires_verification: bool = False  # copied from AuthorizedTask, not inferred from goal type alone


@dataclass(frozen=True)
class RouteRequest:
    goal_id: str
    route_id: str
    source_id: str
    map_epoch: str
    observation_sequence: int
    map_revision: int
    path_map_xy: np.ndarray
    terminal_map_xy: np.ndarray
    is_final_terminal: bool  # intermediate Place arrival is NOT goal arrival
    certificate: Dict[str, Any]
    intent_id: str = ""
    terminal_yaw_rad: Optional[float] = None  # measured heading required before terminal capture

    def validate(self):
        path = np.asarray(self.path_map_xy)
        terminal = np.asarray(self.terminal_map_xy)
        if (not self.goal_id or not self.route_id or not self.source_id or not self.intent_id
                or not self.map_epoch or self.observation_sequence < 0 or self.map_revision < 0):
            raise ValueError("route identity is incomplete")
        if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
            raise ValueError("route must contain finite metric XY points")
        if terminal.shape != (2,) or not np.isfinite(terminal).all() or not np.allclose(path[-1], terminal):
            raise ValueError("route terminal mismatch")
        if not self.certificate.get("geometry_sha256") or self.certificate.get("fallback", True):
            raise ValueError("known-space certificate required; no oracle fallback")
        if self.terminal_yaw_rad is not None and not np.isfinite(self.terminal_yaw_rad):
            raise ValueError("invalid terminal heading")


@dataclass(frozen=True)
class MotionFeedback:
    goal_id: str
    route_id: str
    status: str  # reached / blocked / timeout / cancelled
    frame: Frame  # fresh, measured AFTER execution; never a proposed TSDF next pose
    stopped: bool


@dataclass(frozen=True)
class FeedbackDecision:
    goal_complete: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    verified_intent_id: Optional[str] = None  # only after the original navigator accepts confirmed -> stop


class SensorPort(Protocol):
    def collect(self, *, first_cycle: bool) -> ObservationCycle:
        """Execute the configured view sweep, returning real captures, main view last."""
        ...


class MotionPort(Protocol):
    def stop(self, reason: str) -> None:
        """Idempotent emergency stop; must also cancel any installed route."""
        ...

    def submit(self, request: RouteRequest) -> None:
        """Nonblocking; adapter independently enforces freshness/depth/watchdog limits."""
        ...

    def poll(self) -> Optional[MotionFeedback]:
        """Return measured feedback, not a prediction from the path planner."""
        ...


class OriginalCorePort(Protocol):
    def begin_goal(self, goal: Goal) -> None: ...
    def integrate(self, cycle: ObservationCycle) -> None: ...
    def update_memory(self) -> None: ...
    def update_frontiers_and_hypotheses(self) -> None: ...
    def select(self, goal: Goal) -> Optional[Selection]: ...
    def route(self, selection: Selection) -> Optional[RouteRequest]: ...
    def feedback(self, selection: Selection, route: RouteRequest,
                 feedback: MotionFeedback) -> FeedbackDecision: ...
    def cancel_goal(self, reason: str) -> None: ...


class EventSink(Protocol):
    def __call__(self, event: dict) -> None: ...
