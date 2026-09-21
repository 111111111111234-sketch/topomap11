"""Orchestrate the full stages; this module contains NO navigation policy.

Single owner, serialized API. Sensor/model calls occur with the robot stopped.
Motion ports require an independent watchdog; this coordinator is not a safety
certification, a real-time control loop, or a replacement for robot emergency stop.
"""

import time

from .contracts import State


class FullTopomapWorkflow:
    def __init__(self, core, sensor, motion, events, *, clock=time.monotonic,
                 execution_timeout_s=15., terminal_tolerance_m=.1):
        if execution_timeout_s <= 0 or terminal_tolerance_m <= 0:
            raise ValueError("positive execution limits required")
        self.core, self.sensor, self.motion, self.events = core, sensor, motion, events
        self.clock = clock
        self.execution_timeout_s = execution_timeout_s
        self.terminal_tolerance_m = terminal_tolerance_m
        self.state = State.IDLE
        self.goal = self.selection = self.pending_route = None
        self.last_frame = None
        self.map_epoch = None
        self.step = 0
        self.submitted_at = None

    def _transition(self, state, reason, **details):
        self.state = state
        self.events({"event": "state", "state": state.value, "reason": reason,
                     "goal_id": None if self.goal is None else self.goal.goal_id,
                     "step": self.step, **details})

    def begin(self, goal):
        goal.metadata()
        self.motion.stop("begin_goal")
        if self.goal is not None:
            self.core.cancel_goal("replaced_by_new_goal")
        self.goal = goal
        self.selection = self.pending_route = None
        self.step = 0
        # Map epoch and last capture persist across goals; a new goal does not
        # magically repair a map/SLAM discontinuity or admit replayed frames.
        try:
            self.core.begin_goal(goal)
            self._transition(State.OBSERVING, "goal_started")
        except Exception:
            self._transition(State.ERROR, "begin_goal_failed")
            raise

    def cancel(self, reason="operator_cancel"):
        self.motion.stop(reason)
        try:
            self.core.cancel_goal(reason)
        finally:
            self.pending_route = self.selection = None
            self._transition(State.CANCELLED, reason)

    def _accept_frame(self, frame):
        frame.validate()
        if self.map_epoch is not None and frame.map_epoch != self.map_epoch:
            raise ValueError("map_epoch_changed: reset scene memory before resuming")
        if self.last_frame is not None and (frame.sequence <= self.last_frame.sequence
                                           or frame.stamp_ns <= self.last_frame.stamp_ns):
            raise ValueError("stale observation")
        self.map_epoch, self.last_frame = frame.map_epoch, frame

    def advance(self):
        """One logical stage cycle or one nonblocking feedback poll; no busy loop."""
        if self.state in (State.IDLE, State.ARRIVED, State.ERROR, State.CANCELLED, State.BLOCKED):
            return self.state
        try:
            if self.state == State.EXECUTING:
                self._receive_feedback()
            else:
                self._observe_and_plan()
        except Exception as exc:
            self.motion.stop("pipeline_error")
            self.pending_route = None
            self._transition(State.ERROR, type(exc).__name__)
            raise
        return self.state

    def _observe_and_plan(self):
        self.motion.stop("observe_and_compute")
        self._transition(State.OBSERVING, "capture_actual_views")
        cycle = self.sensor.collect(first_cycle=self.step == 0)
        cycle.validate()
        for frame in cycle.frames:
            self._accept_frame(frame)
        self.step += 1
        self._transition(State.MAPPING, "original_conceptgraph_and_tsdf")
        self.core.integrate(cycle)
        self.core.update_memory()
        self._transition(State.HYPOTHESIZING, "original_frontier_and_hgr")
        self.core.update_frontiers_and_hypotheses()
        self._transition(State.SELECTING, "original_hgr_selector")
        self.selection = self.core.select(self.goal)
        if self.selection is None:
            self._transition(State.BLOCKED, "no_original_hgr_selection")
            return
        if self.selection.goal_id != self.goal.goal_id or not self.selection.intent_id:
            raise ValueError("selection requires the current goal and an authorized intent")
        self._transition(State.ROUTING, "adaptive_authorized_source_route")
        route = self.core.route(self.selection)
        if route is None:
            self._transition(State.BLOCKED, "selected_source_has_no_known_route")
            return
        route.validate()
        if (route.goal_id != self.goal.goal_id or route.source_id != self.selection.source_id
                or route.intent_id != self.selection.intent_id
                or route.map_epoch != self.map_epoch
                or route.observation_sequence != self.last_frame.sequence):
            raise ValueError("route has stale or mismatched provenance")
        self.pending_route = route
        self.submitted_at = self.clock()
        self.motion.submit(route)
        self._transition(State.EXECUTING, "certified_route_submitted", route_id=route.route_id)

    def _receive_feedback(self):
        if self.clock() - self.submitted_at > self.execution_timeout_s:
            self.motion.stop("execution_timeout")
            self.pending_route = None
            self._transition(State.BLOCKED, "execution_timeout")
            return
        feedback = self.motion.poll()
        if feedback is None:
            return
        route = self.pending_route
        if feedback.goal_id != self.goal.goal_id or feedback.route_id != route.route_id:
            raise ValueError("feedback belongs to a different route")
        self.motion.stop("execution_feedback")
        self._accept_frame(feedback.frame)
        if not feedback.stopped:
            raise ValueError("motion adapter has not confirmed stopped state")
        if feedback.status not in ("reached", "blocked", "timeout", "cancelled"):
            raise ValueError("unknown execution feedback")
        if feedback.status == "reached":
            import numpy as np
            if np.linalg.norm(feedback.frame.map_from_base[:2, 3] - route.terminal_map_xy) > self.terminal_tolerance_m:
                raise ValueError("reported arrival disagrees with measured pose")
            if route.terminal_yaw_rad is not None:
                yaw = np.arctan2(feedback.frame.map_from_base[1, 0], feedback.frame.map_from_base[0, 0])
                error = (yaw - route.terminal_yaw_rad + np.pi) % (2 * np.pi) - np.pi
                if abs(error) > .1:
                    raise ValueError("terminal observation has the wrong measured heading")
        self._transition(State.VERIFYING, "fresh_post_motion_evidence")
        outcome = self.core.feedback(self.selection, route, feedback)
        self.pending_route = None
        if outcome.goal_complete:
            if feedback.status != "reached" or not route.is_final_terminal:
                raise ValueError("intermediate or failed motion cannot complete a goal")
            if (self.selection.requires_verification
                    and outcome.verified_intent_id != self.selection.intent_id):
                raise ValueError("Adaptive historical target requires confirmation of the same intent")
            self._transition(State.ARRIVED, outcome.reason)
        elif feedback.status == "cancelled":
            self._transition(State.CANCELLED, outcome.reason)
        else:
            # Geometry/semantic recovery remains owned by the original backend.
            self._transition(State.OBSERVING, outcome.reason)
