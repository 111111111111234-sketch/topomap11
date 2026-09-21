"""MuJoCo RGB-D sweeps + actual LightNav MPC behind the full-core ports.

Explicit start only. One worker owns the renderer, physics and controller. An
independent wall-clock watchdog invalidates leases even while inference/MPC is
blocked. If the physics worker itself stalls, simulated time also stops; on
resumption its stale command is discarded before mj_step. This is NOT a Go2
hardware watchdog or a quadruped gait controller.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import math
import sys
import threading
import time

import numpy as np

from ..control import bounded_command, depth_obstacle
from .contracts import Frame, MotionFeedback, ObservationCycle


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def load_mpc(lightnav_dir, limits):
    package = Path(lightnav_dir).resolve() / "robot_deploy/src/vln_mpc"
    if not (package / "vln_mpc/mpc.py").is_file():
        raise ValueError("LightNav checkout does not contain the MPC controller")
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from vln_mpc.mpc import MPCController, build_pose_aligned_reference
    return (MPCController(horizon=5, dt_s=limits.dt_s, w_max=limits.w_max,
                          a_max_v=.3, a_max_w=.6, q_weights=(10., 10., 1.), r_weights=(.1, .1)),
            build_pose_aligned_reference)


@dataclass(frozen=True)
class SimulationLimits:
    dt_s: float = .1
    v_max: float = .2
    w_max: float = .4
    body_radius_m: float = .36
    sensor_timeout_s: float = 1.
    map_timeout_s: float = 120.
    route_timeout_s: float = 45.
    solver_timeout_s: float = 1.
    position_tolerance_m: float = .06
    yaw_tolerance_rad: float = .04
    stop_speed_m_s: float = .01
    stop_speed_rad_s: float = .02
    scan_views: int = 12
    scan_timeout_s: float = 60.

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("simulation safety limits must be finite and positive")
        if not isinstance(self.scan_views, int) or self.scan_views < 8:
            raise ValueError("at least eight scan bearings required")


class MujocoPorts:
    """One synchronized implementation of SensorPort AND MotionPort."""

    def __init__(self, *, world_factory, controller_factory, geometry_provider,
                 map_epoch="mujoco-map-1", limits=None):
        self.world_factory, self.controller_factory = world_factory, controller_factory
        self.geometry_provider = geometry_provider
        self.map_epoch, self.limits = map_epoch, limits or SimulationLimits()
        self._condition = threading.Condition()
        self._worker = self._watchdog = None
        self._closing = False
        self._error = None
        self._latest = None
        self._stopped = True
        self._generation = 0
        self._request = self._finishing = self._feedback = None
        self._scan = self._scan_result = None
        self._geometry = None
        self._deadline = self._last_progress = 0.
        self._previous = (0., 0.)

    def start(self):
        with self._condition:
            if self._worker is not None:
                raise RuntimeError("simulation ports are single-use")
            self._worker = threading.Thread(target=self._run, name="topomap-mujoco", daemon=True)
            self._watchdog = threading.Thread(target=self._watch, name="topomap-watchdog", daemon=True)
            self._worker.start()
            self._watchdog.start()
        self._wait(lambda: self._latest is not None, 30., "simulator initialization")
        return self

    def _wait(self, predicate, timeout, operation):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate():
                if self._error is not None:
                    raise RuntimeError(f"simulation worker failed during {operation}") from self._error
                if self._closing:
                    raise RuntimeError("simulation is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(operation)
                self._condition.wait(min(.1, remaining))
            if self._error is not None:
                raise RuntimeError("simulation worker failed") from self._error

    @property
    def latest(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("simulation worker failed") from self._error
            if self._latest is None:
                raise RuntimeError("simulation ports have not started")
            return self._latest

    def _invalidate(self, status):
        # Called with condition held; never retain a solver's old authority.
        self._generation += 1
        if self._request is not None:
            self._finishing = (self._request, status)
        self._request = None
        self._scan = None
        self._previous = (0., 0.)

    def stop(self, reason):
        with self._condition:
            sequence = -1 if self._latest is None else self._latest.sequence
            self._invalidate("cancelled")
            self._condition.notify_all()
        self._wait(lambda: self._stopped and self._latest is not None
                   and self._latest.sequence > sequence, 5., f"stop: {reason}")

    def collect(self, *, first_cycle):
        del first_cycle  # same full measured sweep at each stopped mapping cycle
        self.stop("observation_sweep")
        with self._condition:
            base = self._latest.map_from_base
            yaw = math.atan2(base[1, 0], base[0, 0])
            self._scan_result = None
            self._scan = {"targets": [yaw + 2 * math.pi * i / self.limits.scan_views
                                      for i in range(1, self.limits.scan_views + 1)],
                          "frames": [], "origin": base[:2, 3].copy()}
            self._deadline = time.monotonic() + self.limits.scan_timeout_s
            self._generation += 1
        try:
            self._wait(lambda: self._scan_result is not None, self.limits.scan_timeout_s + 1., "measured sweep")
            cycle = self._scan_result
            cycle.validate()
            return cycle
        except Exception:
            self.stop("sweep_failed")
            raise

    def submit(self, request):
        # Deep-copy mutable arrays before crossing into the control thread.
        request = replace(request, path_map_xy=np.asarray(request.path_map_xy, float).copy(),
                          terminal_map_xy=np.asarray(request.terminal_map_xy, float).copy(),
                          certificate=dict(request.certificate))
        request.path_map_xy.setflags(write=False)
        request.terminal_map_xy.setflags(write=False)
        geometry, frame = self.geometry_provider(), self.latest
        if geometry is None:
            raise ValueError("no original TSDF geometry available")
        geometry.validate_route(request, frame, now_ns=time.monotonic_ns(),
            sensor_timeout_s=self.limits.sensor_timeout_s, map_timeout_s=self.limits.map_timeout_s,
            start_tolerance_m=self.limits.position_tolerance_m)
        if not geometry.allows_body_segment(request.path_map_xy[0], request.path_map_xy[-1],
                                            self.limits.body_radius_m):
            raise ValueError("original route has insufficient observed robot clearance")
        with self._condition:
            if self._request is not None or self._scan is not None or not self._stopped:
                raise RuntimeError("motion submission requires stopped, idle ports")
            self._generation += 1
            self._request, self._geometry = request, geometry
            self._feedback = self._finishing = None
            self._deadline = time.monotonic() + self.limits.route_timeout_s
            self._previous = (0., 0.)

    def poll(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("simulation worker failed") from self._error
            result, self._feedback = self._feedback, None
            return result

    def _watch(self):
        while True:
            with self._condition:
                if self._closing:
                    return
                now = time.monotonic()
                if (self._request is not None or self._scan is not None) and (
                        now > self._deadline or now - self._last_progress > self.limits.sensor_timeout_s):
                    self._invalidate("timeout")
                self._condition.wait(.05)

    def _capture(self, world, sequence):
        rgb, depth, intrinsics, camera, base = world.observe()
        # Capture time uses the same monotonic clock as leases. No qpos assignment.
        arrays = [np.array(value, copy=True) for value in (rgb, depth, intrinsics, camera, base)]
        for value in arrays:
            value.setflags(write=False)
        frame = Frame(sequence, time.monotonic_ns(), self.map_epoch, *arrays)
        frame.validate()
        return frame

    def _route_command(self, request, frame, world, controller, reference_builder):
        limits, pose = self.limits, world.pose()
        if (time.monotonic_ns() - frame.stamp_ns > limits.sensor_timeout_s * 1e9
                or time.monotonic_ns() - self._geometry.stamp_ns > limits.map_timeout_s * 1e9
                or self.geometry_provider() is not self._geometry):
            return (0., 0.), "blocked"
        target = request.terminal_map_xy
        if np.linalg.norm(pose[:2] - target) <= limits.position_tolerance_m:
            yaw_error = 0. if request.terminal_yaw_rad is None else wrap(request.terminal_yaw_rad - pose[2])
            if abs(yaw_error) <= limits.yaw_tolerance_rad:
                return (0., 0.), "reached"
            return (0., float(np.clip(2 * yaw_error, -limits.w_max, limits.w_max))), None
        delta = target - request.path_map_xy[0]
        yaw = float(np.arctan2(delta[1], delta[0]))
        count = max(2, int(np.linalg.norm(delta) / .05) + 1)
        xy = np.linspace(request.path_map_xy[0], target, count)
        path = np.column_stack((xy, np.full(count, yaw)))
        reference = reference_builder(path, pose, horizon=5, weights=(10., 10., 1.))
        began = time.monotonic()
        command, _ = controller.solve(pose, reference, self._previous, v_max=limits.v_max)
        if time.monotonic() - began > limits.solver_timeout_s:
            return (0., 0.), "timeout"
        if not np.isfinite(command).all():
            return (0., 0.), "blocked"
        command = bounded_command(*command, limits.v_max, limits.w_max)
        if command[0] > 0 and depth_obstacle(frame.depth_m, frame.intrinsics,
                                            np.linalg.inv(frame.map_from_base) @ frame.map_from_camera_optical):
            return (0., 0.), "blocked"
        # Validate the next physical control interval including tracking error.
        next_xy = pose[:2] + command[0] * limits.dt_s * np.array([math.cos(pose[2]), math.sin(pose[2])])
        if not self._geometry.allows_body_segment(pose[:2], next_xy, limits.body_radius_m):
            return (0., 0.), "blocked"
        return command, None

    def _run(self):
        world = None
        try:
            world = self.world_factory()
            controller, reference_builder = self.controller_factory()
            sequence = 0
            while True:
                began = time.monotonic()
                frame = self._capture(world, sequence)
                sequence += 1
                linear, angular = world.speed()
                stopped = linear <= self.limits.stop_speed_m_s and angular <= self.limits.stop_speed_rad_s
                with self._condition:
                    if self._closing:
                        break
                    self._latest, self._stopped, self._last_progress = frame, stopped, began
                    if self._finishing is not None and stopped:
                        request, status = self._finishing
                        if status == "reached" and np.linalg.norm(world.pose()[:2] - request.terminal_map_xy) > self.limits.position_tolerance_m:
                            status = "blocked"  # braking drift cannot fake arrival
                        if (status == "reached" and request.terminal_yaw_rad is not None
                                and abs(wrap(world.pose()[2] - request.terminal_yaw_rad)) > self.limits.yaw_tolerance_rad):
                            status = "blocked"
                        self._feedback = MotionFeedback(request.goal_id, request.route_id, status, frame, True)
                        self._finishing = None
                    request, scan, generation = self._request, self._scan, self._generation
                    self._condition.notify_all()
                command, status = (0., 0.), None
                if world.contacts:
                    raise RuntimeError("simulator reported body contact; execution stopped")
                if request is not None:
                    command, status = self._route_command(request, frame, world, controller, reference_builder)
                elif scan is not None:
                    if np.linalg.norm(world.pose()[:2] - scan["origin"]) > .1:
                        raise RuntimeError("view sweep drifted outside capture-position tolerance")
                    error = wrap(scan["targets"][0] - world.pose()[2])
                    if abs(error) > self.limits.yaw_tolerance_rad:
                        command = (0., float(np.clip(2 * error, -self.limits.w_max, self.limits.w_max)))
                    elif stopped:
                        with self._condition:
                            if generation == self._generation:
                                scan["frames"].append(frame)
                                scan["targets"].pop(0)
                                if not scan["targets"]:
                                    self._scan_result = ObservationCycle(tuple(scan["frames"]))
                                    self._scan = None
                                    self._condition.notify_all()
                with self._condition:
                    if generation != self._generation or self._closing:
                        command = (0., 0.)
                    elif status is not None:
                        self._finishing = (request, status)
                        self._request = None
                        command = (0., 0.)
                    self._previous = command
                    # Bound the command authority check-to-step race to one DT.
                    world.step(command, self.limits.dt_s)
                    self._condition.wait(max(0., self.limits.dt_s - (time.monotonic() - began)))
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._invalidate("blocked")
                self._condition.notify_all()
        finally:
            if world is not None:
                try:
                    world.step((0., 0.), .5)
                finally:
                    world.close()

    def close(self):
        try:
            if self._worker is not None and self._error is None:
                self.stop("close")
        finally:
            with self._condition:
                self._closing = True
                self._invalidate("cancelled")
                self._condition.notify_all()
            for thread in (self._worker, self._watchdog):
                if thread is not None:
                    thread.join(timeout=5.)
                    if thread.is_alive():
                        raise RuntimeError("simulation worker has not shut down")
