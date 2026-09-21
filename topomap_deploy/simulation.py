"""Run a measured MuJoCo navigation loop with an explicit diagnostic detector.

RGB-D -> observed map -> real topomap Navigator/HGR -> HTTP -> LightNav MPC ->
actuator commands -> mj_step -> new RGB-D. No ROS or real-robot control is started.
"""

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .control import depth_obstacle, motion_allowed, validate_reply
from .protocol import Observation
from .rgbd_mapping import DepthGridMapper
from .runtime import Detection, Navigator
from .semantics import HGRMemory
from .server import Service, make_handler
from .sim_world import World


class MarkerDetector:
    """Image-only connected-component color detector; NOT YOLO/VLM evaluation."""

    def detect(self, rgb):
        values = np.asarray(rgb, dtype=float)
        mask = ((values[:, :, 0] > 85) & (values[:, :, 0] > values[:, :, 1] * 1.8)
                & (values[:, :, 0] > values[:, :, 2] * 1.8))
        labels, _ = ndimage.label(mask)
        result = []
        for label, region in enumerate(ndimage.find_objects(labels), 1):
            if region is None or np.count_nonzero(labels[region] == label) < 30:
                continue
            rows, cols = region
            result.append(Detection("red_marker", .99, [cols.start, rows.start, cols.stop, rows.stop]))
        return result


def load_controller(lightnav_dir):
    package = Path(lightnav_dir).resolve() / "robot_deploy/src/vln_mpc"
    if not (package / "vln_mpc/mpc.py").is_file():
        raise ValueError("--lightnav-dir must point to the existing LightNav-0 checkout")
    sys.path.insert(0, str(package))
    from vln_mpc.mpc import MPCController, build_pose_aligned_reference
    controller = MPCController(horizon=5, dt_s=.1, w_max=.4, a_max_v=.3, a_max_w=.6,
                               q_weights=(10., 10., 1.), r_weights=(.1, .1))
    return controller, build_pose_aligned_reference


def diagnostic_service():
    from src.hypothesis_node_predictor import HypothesisNodePredictor
    from src.semantic_critic import SemanticCritic
    cfg = {"robot_radius_m": .4, "target_confidence": .55, "confirmation_frames": 3,
           "arrival_distance_m": 1.3, "approach_distance_m": .9, "path_horizon_m": 1.}
    hypothesis = {"enable_vlm_hypothesis_prediction": False, "enable_vlm_causality_diagnosis": False,
                  "enable_cascade_deletion": True, "min_dependency_confidence": .6,
                  "residual_weight_category": .5, "residual_weight_feature": 0.,
                  "residual_weight_objects": .5}

    def factory(detector, goal):
        memory = HGRMemory(hypothesis, HypothesisNodePredictor(hypothesis), SemanticCritic)
        return Navigator(detector, memory, goal, cfg)

    return Service(MarkerDetector(), factory, ["red_marker"])


def post(url, path, payload):
    request = Request(url + path, json.dumps(payload, allow_nan=False).encode(),
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=30.) as response:
        return json.load(response)


def to_world(waypoints, pose):
    points = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    world = points.copy()
    world[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]]) + pose[:2]
    world[:, 2] += pose[2]
    return world


def panel(world, rgb, grid, trajectory, status, command):
    canvas = Image.new("RGB", (960, 320), "#101b26")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "TOPOMAP | MuJoCo contact dynamics | PLANAR PROXY / diagnostic marker / ideal pose", fill="white")
    canvas.paste(Image.fromarray(rgb), (0, 45))
    canvas.paste(Image.fromarray(world.overhead()), (320, 45))
    colors = np.full((*grid.cells.shape, 3), [95, 105, 115], np.uint8)
    colors[grid.cells == 0] = [225, 232, 236]
    colors[grid.cells == 100] = [20, 28, 36]
    raster = Image.fromarray(colors).resize((320, 240), Image.Resampling.NEAREST)
    mapping = ImageDraw.Draw(raster)
    if trajectory:
        cells = grid.world_to_cell(np.asarray(trajectory)[:, :2])
        points = [(float(rc[1] * 320 / grid.cells.shape[1]), float(rc[0] * 240 / grid.cells.shape[0])) for rc in cells]
        if len(points) > 1:
            mapping.line(points, fill=(0, 140, 150), width=3)
        x, y = points[-1]
        mapping.ellipse((x-4, y-4, x+4, y+4), fill="orange")
    canvas.paste(raster, (640, 45))
    draw.text((10, 30), "RGB camera", fill="white")
    draw.text((330, 30), "Physics scene (evaluation view)", fill="white")
    draw.text((650, 30), "Depth-built map + actual trajectory", fill="white")
    draw.text((12, 292), f"sim={world.time:.1f}s  {status}  v={command[0]:.2f}m/s  w={command[1]:.2f}rad/s  contacts={world.contacts}", fill="white")
    return np.asarray(canvas)


def run_case(name, output, lightnav_dir, max_seconds=100., video=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    service = diagnostic_service()
    handler = make_handler(service, "")
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server_stopped = False
    url = f"http://127.0.0.1:{server.server_port}"
    world = None
    writer = None
    wall_start = time.monotonic()
    try:
        controller, reference_builder = load_controller(lightnav_dir)
        world = World(obstacle=name == "detour")
        mapper = DepthGridMapper()
        session = post(url, "/reset", {"goal": "red_marker"})["session"]
        writer = imageio.get_writer(str(output / "closed_loop.mp4"), fps=10, codec="libx264", quality=7) if video else None
        trace, trajectory = [], []
        seq = 0
        rgb = None
        initial_distance = world.target_distance_for_evaluation()
        # A real in-place scan bootstraps the depth-only map. It does not rotate
        # the virtual camera independently or assign the robot pose.
        for scan_tick in range(110):
            world.step((0., .6))
            if scan_tick % 2 == 0:
                rgb, depth, k, camera, base = world.observe()
                mapper.integrate(depth, k, camera, base[:2, 3])
                if writer:
                    writer.append_data(panel(world, rgb, mapper.grid, trajectory, "BOOTSTRAP SCAN", (0., .6)))
        world.step((0., 0.), .5)
        start = world.time
        sensor_at = plan_at = -1e6
        obs = None
        world_path = None
        command = previous = (0., 0.)
        arrived = False
        obstacle = False
        fault_at = None
        first_zero_after_fault = None
        fault_position = None
        pre_fault_travel = 0.
        http_errors = 0
        reason = "starting"
        counts = Counter()
        tick = 0
        while world.time - start < max_seconds:
            now = world.time
            if name in ("sensor_loss", "planner_loss") and fault_at is None and now - start >= 5.:
                fault_at, fault_position = now, world.pose().copy()
                if len(trajectory) > 1:
                    pre_fault_travel = float(np.linalg.norm(np.diff(np.asarray(trajectory)[:, :2], axis=0), axis=1).sum())
                if name == "planner_loss":
                    # Close the real listening socket, then keep attempting HTTP
                    # requests. Failed requests must not refresh the plan lease.
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3.)
                    server_stopped = True
            sensor_failed = name == "sensor_loss" and fault_at is not None
            planner_failed = name == "planner_loss" and fault_at is not None
            if tick % 2 == 0 and not sensor_failed:
                rgb, depth, k, camera, base = world.observe()
                mapper.integrate(depth, k, camera, base[:2, 3])
                seq += 1
                obs = Observation(seq, int((now + 1.) * 1e9), rgb, depth, k, camera, base, mapper.grid)
                sensor_at = now
                obstacle = depth_obstacle(depth, k, np.linalg.inv(base) @ camera)
            response = None
            if tick % 4 == 0 and obs is not None and not sensor_failed:
                try:
                    response = post(url, "/step", {"session": session, "observation": obs.encode()})
                    validate_reply(response, obs.seq, obs.stamp_ns)
                except (URLError, OSError):
                    if not planner_failed:
                        raise
                    http_errors += 1
            if response is not None:
                counts[response["state"]] += 1
                reason = response["reason"]
                arrived = response["state"] == "ARRIVED"
                if response["state"] == "RUNNING":
                    capture_pose = np.array([obs.map_from_base[0, 3], obs.map_from_base[1, 3],
                                            math.atan2(obs.map_from_base[1, 0], obs.map_from_base[0, 0])])
                    world_path = to_world(response["waypoints"], capture_pose)
                    plan_at = now
                else:
                    world_path = None
            allowed = (not obstacle and not arrived and world_path is not None and
                       motion_allowed(enabled=True, running=True, now=now, sensor_at=sensor_at,
                                      odom_at=now, plan_at=plan_at, sensor_timeout=.6, plan_timeout=1.5))
            if allowed:
                reference = reference_builder(world_path, world.pose(), horizon=5, weights=(10., 10., 1.))
                command, _ = controller.solve(world.pose(), reference, previous, v_max=.2)
                previous = command
            else:
                command = previous = (0., 0.)
            if fault_at is not None:
                if command != (0., 0.):
                    first_zero_after_fault = None
                elif first_zero_after_fault is None:
                    first_zero_after_fault = now
            if sensor_failed and not allowed:
                reason = "sensor_stale: stopped"
            elif planner_failed and not allowed:
                reason = "plan_stale: stopped"
            trajectory.append(world.pose().tolist())
            trace.append({"t": now, "pose": world.pose().tolist(), "command": list(command),
                          "reason": reason, "depth_stop": obstacle, "motion_allowed": allowed,
                          "known_cells": int(np.count_nonzero(mapper.grid.cells >= 0)),
                          "target_surface_distance_m_eval_only": world.target_distance_for_evaluation()})
            if tick % 2 == 0 and writer:
                frame = panel(world, rgb, mapper.grid, trajectory,
                              f"{name}: {'ARRIVED' if arrived else reason}", command)
                writer.append_data(frame)
            if tick % 50 == 0:
                print(json.dumps({"case": name, "sim_s": round(now-start, 1), "pose": world.pose().round(2).tolist(),
                                  "state": reason, "depth_stop": obstacle, "command": np.round(command, 3).tolist()}), flush=True)
            if arrived or (fault_at is not None and now - fault_at >= 3.):
                break
            world.step(command)
            tick += 1
        stop_position = world.pose().copy()
        world.step((0., 0.), 1.)
        stop_drift = float(np.linalg.norm(world.pose()[:2] - stop_position[:2]))
        travelled = float(np.linalg.norm(np.diff(np.asarray(trajectory)[:, :2], axis=0), axis=1).sum()) if len(trajectory) > 1 else 0.
        last_frame = panel(world, rgb, mapper.grid, trajectory, f"{name} | STOPPED", (0., 0.))
        Image.fromarray(last_frame).save(output / "final.png")
        if writer:
            for _ in range(10):
                writer.append_data(last_frame)
        passed = bool(arrived and world.target_distance_for_evaluation() <= 1.3
                      and travelled > 1.5 and world.contacts == 0 and stop_drift < .04)
        if name in ("sensor_loss", "planner_loss"):
            delay = None if first_zero_after_fault is None else first_zero_after_fault - fault_at
            passed = bool(pre_fault_travel > .3 and delay is not None and delay <= (0.7 if name == "sensor_loss" else 1.6)
                          and world.contacts == 0 and np.linalg.norm(world.data.qvel[:2]) < .01
                          and abs(world.data.qvel[2]) < .01
                          and (name != "planner_loss" or http_errors > 0))
        else:
            delay = None
        report = {"case": name, "passed": passed, "arrived": arrived,
                  "physics": "MuJoCo mj_step, force-limited planar actuator proxy; NOT Go2 gait",
                  "perception": "RGB-only red-marker detector; NOT YOLO or VLM inference",
                  "mapping": "depth ray integration; flat-floor footprint prior; no scene-geometry map",
                  "localization": "ideal simulator pose; SLAM localization NOT evaluated",
                  "semantic_prediction": "real HGR heuristic mode; no paid API calls",
                  "controller": "LightNav vln_mpc MPCController/CasADi/IPOPT",
                  "transport": "real loopback HTTP with Observation v1 JPEG/depth/map payloads",
                  "ros2": "not part of this Mac run",
                  "initial_target_surface_distance_m_eval_only": initial_distance,
                  "final_target_surface_distance_m_eval_only": world.target_distance_for_evaluation(),
                  "travelled_m": travelled, "stop_drift_m": stop_drift,
                  "contact_samples": world.contacts, "max_penetration_m": world.max_penetration,
                  "physics_steps": world.steps, "simulation_s": world.time,
                  "wall_s": time.monotonic()-wall_start, "planner_states": dict(counts),
                  "fault_s": fault_at, "sustained_zero_delay_s": delay,
                  "pre_fault_travelled_m": pre_fault_travel, "http_errors": http_errors,
                  "fault_to_stop_distance_m": None if fault_position is None else float(np.linalg.norm(world.pose()[:2] - fault_position[:2])),
                  "final_linear_speed_m_s": float(np.linalg.norm(world.data.qvel[:2])),
                  "final_angular_speed_rad_s": abs(float(world.data.qvel[2])),
                  "known_cells": int(np.count_nonzero(mapper.grid.cells >= 0)),
                  "hgr_hypothesis_nodes_created": service.navigator.memory.graph.stats["hypothesis_nodes_created"]}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        (output / "trace.json").write_text(json.dumps(trace))
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        if writer:
            writer.close()
        if world:
            world.close()
        if not server_stopped:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightnav-dir", required=True)
    parser.add_argument("--output", required=True, help="new output directory; never overwrite an earlier run")
    parser.add_argument("--case", choices=("direct", "detour", "sensor_loss", "planner_loss", "all"), default="all")
    parser.add_argument("--max-seconds", type=float, default=100.)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--backend", choices=("diagnostic",), required=True,
                        help="explicit acknowledgement: marker perception + heuristic HGR, no trained navigation model")
    args = parser.parse_args()
    if args.max_seconds <= 0 or not math.isfinite(args.max_seconds):
        parser.error("--max-seconds must be positive")
    names = ("direct", "detour", "sensor_loss", "planner_loss") if args.case == "all" else (args.case,)
    reports = [run_case(name, Path(args.output) / name, args.lightnav_dir,
                        max_seconds=args.max_seconds, video=not args.no_video) for name in names]
    Path(args.output, "summary.json").write_text(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(report["passed"] for report in reports) else 1)


if __name__ == "__main__":
    main()
