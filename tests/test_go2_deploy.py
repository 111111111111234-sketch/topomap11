"""CPU deployment contracts; no ROS, Habitat, weights, or model API calls."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.hypothesis_graph import SemanticDistribution
from topomap_deploy.control import bounded_command, depth_obstacle, motion_allowed, validate_reply
from topomap_deploy.planning import connected_free, depth_point, frontiers, route, traversable
from topomap_deploy.protocol import Grid, Observation, body_waypoints, pack_array
from topomap_deploy.runtime import Detection, Navigator
from topomap_deploy.semantics import HGRMemory, frontier_crop, room_from_labels
from topomap_deploy.server import Service, validate_config


def observation(seq=1, yaw=0.):
    cells = np.zeros((80, 80), dtype=np.int8)
    grid = Grid(cells, .1, [-4, -4, 0.])
    base = np.eye(4)
    base[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    # Optical x=right, y=down, z=forward; robot x=forward, y=left, z=up.
    optical = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, .5], [0, 0, 0, 1.]])
    return Observation(seq, seq * 1_000_000_000, np.zeros((40, 60, 3), np.uint8),
                       np.full((40, 60), 2., np.float32), np.array([[50., 0, 30], [0, 50, 20], [0, 0, 1]]),
                       base @ optical, base, grid)


def config():
    return json.loads((Path(__file__).parents[1] / "deploy/go2/server.json").read_text())


def test_observation_round_trip_and_depth_units():
    obs = observation()
    restored = Observation.decode(obs.encode())
    np.testing.assert_allclose(restored.depth, 2.)
    np.testing.assert_array_equal(restored.grid.cells, obs.grid.cells)
    np.testing.assert_allclose(restored.map_from_camera, obs.map_from_camera)


@pytest.mark.parametrize("change", [
    lambda p: p.update(version=2), lambda p: p.update(seq=True),
    lambda p: p.update(frame_id="odom"),
    lambda p: p.update(intrinsics=[[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    lambda p: p.update(map_from_base=np.zeros((4, 4)).tolist()),
    lambda p: p.update(depth_f32_m=pack_array(np.zeros((40, 60)), "<f4")),
    lambda p: p.update(depth_f32_m={"shape": [100000, 100000], "data": ""}),
    lambda p: p["depth_f32_m"].update(data="bad!"),
])
def test_invalid_observations_rejected(change):
    payload = observation().encode()
    change(payload)
    with pytest.raises((ValueError, TypeError)):
        Observation.decode(payload)


def test_invalid_depth_stays_unknown():
    obs = observation()
    obs.depth[0, :4] = [float("nan"), float("inf"), -1, 900]
    restored = Observation.decode(obs.encode())
    np.testing.assert_array_equal(restored.depth[0, :4], [0, 0, 0, 0])


def test_rotated_grid_round_trip():
    grid = Grid(np.zeros((20, 30)), .1, [2., -3., np.pi/2])
    points = np.array([[0, 0], [4.2, 8.1], [19, 29]])
    np.testing.assert_allclose(grid.world_to_cell(grid.cell_to_world(points)), points, atol=1e-10)


def test_body_waypoints_are_absolute_and_heading_correct():
    obs = observation(yaw=np.pi/2)
    obs.map_from_base[:2, 3] = [3., 2.]
    points = body_waypoints([[3., 2.], [3., 3.], [2., 3.]], obs.map_from_base,
                           spacing=.25, max_distance=2.)
    np.testing.assert_allclose(points[-1][:2], [1., 1.], atol=1e-10)
    assert points[-1][2] == pytest.approx(np.pi/2)
    assert points[2][0] == pytest.approx(.5)  # not a sequence of incremental .25 steps


def test_depth_projection_uses_camera_extrinsics():
    obs = observation(yaw=np.pi/2)
    point = depth_point(obs.depth, obs.intrinsics, [20, 10, 40, 30], obs.map_from_camera)
    np.testing.assert_allclose(point, [0, 2, .5], atol=1e-8)
    assert depth_point(np.zeros_like(obs.depth), obs.intrinsics, [20, 10, 40, 30], obs.map_from_camera) is None


def test_unknown_wall_and_narrow_corridor_block_robot_footprint():
    obs = observation()
    obs.grid.cells[:, 45] = -1
    mask = connected_free(obs.grid, [0, 0], .4)
    assert route(obs.grid, mask, [0, 0], [2, 0]) is None
    obs.grid.cells[:, 45] = 100
    obs.grid.cells[39:42, 45] = 0  # 30 cm door cannot admit a Go2 footprint
    assert route(obs.grid, connected_free(obs.grid, [0, 0], .4), [0, 0], [2, 0]) is None


def test_obstacle_detour_is_certified_and_map_boundaries_blocked():
    obs = observation()
    obs.grid.cells[36:44, 47] = 100
    mask = connected_free(obs.grid, [0, 0], .4)
    path = route(obs.grid, mask, [0, 0], [2, 0])
    assert path is not None
    assert max(abs(path[:, 1])) > .4
    assert not traversable(obs.grid, .4)[0].any()


def test_frontiers_have_reachable_standoff():
    obs = observation()
    obs.grid.cells[:, 60:] = -1
    mask = connected_free(obs.grid, [0, 0], .4)
    found = frontiers(obs.grid, mask)
    assert found
    for center, approach, _ in found:
        assert route(obs.grid, mask, [0, 0], approach) is not None
        assert approach[0] < center[0]


def test_no_synthetic_frontier_views():
    obs = observation()
    assert frontier_crop(obs, [2, 0]) is not None
    assert frontier_crop(obs, [-2, 0]) is None
    assert frontier_crop(obs, [0, 2]) is None


def test_ambiguous_chair_detection_cannot_verify_room_type():
    assert room_from_labels(["chair"]) is None
    assert room_from_labels(["sofa", "chair"]) == "living_room"


class Predictor:
    def batch_predict_frontiers(self, frontiers, context, history):
        return {f.frontier_id: SemanticDistribution(["living_room", "kitchen"], np.array([.8, .2])) for f in frontiers}


class Critic:
    def __init__(self, cfg, graph):
        self.graph, self.calls = graph, []

    def verify_hypothesis_node_arrival(self, node, observation, objects):
        self.calls.append(observation)


def memory():
    return HGRMemory(config()["hypothesis"], Predictor(), Critic)


class Detector:
    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect(self, rgb):
        return self.detections


def test_runtime_navigates_from_real_contract_without_simulator():
    obs = observation()
    obs.grid.cells[:, 60:] = -1
    nav = Navigator(Detector(), memory(), "chair", config())
    result = nav.step(Observation.decode(obs.encode()))
    assert result["state"] == "RUNNING"
    assert result["reason"] == "hgr_frontier"
    assert not result["stop"]
    validate_reply(result, obs.seq, obs.stamp_ns)
    assert nav.memory.graph.hypothesis_nodes


def test_hgr_records_real_context_dependency_and_arrival_evidence():
    obs = observation()
    mem = memory()
    mem.observe(obs, ["sofa"])
    ranked = mem.rank([(np.array([2., 0]), np.array([1.5, 0]), 20)], obs, ["sofa"], "chair")
    node_id = ranked[0][2]
    assert mem.graph.nodes[node_id].parent_ids
    obs2 = observation(2)
    mem.arrived(node_id, obs2, [])
    assert not mem.critic.calls
    mem.arrived(node_id, obs2, ["refrigerator"])
    assert mem.critic.calls[0]["observation_id"] == "rgb:2000000000"
    assert mem.critic.calls[0]["evidence_source"] == "post_motion_terminal"


def test_arrival_requires_repeated_fresh_depth_supported_target():
    detector = Detector([Detection("chair", .9, [20, 10, 40, 30])])
    nav = Navigator(detector, memory(), "chair", config())
    for i in (1, 2, 3):
        obs = observation(i)
        obs.depth[:] = 1.1
        result = nav.step(obs)
        assert result["stop"] == (i == 3)
    with pytest.raises(ValueError, match="increase"):
        nav.step(obs)


def test_missing_detection_breaks_confirmation_chain():
    detector = Detector([Detection("chair", .9, [20, 10, 40, 30])])
    nav = Navigator(detector, memory(), "chair", config())
    for i in range(1, 5):
        detector.detections = [] if i == 2 else [Detection("chair", .9, [20, 10, 40, 30])]
        obs = observation(i)
        obs.depth[:] = 1.1
        assert not nav.step(obs)["stop"]


def test_target_approach_does_not_stall_outside_arrival_threshold():
    detector = Detector([Detection("chair", .9, [20, 10, 40, 30])])
    nav = Navigator(detector, memory(), "chair", config())
    obs = observation()
    obs.depth[:] = 1.45
    result = nav.step(obs)
    assert result["state"] == "RUNNING"
    endpoint = np.asarray(result["waypoints"][-1][:2])
    assert np.linalg.norm(endpoint - np.array([1.45, 0])) < config()["arrival_distance_m"]


def test_unknown_start_cannot_move_and_empty_map_never_means_arrival():
    nav = Navigator(Detector(), memory(), "chair", config())
    obs = observation()
    obs.grid.cells[:] = -1
    result = nav.step(obs)
    assert result["state"] == "BLOCKED" and not result["stop"] and not result["waypoints"]


def test_session_reset_rejects_old_requests_and_validates_categories():
    service = Service(Detector(), lambda d, goal: Navigator(d, memory(), goal, config()), ["chair"])
    old = service.dispatch("/reset", {"goal": "chair"})["session"]
    new = service.dispatch("/reset", {"goal": "chair"})["session"]
    assert old != new
    with pytest.raises(ValueError, match="expired"):
        service.dispatch("/step", {"session": old, "observation": observation().encode()})
    result = service.dispatch("/step", {"session": new, "observation": observation().encode()})
    assert result["state"] == "BLOCKED"
    service.dispatch("/stop", {"session": new})
    with pytest.raises(ValueError):
        service.dispatch("/reset", {"goal": "walk upstairs"})


def test_stale_feedback_and_limits_gate_motion():
    kwargs = dict(enabled=True, running=True, now=10., sensor_at=9.9, odom_at=9.9,
                  plan_at=9., sensor_timeout=.6, plan_timeout=1.5)
    assert motion_allowed(**kwargs)
    for key, value in (("enabled", False), ("running", False), ("sensor_at", 0.),
                       ("odom_at", 0.), ("plan_at", 0.), ("plan_at", 11.)):
        assert not motion_allowed(**dict(kwargs, **{key: value}))
    assert bounded_command(4., -5., .2, .4) == (.2, -.4)
    assert bounded_command(float("nan"), 0., .2, .4) == (0., 0.)


def test_depth_stop_volume_uses_metric_depth_and_base_extrinsics():
    obs = observation()
    assert not depth_obstacle(obs.depth, obs.intrinsics, obs.map_from_camera)
    obs.depth[:] = .45
    assert depth_obstacle(obs.depth, obs.intrinsics, obs.map_from_camera)
    assert depth_obstacle(np.zeros_like(obs.depth), obs.intrinsics, obs.map_from_camera)


def test_malformed_reply_fails_closed():
    valid = dict(seq=1, capture_stamp_ns=1, state="RUNNING", stop=False, waypoints=[[.1, 0., 0.]])
    for change in ({"seq": 2}, {"waypoints": []}, {"stop": True},
                   {"waypoints": [[float("nan"), 0, 0]]}, {"waypoints": [[20, 0, 0]]}):
        with pytest.raises(ValueError):
            validate_reply(dict(valid, **change), 1, 1)


def test_checked_in_configuration_is_valid():
    validate_config(config())
    bad = config()
    bad["robot_radius_m"] = float("nan")
    with pytest.raises(ValueError):
        validate_config(bad)
