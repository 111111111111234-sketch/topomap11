"""Simulation-specific checks: pixels, depth-only map, and contact physics."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("imageio")

from topomap_deploy.rgbd_mapping import DepthGridMapper
from topomap_deploy.sim_world import World
from topomap_deploy.simulation import MarkerDetector


def test_marker_detector_requires_image_evidence():
    detector = MarkerDetector()
    image = np.zeros((100, 100, 3), np.uint8)
    assert detector.detect(image) == []
    image[20:50, 30:60] = [200, 10, 10]
    result = detector.detect(image)
    assert result[0].label == "red_marker"
    assert result[0].box == [30, 20, 60, 50]


def test_depth_mapper_does_not_prefill_unobserved_map():
    mapper = DepthGridMapper()
    assert np.all(mapper.grid.cells == -1)
    k = np.array([[50., 0, 30], [0, 50, 20], [0, 0, 1]])
    camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, .6], [0, 0, 0, 1.]])
    depth = np.full((40, 60), 2.)
    mapper.integrate(depth, k, camera, [0, 0])
    assert np.count_nonzero(mapper.grid.cells == 0) > 0
    assert np.count_nonzero(mapper.grid.cells == 100) > 0
    behind = np.rint(mapper.grid.world_to_cell([-1., 0.])).astype(int)
    assert mapper.grid.cells[tuple(behind)] == -1


def test_physics_contact_blocks_commanded_motion_and_braking_settles():
    world = World(obstacle=True)
    try:
        world.step((.3, 0.), 8.)
        assert 1.0 < world.pose()[0] < 1.52
        assert world.contacts > 0
        assert world.max_penetration < .04
        world.step((0., 0.), 1.)
        assert np.linalg.norm(world.data.qvel[:2]) < .01
        assert world.steps == 1800
    finally:
        world.close()


def test_rendered_rgbd_and_camera_transform_use_metric_optical_coordinates():
    world = World(obstacle=False)
    try:
        rgb, depth, k, camera, base = world.observe()
        assert rgb.shape == (240, 320, 3) and depth.shape == (240, 320)
        assert rgb.dtype == np.uint8
        assert np.isfinite(depth).all() and np.median(depth) > .1
        assert np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3))
        forward = camera[:3, 2]
        assert forward[0] > .8 and forward[2] < 0
        assert MarkerDetector().detect(rgb)
        ray = camera[:3, :3] @ np.linalg.solve(k, [160., 220., 1.])
        expected_floor_depth = -camera[2, 3] / ray[2]
        # Also bounds the macOS GL depth fallback's error in the operating range.
        assert abs(float(depth[220, 160]) - expected_floor_depth) < .02
        assert world.target_distance_for_evaluation() == pytest.approx(np.hypot(4.4, .55))
    finally:
        world.close()


@pytest.mark.parametrize("case", ["direct", "detour", "sensor_loss", "planner_loss"])
def test_closed_loop_scenarios(tmp_path, case):
    pytest.importorskip("casadi")
    pytest.importorskip("torch")
    lightnav = Path(__file__).resolve().parents[2] / "LightNav-0"
    if not (lightnav / "robot_deploy/src/vln_mpc/vln_mpc/mpc.py").is_file():
        pytest.skip("closed-loop integration requires the sibling LightNav-0 checkout")
    from topomap_deploy.simulation import run_case
    report = run_case(case, tmp_path / case, lightnav, max_seconds=60., video=False)
    assert report["passed"], report
    if case == "planner_loss":
        assert report["http_errors"] > 0
    if case == "detour":
        assert report["hgr_hypothesis_nodes_created"] > 0
