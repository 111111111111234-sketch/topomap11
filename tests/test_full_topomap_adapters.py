"""Offline adapter contracts. Added in the implementation task, NOT run yet.

Fixtures substitute observations/physics, not a deployment semantic backend.
Navigator/RouteGuidance/known-space executor tests use the original classes.
"""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time

import numpy as np
import pytest

from topomap_deploy.full.contracts import Frame, Goal, MotionFeedback, ObservationCycle, RouteRequest, Selection
from topomap_deploy.full.lifecycle import OriginalAdaptiveLifecycle
from topomap_deploy.full.observations import CapturedViews, MissingCapturedView
from topomap_deploy.full.original import OriginalStageCalls
from topomap_deploy.full.safety import GeometrySnapshot
from topomap_deploy.full.sim_io import MujocoPorts, SimulationLimits


ROOT = Path(__file__).resolve().parents[1]


def capture(sequence=1, xy=(2., 2.), yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    base = np.eye(4)
    base[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    base[:3, 3] = [*xy, .25]
    optical = base.copy()
    optical[:3, :3] = base[:3, :3] @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    optical[2, 3] = .6
    return Frame(sequence, sequence * 1_000_000_000, "map", np.zeros((8, 8, 3), np.uint8),
                 np.ones((8, 8)), np.array([[4., 0, 4], [0, 4., 4], [0, 0, 1.]]), optical, base)


def test_frontier_views_are_real_captures_not_robot_actions():
    views = CapturedViews()
    a, b = capture(yaw=0.), capture(2, yaw=np.pi / 2)
    views.bind(ObservationCycle((a, b)))
    # Habitat vector (0,0,-1) is map north (+Y).
    obs = views.frontier_observation(np.array([2., .25, -2.]), np.array([0., 0., -1.]))
    assert obs["capture_sequence"] == 2 and obs["color_sensor"] is b.rgb
    with pytest.raises(MissingCapturedView):
        views.frontier_observation(np.array([2., .25, -2.]), np.array([-1., 0., 0.]))
    with pytest.raises(MissingCapturedView):
        views.frontier_observation(np.array([6., .25, -2.]), np.array([1., 0., 0.]))


def test_optical_rays_and_original_conceptgraph_pixel_convention_agree():
    frame = capture(yaw=.7)
    position, legacy = OriginalStageCalls.legacy_capture(frame)
    from src.coordinates import pos_habitat_to_normal
    assert np.allclose(pos_habitat_to_normal(position), frame.map_from_base[:3, 3])
    u, v, depth = 2., 1., 2.
    k = frame.intrinsics
    original_cy = frame.rgb.shape[0] - 1 - k[1, 2]
    gl_point = np.array([(u-k[0, 2])*depth/k[0, 0],
                         (frame.rgb.shape[0]-1-v-original_cy)*depth/k[1, 1], -depth, 1.])
    optical_point = np.array([(u-k[0, 2])*depth/k[0, 0], (v-k[1, 2])*depth/k[1, 1], depth, 1.])
    assert np.allclose(pos_habitat_to_normal((legacy @ gl_point)[:3]),
                       (frame.map_from_camera_optical @ optical_point)[:3])


def test_no_usable_depth_rejected_even_if_inf_is_positive():
    f = capture()
    depth = np.zeros_like(f.depth_m)
    depth[0, 0] = np.inf
    with pytest.raises(ValueError, match="usable"):
        replace(f, depth_m=depth).validate()


class PlannerFixture:
    def __init__(self):
        self.island = self.unoccupied = np.ones((40, 40), bool)
        self.occupied = np.zeros((40, 40), bool)
        self._vol_origin = np.zeros(3)
        self._vol_bnds = np.array([[0., 8.], [0., 8.], [0., 3.]])
        self._voxel_size = .2
        self.max_point = self.target_point = self.look_at_point = None
        self.frontiers = []
        from src.hypothesis_graph import HypothesisGraph
        self.hypothesis_graph = HypothesisGraph({})

    def habitat2voxel(self, point):
        from src.coordinates import pos_habitat_to_normal
        return np.rint(pos_habitat_to_normal(point) / self._voxel_size).astype(int)

    def set_next_navigation_point(self, choice, pts, objects, cfg, pathfinder,
                                  target_point_override, look_at_point, known_space_only):
        assert pathfinder is None and known_space_only
        self.max_point = choice
        self.target_point = np.asarray(target_point_override)
        self.look_at_point = np.asarray(look_at_point)
        return True

    def agent_step(self, *args, **kwargs):
        raise AssertionError("physical adapter must not execute speculative TSDF.agent_step")


def lifecycle_fixture(kind="snapshot", verification=False):
    from src.dual_dynamic_navigation.unified_navigator import HypothesisAwareNavigator
    from src.dual_dynamic_navigation.task_planner import ActionKind, AuthorizedTask
    from src.place_topology import PlaceTopology
    from src.persistent_belief_topology import PersistentBeliefTopology
    from topomap_deploy.full.config import load_config
    cfg = load_config(ROOT / "deploy/full/adaptive_mac.yaml")
    navigator = HypothesisAwareNavigator(dict(cfg.dual_dynamic_navigation))
    planner = PlannerFixture()
    place = PlaceTopology(1.5, .2, "route_only")
    place.observe_pose([10., 10.], 0)
    lifecycle = OriginalAdaptiveLifecycle(navigator=navigator, place_topology=place,
        persistent_topology=PersistentBeliefTopology({}, voxel_size=.2))
    scene = SimpleNamespace(objects={7: {}}, object_id_aliases={})
    frame = capture()
    stages = SimpleNamespace(planner=planner, cfg=cfg, scene=scene, cycle=ObservationCycle((frame,)),
        map_revision=1, legacy_capture=OriginalStageCalls.legacy_capture,
        integrate_feedback=Mock(return_value=["chair"]), verify_frontier=Mock(return_value={}),
        verify_selected_entity=Mock(return_value=("confirmed", {})))
    goal = Goal("goal", "description", "Find this chair", "chair")
    stages.goal = goal
    lifecycle.begin_goal(goal, stages)
    choice = SimpleNamespace(cluster=[7], topo_id="frontier:1", orientation=np.array([1., 0.]))
    action = ActionKind.EXPLORE if kind == "frontier" else (ActionKind.REVISIT if verification else ActionKind.APPROACH)
    task = AuthorizedTask(action, "source", "7", "digest",
        {"terminal": [20., 10.], "look_at": [25., 10.]}, choice,
        requires_verification=verification)
    navigator.install(task, planner.hypothesis_graph)
    stages._require_active_selection = lambda selection: navigator.active_intent
    stages.selection_for_active_intent = lambda c: selection
    selection = Selection(goal.goal_id, task.source_id, kind, choice,
                          navigator.active_intent.intent_id, verification)
    return lifecycle, stages, selection


def test_route_is_one_original_macro_step_not_teleport_or_full_path():
    lifecycle, stages, selection = lifecycle_fixture()
    original_target = [20., 10.]
    route = lifecycle.make_route(selection, stages)
    assert route is not None and not route.is_final_terminal
    assert np.linalg.norm(route.terminal_map_xy - route.path_map_xy[0]) <= 1. + 1e-8
    assert np.allclose(stages.planner.target_point, original_target)
    assert lifecycle.navigator.active_intent.motion_steps == 0
    assert np.allclose(stages.cycle.latest.map_from_base[:2, 3], [2., 2.])
    result = lifecycle.apply_feedback(selection, route,
        MotionFeedback("goal", route.route_id, "reached", capture(2, xy=route.terminal_map_xy), True), stages)
    assert not result.goal_complete
    assert lifecycle.navigator.active_intent.motion_steps == 1
    with pytest.raises(ValueError, match="already applied"):
        lifecycle.apply_feedback(selection, route,
            MotionFeedback("goal", route.route_id, "reached", capture(3), True), stages)


def test_original_three_motion_frontier_window_reselects():
    lifecycle, stages, selection = lifecycle_fixture(kind="frontier")
    lifecycle.navigator.active_intent.motion_steps = 3
    choose = Mock(return_value=None)
    assert lifecycle.select_or_continue(stages.goal, stages, choose) is None
    choose.assert_called_once()
    assert lifecycle.navigator.active_intent is None
    assert lifecycle.navigator.stats["bounded_intent_expirations"] == 1


def final_route(lifecycle, stages, selection):
    route = lifecycle.make_route(selection, stages)
    # Fixture places the authorized terminal at one macro segment's endpoint.
    lifecycle.navigator.active_intent.task.route["terminal"] = (route.terminal_map_xy / .2).tolist()
    return lifecycle.make_route(selection, stages)


@pytest.mark.parametrize("verdict,complete", [("confirmed", True), ("rejected", False), ("error", False)])
def test_selective_goal_verdict_uses_original_feedback(verdict, complete):
    lifecycle, stages, selection = lifecycle_fixture(verification=True)
    route = final_route(lifecycle, stages, selection)
    stages.verify_selected_entity.return_value = verdict, {"reason": verdict}
    result = lifecycle.apply_feedback(selection, route,
        MotionFeedback("goal", route.route_id, "reached", capture(2, xy=route.terminal_map_xy), True), stages)
    assert result.goal_complete is complete
    assert result.verified_intent_id == (selection.intent_id if complete else None)
    if not complete:
        assert lifecycle.navigator.active_intent is None


def test_uncertain_requests_another_original_unchecked_view():
    lifecycle, stages, selection = lifecycle_fixture(verification=True)
    route = final_route(lifecycle, stages, selection)
    stages.verify_selected_entity.return_value = "uncertain", {}
    lifecycle.navigator.resolver.object_view = Mock(return_value={"status": "valid",
        "terminal": [20., 15.], "look_at": [25., 10.]})
    result = lifecycle.apply_feedback(selection, route,
        MotionFeedback("goal", route.route_id, "reached", capture(2, xy=route.terminal_map_xy), True), stages)
    assert not result.goal_complete and result.reason == "adaptive_new_view_required"
    assert lifecycle.navigator.active_intent.intent_id == selection.intent_id
    assert lifecycle.navigator.active_intent.verification_views == 1
    assert lifecycle.navigator.checked_views["7"]
    assert np.allclose(lifecycle.navigator.active_intent.task.route["terminal"], [20., 15.])


def test_blocked_motion_uses_original_recovery_and_does_not_count_stationary_ticks():
    lifecycle, stages, selection = lifecycle_fixture()
    route = lifecycle.make_route(selection, stages)
    lifecycle.navigator.recover = Mock(return_value=True)
    result = lifecycle.apply_feedback(selection, route,
        MotionFeedback("goal", route.route_id, "blocked", capture(2), True), stages)
    assert not result.goal_complete and result.reason == "route_recovered"
    assert lifecycle.navigator.active_intent.motion_steps == 0
    lifecycle.navigator.recover.assert_called_once()


def test_cross_goal_scene_memory_is_not_reset():
    lifecycle, stages, _ = lifecycle_fixture()
    memory, places = lifecycle.navigator.scene_map, lifecycle.place_topology
    lifecycle.begin_goal(Goal("next", "object", "Find a table", "table"), stages)
    assert lifecycle.navigator.scene_map is memory and lifecycle.place_topology is places
    assert stages.scene.objects == {7: {}}
    assert lifecycle.navigator.active_intent is None


def test_motion_side_rechecks_geometry_and_capture_age():
    lifecycle, stages, selection = lifecycle_fixture()
    route = lifecycle.make_route(selection, stages)
    geometry, frame = lifecycle.geometry, capture(2)
    params = dict(now_ns=2_000_000_000, sensor_timeout_s=1., map_timeout_s=3., start_tolerance_m=.1)
    geometry.validate_route(route, frame, **params)
    with pytest.raises(ValueError, match="geometry"):
        geometry.validate_route(replace(route, map_revision=99), frame, **params)
    with pytest.raises(ValueError, match="stale sensor"):
        geometry.validate_route(route, frame, **{**params, "now_ns": 5_000_000_000})
    blocked = geometry.mask.copy()
    blocked[15, 10] = False
    assert not replace(geometry, mask=blocked).allows_body_segment([2., 2.], [3., 2.], .2)


def test_ports_have_no_startup_side_effect_and_stop_invalidates_solver_authority():
    world_factory, controller_factory = Mock(), Mock()
    ports = MujocoPorts(world_factory=world_factory, controller_factory=controller_factory,
                        geometry_provider=lambda: None)
    world_factory.assert_not_called()
    controller_factory.assert_not_called()
    generation = ports._generation
    with ports._condition:
        ports._request = SimpleNamespace(route_id="pending")
        ports._invalidate("timeout")
    assert ports._request is None and ports._generation > generation
    assert ports._finishing[1] == "timeout"


def test_threaded_ports_return_new_measured_stopped_frame_without_native_simulator():
    """Fake physics verifies threading/feedback only, not MuJoCo navigation."""
    class StationaryWorld:
        contacts = 0
        commands = []

        def observe(self):
            f = capture()
            return f.rgb, f.depth_m, f.intrinsics, f.map_from_camera_optical, f.map_from_base

        def pose(self): return np.array([2., 2., 0.])
        def speed(self): return 0., 0.
        def step(self, command, seconds): self.commands.append(command)
        def close(self): pass

    holder = {}
    ports = MujocoPorts(world_factory=StationaryWorld,
        controller_factory=lambda: (Mock(), Mock()), geometry_provider=lambda: holder.get("geometry"))
    try:
        ports.start()
        frame = ports.latest
        geometry = GeometrySnapshot.from_planner(PlannerFixture(), frame, 1)
        holder["geometry"] = geometry
        route = RouteRequest("g", "r", "source", "mujoco-map-1", frame.sequence, 1,
            np.array([[2., 2.], [2., 2.]]), np.array([2., 2.]), True,
            {"geometry_sha256": geometry.digest, "fallback": False}, "intent", 0.)
        ports.submit(route)
        result = None
        deadline = time.monotonic() + 3.
        while result is None and time.monotonic() < deadline:
            result = ports.poll()
            time.sleep(.01)
        assert result is not None and result.status == "reached" and result.stopped
        assert result.frame.sequence > route.observation_sequence
        assert np.allclose(result.frame.map_from_base[:2, 3], [2., 2.])
        assert all(command == (0., 0.) for command in StationaryWorld.commands)
    finally:
        ports.close()


@pytest.mark.parametrize("filename", ["src/scene_goatbench.py", "src/tsdf_planner.py",
    "src/tsdf_base.py", "src/utils.py", "src/query_vlm_hypothesis.py"])
def test_external_perception_modules_do_not_eagerly_import_habitat(filename):
    tree = ast.parse((ROOT / filename).read_text())
    # Runtime-only simulator branches are allowed to retain their local imports.
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any((getattr(node, "module", "") or "").startswith(("habitat", "src.habitat"))
                   or any(alias.name.startswith("habitat") for alias in node.names) for node in imports)


def test_default_configuration_cannot_start_simulation_or_load_models():
    from topomap_deploy.full.bootstrap import start_simulation
    from topomap_deploy.full.config import load_config
    from topomap_deploy.full.contracts import IntegrationPending
    cfg = load_config(ROOT / "deploy/full/adaptive_mac.yaml")
    with pytest.raises(IntegrationPending, match="disabled"):
        start_simulation(cfg, root=ROOT, lightnav_dir=ROOT.parent / "LightNav-0", events=lambda event: None)
