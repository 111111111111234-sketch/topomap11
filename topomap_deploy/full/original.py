"""Direct call sites into the original topomap modules, not a reduced navigator."""

from importlib import import_module
from typing import Protocol

import numpy as np

from .contracts import IntegrationPending, Selection
from .config import validate_adaptive_policy
from .observations import capture_name


# Strings keep framework inspection free of heavy imports/model downloads.
ORIGINAL_SYMBOLS = {
    "YOLOWorld": ("ultralytics", "YOLOWorld"),
    "SAM": ("ultralytics", "SAM"),
    "create_clip": ("open_clip", "create_model_and_transforms"),
    "configure_vlm_runtime": ("src.vlm_runtime", "configure_vlm_runtime"),
    "Scene": ("src.scene_goatbench", "Scene"),
    "TSDFPlanner": ("src.tsdf_planner", "TSDFPlanner"),
    "SnapShot": ("src.tsdf_planner", "SnapShot"),
    "Frontier": ("src.tsdf_planner", "Frontier"),
    "HypothesisGraph": ("src.hypothesis_graph", "HypothesisGraph"),
    "HypothesisNodePredictor": ("src.hypothesis_node_predictor", "HypothesisNodePredictor"),
    "SemanticCritic": ("src.semantic_critic", "SemanticCritic"),
    "choose": ("src.query_vlm_goatbench", "query_vlm_for_response"),
    "verify": ("src.query_vlm_hypothesis", "verify_hypothesis_node_arrival"),
    "infer_room": ("src.query_vlm_hypothesis", "infer_room_type_from_objects"),
    "PlaceTopology": ("src.place_topology", "PlaceTopology"),
    "PersistentBeliefTopology": ("src.persistent_belief_topology", "PersistentBeliefTopology"),
    "HypothesisAwareNavigator": ("src.dual_dynamic_navigation.unified_navigator", "HypothesisAwareNavigator"),
    "HGRAdapter": ("src.dual_dynamic_navigation.hgr_adapter", "HGRAdapter"),
    "RouteResolver": ("src.dual_dynamic_navigation.route_resolver", "RouteResolver"),
    "verify_entity": ("src.dual_dynamic_navigation.feedback", "verify_entity"),
    "KnownSpaceExecutor": ("src.dual_dynamic_navigation.executor", "KnownSpaceExecutor"),
    "plan_place_route": ("src.topology_navigation", "plan_place_route"),
    "PlaceRouteExecution": ("src.topology_navigation", "PlaceRouteExecution"),
    "SnapshotSourceCommitment": ("src.topology_navigation", "SnapshotSourceCommitment"),
    "resolve_object_approach": ("src.object_approach", "resolve_object_approach"),
    "resolve_frontier_approach": ("src.object_approach", "resolve_frontier_approach"),
    "resize_image": ("src.utils", "resize_image"),
}


def load_original_symbols():
    """Explicit runtime operation; missing native dependencies remain errors."""
    return {name: getattr(import_module(module), symbol)
            for name, (module, symbol) in ORIGINAL_SYMBOLS.items()}


class AdaptiveLifecycle(Protocol):
    """Bind the ORIGINAL HypothesisAwareNavigator(stage='adaptive') to real I/O.

    The navigator owns scene/goal memory, source-bound intents, deduplication,
    the three-MACRO-motion-step frontier window, and selective verification.
    Count original agent-step-equivalent physical segments, not MPC ticks or
    observation sweeps. Do not reimplement these policies in the workflow.
    """
    navigator: object  # one original HypothesisAwareNavigator, shared across goals
    place_topology: object
    remaining_steps: int

    def begin_goal(self, goal, stages): ...
    def update_topology(self, stages): ...
    def select_or_continue(self, goal, stages, adaptive_select): ...
    def make_route(self, selection, stages): ...
    def apply_feedback(self, selection, route, feedback, stages): ...
    def cancel_goal(self, reason, stages): ...


class OriginalStageCalls:
    """Stage implementations sharing original Scene/TSDF/HGR objects.

The caller must construct Scene with a measured-view adapter; the original
Habitat Scene.get_observation teleports an agent and is NOT an acceptable
physical sensor adapter. No construction/download happens in this class.
"""

    def __init__(self, *, scene, planner, critic, cfg, lifecycle, symbols=None):
        if lifecycle is None:
            raise IntegrationPending("Adaptive runner lifecycle has not been bound")
        validate_adaptive_policy(cfg)
        self.symbols = load_original_symbols() if symbols is None else symbols
        navigator = lifecycle.navigator
        if (not isinstance(navigator, self.symbols["HypothesisAwareNavigator"])
                or navigator.stage != "adaptive"):
            raise ValueError("Adaptive lifecycle must own the original HypothesisAwareNavigator")
        if navigator.config != dict(cfg.dual_dynamic_navigation):
            raise ValueError("Adaptive navigator configuration differs from the selected configuration")
        self.scene, self.planner, self.critic = scene, planner, critic
        self.cfg, self.lifecycle = cfg, lifecycle
        self.goal = self.cycle = None
        self.frame_index = 0
        self.map_revision = 0
        self.step = 0
        self.fresh_object_ids = set()
        self.fresh_labels = []
        self.ego_views = []
        self.capture_poses = {}
        self._last_integrated = None

    @staticmethod
    def legacy_capture(frame):
        """Map z-up -> Habitat y-up; optical -> legacy camera x-right/y-up/z-back.

        Only coordinate conversion, no access to simulator geometry. Original
        ConceptGraph's pixel-center convention still needs calibration tests.
        """
        habitat_from_map = np.array([[1., 0, 0, 0], [0, 0, 1., 0],
                                     [0, -1., 0, 0], [0, 0, 0, 1.]])
        optical_from_legacy = np.diag([1., -1., -1., 1.])
        position = (habitat_from_map @ frame.map_from_base)[:3, 3]
        camera = habitat_from_map @ frame.map_from_camera_optical @ optical_from_legacy
        return position, camera

    def begin_goal(self, goal):
        self.goal = goal
        # Lifecycle owns goal-only reset vs scene/episode reset; do not throw
        # away cross-goal ConceptGraph/Place/HGR memory unconditionally.
        self.lifecycle.begin_goal(goal, self)

    def integrate(self, cycle):
        cycle.validate()
        self.cycle = cycle
        self.step += 1
        provider = getattr(self.scene, "observation_provider", None)
        if provider is not None:
            provider.bind(cycle)
        self._integrate_frames(cycle.frames)

    def _integrate_frames(self, frames):
        self.fresh_object_ids = set()
        self.ego_views = []
        import torch
        for frame in frames:
            frame.validate()
            if self._last_integrated is not None:
                old = self._last_integrated
                if (frame.map_epoch != old.map_epoch or frame.sequence <= old.sequence
                        or frame.stamp_ns <= old.stamp_ns):
                    raise ValueError("perception requires monotonic captures in the same map")
            position, camera = self.legacy_capture(frame)
            map_position = frame.map_from_base[:3, 3]
            if (np.any(map_position < self.planner._vol_bnds[:, 0])
                    or np.any(map_position >= self.planner._vol_bnds[:, 1])):
                raise ValueError("measured base left the configured TSDF volume; no clipped localization")
            image_id = capture_name(frame)
            self.capture_poses[image_id] = self.planner.habitat2voxel(position)[:2].copy()
            depth = np.where(np.isfinite(frame.depth_m) & (frame.depth_m > 0), frame.depth_m, 0.).astype(np.float32)
            # Original ConceptGraph indexes its vertical camera coordinate as
            # H-1-v. Adjust ONLY its principal point, so GL rays exactly equal
            # optical rays after the rigid camera-axis conversion.
            legacy_intrinsics = frame.intrinsics.copy()
            legacy_intrinsics[1, 2] = frame.rgb.shape[0] - 1 - frame.intrinsics[1, 2]
            with torch.no_grad():
                _, added_ids, _ = self.scene.update_scene_graph(
                    image_rgb=frame.rgb, depth=depth,
                    intrinsics=legacy_intrinsics, cam_pos=camera, pts=position,
                    pts_voxel=self.planner.habitat2voxel(position),
                    img_path=image_id, frame_idx=self.frame_index,
                    semantic_obs=None, gt_target_obj_ids=None,
                )
            self.scene.all_observations[image_id] = frame.rgb.copy()
            self.fresh_object_ids.update(added_ids)
            self.scene.periodic_cleanup_objects(frame_idx=self.frame_index, pts=position)
            self.planner.integrate(
                color_im=frame.rgb, depth_im=depth,
                cam_intr=frame.intrinsics, cam_pose=frame.map_from_camera_optical,
                obs_weight=1., margin_h=int(self.cfg.margin_h_ratio * frame.rgb.shape[0]),
                margin_w=int(self.cfg.margin_w_ratio * frame.rgb.shape[1]),
                explored_depth=self.cfg.explored_depth,
            )
            self.ego_views.append(self.symbols["resize_image"](
                frame.rgb, self.cfg.prompt_h, self.cfg.prompt_w))
            self.frame_index += 1
            self._last_integrated = frame
        self.fresh_object_ids.intersection_update(self.scene.objects)
        self.fresh_labels = [self.scene.objects[key]["class_name"] for key in self.fresh_object_ids]
        self.map_revision += 1

    def integrate_feedback(self, frame):
        """Perception of actual post-motion evidence; retain pre-motion cycle ID."""
        self._integrate_frames((frame,))
        # Match the original agent_step's surrounding-explored update, but at
        # the measured endpoint and ONLY after physical acknowledgment.
        position, _ = self.legacy_capture(frame)
        current = self.planner.habitat2voxel(position)[:2]
        cells = np.argwhere(self.planner.unoccupied)
        nearby = cells[np.linalg.norm(cells - current, axis=1)
                       < self.cfg.planner.surrounding_explored_radius / self.planner._voxel_size]
        self.planner._explore_vol_cpu[nearby[:, 0], nearby[:, 1], :] = 1
        return list(self.fresh_labels)

    def update_memory(self):
        position, _ = self.legacy_capture(self.cycle.latest)
        targets = set(self.fresh_object_ids)
        for key, obj in self.scene.objects.items():
            if np.linalg.norm(obj["bbox"].center[[0, 2]] - position[[0, 2]]) < self.cfg.scene_graph.obj_include_dist + .5:
                targets.add(key)
        self.scene.update_snapshots(obj_ids=targets, min_detection=self.cfg.min_detection)
        for snapshot in self.scene.snapshots.values():
            names = [self.scene.objects[key]["class_name"] for key in snapshot.cluster if key in self.scene.objects]
            self.planner.update_observation_history(snapshot, self.symbols["infer_room"](names))

    def update_frontiers_and_hypotheses(self):
        position, _ = self.legacy_capture(self.cycle.latest)
        updated = self.planner.update_frontier_map(
            pts=position, cfg=self.cfg.planner, scene=self.scene, cnt_step=self.step,
            save_frontier_image=False, prompt_img_size=(self.cfg.prompt_h, self.cfg.prompt_w),
        )
        sources = self.planner.hypothesis_node_predictor.last_prediction_sources
        if any(source != "vlm" for source in sources.values()):
            raise RuntimeError(f"Full Adaptive refuses heuristic hypothesis fallback: {sources}")
        self.lifecycle.update_topology(self)
        return updated

    def _adaptive_select(self, goal):
        # Exactly the unified selection boundary used by the adaptive runner:
        # raw HGR query -> original navigator's authorization/intent policy.
        # Calling query_vlm_for_response directly would bypass Adaptive.
        position, _ = self.legacy_capture(self.cycle.latest)
        response = self.lifecycle.navigator.select(
            self.symbols["choose"], goal.metadata(), self.scene, self.planner,
            self.lifecycle.place_topology, self.ego_views, self.cfg, position,
            self.fresh_object_ids, remaining_steps=self.lifecycle.remaining_steps,
        )
        if response is None:
            return None
        choice, _ = response
        return self.selection_for_active_intent(choice)

    def selection_for_active_intent(self, choice):
        """Also used after the original navigator.continue_choice succeeds."""
        intent = self.lifecycle.navigator.active_intent
        if intent is None or intent.goal_id != self.goal.goal_id or intent.task.choice is not choice:
            raise ValueError("choice is not owned by the current Adaptive intent")
        if isinstance(choice, self.symbols["SnapShot"]):
            kind = "snapshot"
        elif isinstance(choice, self.symbols["Frontier"]):
            kind = "frontier"
        else:
            raise TypeError("original HGR returned an unknown choice type")
        return Selection(self.goal.goal_id, intent.task.source_id, kind, choice,
                         intent.intent_id, bool(intent.task.requires_verification))

    def select(self, goal):
        # continue_choice owns the bounded frontier intent; only a release or
        # absence of an intent authorizes the next adaptive selection call.
        return self.lifecycle.select_or_continue(goal, self, self._adaptive_select)

    def route(self, selection):
        return self.lifecycle.make_route(selection, self)

    def feedback(self, selection, route, feedback):
        # The lifecycle must call record_motion_progress exactly once per
        # executed macro step, then apply the active task's verification policy.
        self._require_active_selection(selection)
        return self.lifecycle.apply_feedback(selection, route, feedback, self)

    def _require_active_selection(self, selection):
        intent = self.lifecycle.navigator.active_intent
        if (intent is None or intent.intent_id != selection.intent_id
                or intent.goal_id != selection.goal_id
                or intent.task.source_id != selection.source_id
                or bool(intent.task.requires_verification) != selection.requires_verification):
            raise ValueError("stale Adaptive intent/source")
        return intent

    def verify_selected_entity(self, selection, fresh_frame):
        """Goal-instance verification, separate from Frontier/room correction.

        Lifecycle consumes the verdict via navigator.feedback and handles its
        stop/new_view/reselect result; this helper does not declare arrival.
        """
        intent = self._require_active_selection(selection)
        if (not selection.requires_verification
                or not self.lifecycle.navigator.requires_terminal_verification(intent.task)):
            raise ValueError("this Adaptive task does not authorize terminal verification")
        fresh_frame.validate()
        if (fresh_frame.map_epoch != self.cycle.latest.map_epoch
                or fresh_frame.stamp_ns <= self.cycle.latest.stamp_ns):
            raise ValueError("entity verification requires fresh post-motion evidence")
        return self.symbols["verify_entity"](
            self.goal.metadata(), fresh_frame.rgb, intent.task.choice, self.cfg,
        )

    def verify_frontier(self, selection, fresh_frame, detected_names):
        if selection.kind != "frontier":
            raise ValueError("frontier verification cannot certify an object goal")
        fresh_frame.validate()
        if fresh_frame.map_epoch != self.cycle.latest.map_epoch:
            raise ValueError("verification belongs to another map epoch")
        if fresh_frame.stamp_ns <= self.cycle.latest.stamp_ns:
            raise ValueError("verification requires post-motion evidence")
        errors_before = self.critic.stats.get("feature_residual_errors", 0)
        result = self.symbols["verify"](
            frontier=selection.original_choice, scene=self.scene, tsdf_planner=self.planner,
            semantic_critic=self.critic, actual_rgb=fresh_frame.rgb,
            actual_depth=fresh_frame.depth_m, detected_objects=detected_names,
            apply_graph_update=True,
        )
        if self.critic.stats.get("feature_residual_errors", 0) > errors_before:
            raise RuntimeError("Full Adaptive feature verification failed; degraded result not accepted")
        return result

    def cancel_goal(self, reason):
        self.lifecycle.cancel_goal(reason, self)
