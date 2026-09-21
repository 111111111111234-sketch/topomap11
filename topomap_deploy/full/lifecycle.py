"""Measured-I/O lifecycle for the ORIGINAL Adaptive navigator.

Semantic decisions, persistent intent windows, aliases, recovery and verification
are owned by src.dual_dynamic_navigation, not by this adapter. TSDF.agent_step
is deliberately not called: it marks *proposed* locations explored and clears
targets before a physical robot could acknowledge them. Its original
KnownSpaceExecutor/RouteGuidance geometry is used to propose one macro segment.
"""

import numpy as np

from .contracts import FeedbackDecision, RouteRequest
from .safety import GeometrySnapshot


class OriginalAdaptiveLifecycle:
    def __init__(self, *, navigator, place_topology, persistent_topology, max_motion_steps=100):
        if max_motion_steps <= 0:
            raise ValueError("positive motion budget required")
        self.navigator, self.place_topology = navigator, place_topology
        self.persistent_topology = persistent_topology
        self.max_motion_steps = max_motion_steps
        self.remaining_steps = max_motion_steps
        self.route_sequence = 0
        self._feedback_ids = set()
        self._capture_places = {}
        self._embeddings = {}
        self.geometry = None

    @staticmethod
    def _clear_target(stages):
        stages.planner.max_point = stages.planner.target_point = stages.planner.look_at_point = None

    def begin_goal(self, goal, stages):
        self.navigator.release("new_goal", stages.planner.hypothesis_graph)
        self.navigator.begin_goal(goal.goal_id)
        self.remaining_steps = self.max_motion_steps
        self._feedback_ids.clear()
        self._clear_target(stages)
        stages.planner.hypothesis_graph.preserve_independent_evidence = True

    def _embedding(self, key, image, stages):
        if key not in self._embeddings:
            import torch
            from PIL import Image
            model = stages.scene.clip_model
            device = next(model.parameters()).device
            tensor = stages.scene.clip_preprocess(Image.fromarray(np.asarray(image)).convert("RGB"))
            with torch.no_grad():
                vector = model.encode_image(tensor.unsqueeze(0).to(device))
                vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self._embeddings[key] = vector.detach().cpu().numpy().reshape(-1)
        return self._embeddings[key]

    def update_topology(self, stages):
        from src.dual_dynamic_navigation.executor import known_mask
        graph, persistent, planner = self.place_topology, self.persistent_topology, stages.planner
        frame = stages._last_integrated
        position, _ = stages.legacy_capture(frame)
        current = planner.habitat2voxel(position)[:2]
        mask = known_mask(planner)
        persistent.decay_confidence(stages.step)
        visited = persistent.upsert_visited(current, stages.step, evidence_ref=f"pose:{frame.sequence}")
        # Bind each capture at its capture-time pose, never at a later goal's pose.
        for image, point in stages.capture_poses.items():
            if image in self._capture_places:
                continue
            paths = graph.known_grid_path_lengths_m(mask, point, graph.local_connection_targets(point), planner._voxel_size)
            assigned = graph.observe_pose(point, stages.step, observation_id=image, verified_paths_m=paths)
            graph.add_known_free_connections(paths, stages.step, image)
            self._capture_places[image] = assigned.place_id
        paths = graph.known_grid_path_lengths_m(mask, current, graph.local_connection_targets(current), planner._voxel_size)
        assigned = graph.observe_pose(current, stages.step, observation_id=f"pose:{frame.sequence}", verified_paths_m=paths)
        graph.add_known_free_connections(paths, stages.step, f"pose:{frame.sequence}")
        for image, snapshot in stages.scene.snapshots.items():
            if image not in self._capture_places:
                raise ValueError("Snapshot has no measured capture provenance")
            labels = [stages.scene.objects[key]["class_name"] for key in snapshot.cluster if key in stages.scene.objects]
            persistent.upsert_observed(snapshot, snapshot.obs_point[:2], labels, stages.step, visited,
                evidence_ref=f"snapshot:{image}",
                visual_embedding=self._embedding(f"snapshot:{image}", stages.scene.all_observations[image], stages))
            if f"snapshot:{image}" not in graph.observations:
                graph.bind_observation(f"snapshot:{image}", self._capture_places[image],
                                       stages.capture_poses[image], stages.step)
        embeddings = {i: self._embedding(f"frontier:{frontier.image}", frontier.feature, stages)
                      for i, frontier in enumerate(planner.frontiers)}
        persistent.match_frontiers(planner.frontiers, stages.step, visual_embeddings=embeddings)
        targets = {persistent.frontier_source_map[i]: f.position for i, f in enumerate(planner.frontiers)}
        distances = graph.known_grid_path_lengths_m(mask, current, targets, planner._voxel_size)
        for i, frontier in enumerate(planner.frontiers):
            source = persistent.frontier_source_map[i]
            distance = distances.get(source)
            graph.observe_frontier(source, frontier.position, assigned.place_id, stages.step,
                                   source_frontier_index=i, verified_approach_path_m=(
                                       None if distance is None else distance + graph.current_place_offset_m))
        self.navigator.sync(graph, stages.scene, planner.frontiers,
                            f"{stages.goal.goal_id}:{stages.step}", min_detection=stages.cfg.min_detection)
        self.geometry = GeometrySnapshot.from_planner(planner, frame, stages.map_revision)

    def select_or_continue(self, goal, stages, adaptive_select):
        if self.remaining_steps <= 0:
            self.cancel_goal("motion_budget_exhausted", stages)
            return None
        position, _ = stages.legacy_capture(stages.cycle.latest)
        choice = self.navigator.continue_choice(stages.scene, stages.planner,
                                               self.place_topology, position, stages.cfg)
        if choice is not None:
            return stages.selection_for_active_intent(choice)
        self._clear_target(stages)
        return adaptive_select(goal)

    def make_route(self, selection, stages):
        from src.dual_dynamic_navigation.executor import known_mask
        from src.route_guidance import visible, path_mask
        intent = stages._require_active_selection(selection)
        planner, cfg, frame = stages.planner, stages.cfg, stages.cycle.latest
        position, _ = stages.legacy_capture(frame)
        self._clear_target(stages)
        executor = self.navigator.executor
        ready = self.navigator.setup(planner, position, cfg) and executor.prepare(planner, position)
        if not ready:
            # Preserve the original runner's geometry-recovery branch even
            # when a fresh map blocks the route before motion submission.
            recovered = self.navigator.recover(planner, self.place_topology, position,
                                                cfg.planner.final_observe_distance)
            self._clear_target(stages)
            if not (recovered and self.navigator.setup(planner, position, cfg)
                    and executor.prepare(planner, position)):
                return None
        measured = (frame.map_from_base[:2, 3] - planner._vol_origin[:2]) / planner._voxel_size
        guide = executor.guidance
        if not np.allclose(measured, guide.path[0]):
            if not visible(known_mask(planner), measured, guide.path[0]):
                return None
            guide.path = np.vstack([measured, guide.path])
            guide.corridor = path_mask(known_mask(planner).shape, guide.path)
        limit = (cfg.planner.max_dist_from_cur_phase_1 if selection.kind == "frontier"
                 else cfg.planner.max_dist_from_cur_phase_2)
        proposed = guide.step(measured, limit, planner._voxel_size, known_mask(planner))
        if proposed is None:
            return None
        terminal, final = proposed
        # Include actual continuous XY, not only the rounded voxel center.
        if not visible(known_mask(planner), measured, terminal):
            return None
        path = np.asarray([measured, terminal]) * planner._voxel_size + planner._vol_origin[:2]
        look_at = np.asarray(intent.task.route["look_at"]) if final else terminal
        direction = look_at - (terminal if final else measured)
        yaw = (float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction) > 1e-8
               else float(np.arctan2(frame.map_from_base[1, 0], frame.map_from_base[0, 0])))
        self.route_sequence += 1
        self.geometry = GeometrySnapshot.from_planner(planner, frame, stages.map_revision)
        return RouteRequest(selection.goal_id, f"{selection.intent_id}:motion:{self.route_sequence}",
            selection.source_id, frame.map_epoch, frame.sequence, stages.map_revision,
            path, path[-1].copy(), bool(final),
            {**executor.last_audit, "geometry_sha256": self.geometry.digest,
             "fallback": False, "macro_step": self.route_sequence,
             "proposed_segment_map_xy": path.tolist()}, selection.intent_id, yaw)

    def _recover(self, stages, position, reason):
        recovered = self.navigator.recover(stages.planner, self.place_topology, position,
                                           stages.cfg.planner.final_observe_distance)
        if not recovered:
            self.cancel_goal(reason, stages)
        return FeedbackDecision(False, "route_recovered" if recovered else reason)

    def apply_feedback(self, selection, route, feedback, stages):
        if route.route_id in self._feedback_ids:
            raise ValueError("motion feedback already applied")
        intent = stages._require_active_selection(selection)
        self._feedback_ids.add(route.route_id)
        self.remaining_steps -= 1
        position, _ = stages.legacy_capture(feedback.frame)
        moved = np.linalg.norm(feedback.frame.map_from_base[:2, 3] - route.path_map_xy[0]) > 1e-4
        arrived = feedback.status == "reached" and route.is_final_terminal
        stalled = self.navigator.record_motion_progress(stages.planner, self.place_topology,
                                                       position, moved=bool(moved), arrived=arrived)
        if feedback.status == "cancelled":
            self.cancel_goal("motion_cancelled", stages)
            return FeedbackDecision(False, "motion_cancelled")
        if stalled or feedback.status != "reached":
            return self._recover(stages, position, "motion_recovery_exhausted")
        # Only real feedback is fused. A proposed endpoint never updates TSDF.
        labels = stages.integrate_feedback(feedback.frame)
        if not arrived:
            return FeedbackDecision(False, "macro_motion_acknowledged")
        if selection.kind == "frontier":
            if getattr(selection.original_choice, "hypothesis_node_id", None) is not None:
                result = stages.verify_frontier(selection, feedback.frame, labels)
                if result.get("is_falsified", False) and stages.cfg.hypothesis.enable_cascade_deletion:
                    stages.planner.hypothesis_graph.prune_falsified_nodes()
            self.cancel_goal("frontier_observed", stages)
            return FeedbackDecision(False, "frontier_observed")
        if selection.requires_verification:
            verdict, detail = stages.verify_selected_entity(selection, feedback.frame)
            action = self.navigator.feedback(verdict, intent.intent_id,
                f"{route.route_id}:verify:{feedback.frame.sequence}", stages.planner)
            if action == "stop":
                return FeedbackDecision(True, "adaptive_entity_confirmed", detail, intent.intent_id)
            if action == "new_view":
                rebound = self.navigator.continue_choice(stages.scene, stages.planner,
                    self.place_topology, position, stages.cfg)
                if rebound is None:
                    self.cancel_goal("verification_source_invalid", stages)
                    return FeedbackDecision(False, "verification_source_invalid", detail)
                next_route = self.navigator.resolver.object_view(self.place_topology, stages.planner,
                    stages.scene.objects, intent.task.choice.cluster[0],
                    stages.planner.habitat2voxel(position)[:2], stages.cfg.planner.final_observe_distance,
                    self.navigator.checked_views.get(intent.task.entity_id, ()))
                if next_route.get("status") == "valid":
                    intent.task.route = next_route
                    self._clear_target(stages)
                    if self.navigator.setup(stages.planner, position, stages.cfg):
                        return FeedbackDecision(False, "adaptive_new_view_required", detail)
                self.cancel_goal("no_unchecked_view", stages)
            self._clear_target(stages)
            return FeedbackDecision(False, "adaptive_reselect", detail)
        if intent.task.kind.value == "REVISIT":
            if self.navigator.promote_revisit(intent.intent_id) is None:
                raise RuntimeError("arrival lost its authoritative revisit")
        return FeedbackDecision(True, "original_snapshot_arrival")

    def cancel_goal(self, reason, stages):
        self.navigator.release(reason, stages.planner.hypothesis_graph)
        self._clear_target(stages)
