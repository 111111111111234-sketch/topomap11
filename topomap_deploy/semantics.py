"""HGR memory for observed ROS maps, without importing a Habitat scene."""

from types import SimpleNamespace

import numpy as np

from src.hypothesis_graph import (
    CognitiveDependency, HypothesisGraph, HypothesisNode, NodeType,
    create_hypothesis_node_from_frontier,
)

ROOM_OBJECTS = {
    "bedroom": {"bed", "nightstand", "wardrobe"},
    "bathroom": {"toilet", "sink", "shower", "bathtub"},
    "kitchen": {"refrigerator", "oven", "stove", "microwave"},
    "living_room": {"sofa", "tv", "coffee table", "chair"},
    "dining_room": {"dining table", "chair"},
    "office": {"desk", "chair", "monitor", "bookshelf"},
}


def room_from_labels(labels):
    scores = {room: len(set(labels) & objects) for room, objects in ROOM_OBJECTS.items()}
    best = max(scores, key=scores.get)
    winners = [room for room, score in scores.items() if score == scores[best]]
    return best if scores[best] and len(winners) == 1 else None


def frontier_crop(observation, xy):
    # Only an actually captured view in the requested direction may be used as
    # frontier visual evidence. Never render/teleport/duplicate unseen viewpoints.
    point = np.r_[xy, observation.map_from_camera[2, 3], 1.]
    camera = np.linalg.inv(observation.map_from_camera) @ point
    if camera[2] <= .1:
        return None
    pixel = observation.intrinsics @ camera[:3]
    u, v = pixel[:2] / pixel[2]
    height, width = observation.rgb.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return None
    half = max(8, width // 6)
    return observation.rgb[:, max(0, int(u) - half):min(width, int(u) + half)].copy()


class HGRMemory:
    def __init__(self, cfg, predictor, critic_factory):
        self.cfg = cfg
        self.predictor = predictor
        self.graph = HypothesisGraph(cfg)
        self.graph.preserve_independent_evidence = True
        self.critic = critic_factory(cfg, self.graph)
        self.history = []
        self.visited = []
        self.counter = 0
        self.parent = None

    def observe(self, observation, labels):
        xy = observation.map_from_base[:2, 3]
        room = room_from_labels(labels)
        if room is None:
            return
        if self.visited and min(np.linalg.norm(xy - p) for p in self.visited) < 1.:
            return
        self.counter += 1
        node = HypothesisNode(f"observed_{self.counter}", NodeType.OBSERVED,
                              position=xy.copy(), observed_class=room, confidence=.7,
                              independent_observation_refs=[f"rgb:{observation.stamp_ns}"])
        self.parent = self.graph.add_node(node)
        self.visited.append(xy.copy())
        self.history.append({"room_type": room, "objects": list(labels)})

    def rank(self, candidates, observation, labels, goal):
        ranked = []
        for xy, approach, area in candidates:
            if self.visited and min(np.linalg.norm(approach - p) for p in self.visited) < .4:
                continue
            node = next((n for n in self.graph.nodes.values()
                         if n.node_type == NodeType.HYPOTHESIS and n.position is not None
                         and np.linalg.norm(n.position - xy) < .6), None)
            if node is None:
                self.counter += 1
                frontier = SimpleNamespace(frontier_id=self.counter, position=xy.copy(),
                                           orientation=xy - observation.map_from_base[:2, 3],
                                           feature=frontier_crop(observation, xy))
                predictions = self.predictor.batch_predict_frontiers(
                    [frontier], {"nearby_objects": labels, "current_room_type": room_from_labels(labels)},
                    self.history[-20:])
                node = create_hypothesis_node_from_frontier(frontier, predictions[frontier.frontier_id])
                self.graph.add_node(node)
                if self.parent in self.graph.nodes:
                    self.graph.add_dependency(CognitiveDependency(
                        self.parent, node.node_id, "observed_context_to_frontier", .7,
                        "Prediction conditioned on actually observed nearby room"))
            likely_rooms = [room for room, objects in ROOM_OBJECTS.items() if goal in objects]
            relevance = sum(node.semantic_dist.get_expected_semantic_score(room) for room in likely_rooms)
            distance = np.linalg.norm(approach - observation.map_from_base[:2, 3])
            score = relevance - .1 * distance + .05 * node.semantic_dist.entropy
            ranked.append((score, approach, node.node_id))
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def arrived(self, node_id, observation, labels):
        node = self.graph.nodes.get(node_id)
        room = room_from_labels(labels)
        # Absence of detections alone is not independent disproof of a room.
        if node is not None and node.node_type == NodeType.HYPOTHESIS and room is not None:
            self.critic.verify_hypothesis_node_arrival(
                node, {"observation_id": f"rgb:{observation.stamp_ns}",
                       "independent_support": True, "evidence_source": "post_motion_terminal",
                       "rgb_image": observation.rgb, "depth": observation.depth,
                       "detected_objects": labels, "semantic_class": room}, {})
        self.visited.append(observation.map_from_base[:2, 3].copy())
        if node_id in self.graph.observed_nodes:
            self.parent = node_id
