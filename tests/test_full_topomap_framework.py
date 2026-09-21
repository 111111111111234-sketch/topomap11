"""Framework contract tests only. Written but NOT executed in the framework task.

These fakes test orchestration, never the original models or navigation quality.
"""

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from topomap_deploy.full.config import inspect_framework, load_config, validate_adaptive_policy
from topomap_deploy.full.contracts import (
    FeedbackDecision, Frame, Goal, MotionFeedback, ObservationCycle,
    RouteRequest, Selection, State,
)
from topomap_deploy.full.workflow import FullTopomapWorkflow


def frame(sequence, x=0., epoch="map-1"):
    pose = np.eye(4)
    pose[0, 3] = x
    return Frame(sequence, sequence * 1_000_000_000, epoch,
                 np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)),
                 np.array([[10., 0, 4], [0, 10., 4], [0, 0, 1.]]), np.eye(4), pose)


class Core:
    def __init__(self, final=True, complete=False):
        self.calls = []
        self.final, self.complete = final, complete

    def begin_goal(self, goal): self.calls.append("begin")
    def integrate(self, cycle): self.calls.append("integrate")
    def update_memory(self): self.calls.append("memory")
    def update_frontiers_and_hypotheses(self): self.calls.append("hypotheses")
    def select(self, goal):
        self.calls.append("select")
        return Selection(goal.goal_id, "snapshot:a:objects:1", "snapshot", object(), "g1:intent:1")
    def route(self, choice):
        self.calls.append("route")
        return RouteRequest(choice.goal_id, "r1", choice.source_id, "map-1", 1, 1,
                            np.array([[0., 0.], [1., 0.]]), np.array([1., 0.]),
                            self.final, {"geometry_sha256": "test-only", "fallback": False},
                            intent_id=choice.intent_id)
    def feedback(self, selection, route, feedback):
        self.calls.append("feedback")
        return FeedbackDecision(self.complete, "original_policy_feedback")
    def cancel_goal(self, reason): self.calls.append("cancel")


class Sensor:
    def collect(self, *, first_cycle): return ObservationCycle((frame(1),))


class Motion:
    def __init__(self): self.stops, self.submitted, self.result = [], [], None
    def stop(self, reason): self.stops.append(reason)
    def submit(self, request): self.submitted.append(request)
    def poll(self): return self.result


def workflow(**kwargs):
    core, sensor, motion, events = Core(**kwargs), Sensor(), Motion(), []
    engine = FullTopomapWorkflow(core, sensor, motion, events.append)
    engine.begin(Goal("g1", "object", "Find the chair", "chair"))
    return engine, core, motion, events


def test_full_stage_order_and_actual_feedback_gate():
    engine, core, motion, _ = workflow(complete=True)
    assert engine.advance() == State.EXECUTING
    assert core.calls == ["begin", "integrate", "memory", "hypotheses", "select", "route"]
    assert engine.advance() == State.EXECUTING  # no fabricated movement result
    motion.result = MotionFeedback("g1", "r1", "reached", frame(2, x=1.), True)
    assert engine.advance() == State.ARRIVED
    assert core.calls[-1] == "feedback"


def test_intermediate_place_cannot_be_reported_as_goal_completion():
    engine, _, motion, _ = workflow(final=False, complete=True)
    engine.advance()
    motion.result = MotionFeedback("g1", "r1", "reached", frame(2, x=1.), True)
    with pytest.raises(ValueError, match="intermediate"):
        engine.advance()
    assert engine.state == State.ERROR and motion.stops[-1] == "pipeline_error"


@pytest.mark.parametrize("result", [
    MotionFeedback("old-goal", "r1", "reached", frame(2, x=1.), True),
    MotionFeedback("g1", "old-route", "reached", frame(2, x=1.), True),
    MotionFeedback("g1", "r1", "reached", frame(1, x=1.), True),
    MotionFeedback("g1", "r1", "reached", frame(2, x=0.), True),
    MotionFeedback("g1", "r1", "reached", frame(2, x=1., epoch="map-2"), True),
    MotionFeedback("g1", "r1", "reached", frame(2, x=1.), False),
])
def test_invalid_feedback_stops_and_cannot_complete(result):
    engine, _, motion, _ = workflow(complete=True)
    engine.advance()
    motion.result = result
    with pytest.raises(ValueError): engine.advance()
    assert engine.state == State.ERROR


def test_timeout_does_not_depend_on_model_or_sensor_callbacks():
    engine, _, motion, _ = workflow()
    clock = [0.]
    engine.clock = lambda: clock[0]
    engine.advance()
    clock[0] = 20.
    assert engine.advance() == State.BLOCKED
    assert motion.stops[-1] == "execution_timeout"


def test_cancel_invalidates_pending_motion():
    engine, _, motion, _ = workflow()
    engine.advance()
    engine.cancel()
    assert engine.pending_route is None
    assert engine.advance() == State.CANCELLED
    assert motion.stops[-1] == "operator_cancel"


def test_observation_cycle_refuses_epoch_mix_and_replay():
    with pytest.raises(ValueError): ObservationCycle((frame(2), frame(1))).validate()
    with pytest.raises(ValueError): ObservationCycle((frame(1), frame(2, epoch="map-2"))).validate()


def test_framework_keeps_original_adaptive_and_vlm_config():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1]
    cfg = inspect_framework(root / "deploy/full/adaptive_mac.yaml")
    original = load_config(root / "cfg/eval_goatbench_hgr_dual_topo_adaptive_hybrid.yaml")
    assert cfg.navigation_backend == "hgr_dual_topo_adaptive"
    assert OmegaConf.to_container(cfg.dual_dynamic_navigation) == OmegaConf.to_container(original.dual_dynamic_navigation)
    assert cfg.hypothesis.enable_vlm_hypothesis_prediction
    assert cfg.hypothesis.residual_weight_feature > 0
    assert cfg.full_framework.execution_enabled is False


@pytest.mark.parametrize("key,value", [
    ("max_frontier_intent_motion_steps", 0),
    ("historical_confirmation_views", 2),
    ("enable_selective_verification", False),
    ("enable_revisit_deduplication", False),
    ("enable_revisit_cost_gate", True),
    ("enable_incremental_preemption", True),
    ("enable_unscored_frontier_fallback", True),
    ("semantic_selection_mode", "executable_subset"),
])
def test_accidental_changes_to_selected_adaptive_policy_are_rejected(key, value):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "deploy/full/adaptive_mac.yaml")
    cfg.dual_dynamic_navigation[key] = value
    with pytest.raises(ValueError): validate_adaptive_policy(cfg)


def test_route_with_another_intent_is_not_executed():
    engine, core, motion, _ = workflow()
    original_route = core.route
    core.route = lambda choice: replace(original_route(choice), intent_id="old-intent")
    with pytest.raises(ValueError, match="provenance"): engine.advance()
    assert not motion.submitted


@pytest.mark.parametrize("confirmed_intent", [None, "old-intent", "g1:intent:1"])
def test_selective_verification_requires_confirmation_of_same_intent(confirmed_intent):
    engine, core, motion, _ = workflow(complete=True)
    original_select = core.select
    core.select = lambda goal: replace(original_select(goal), requires_verification=True)
    core.feedback = lambda *args: FeedbackDecision(True, "confirmed", verified_intent_id=confirmed_intent)
    engine.advance()
    motion.result = MotionFeedback("g1", "r1", "reached", frame(2, x=1.), True)
    if confirmed_intent == "g1:intent:1":
        assert engine.advance() == State.ARRIVED
    else:
        with pytest.raises(ValueError, match="same intent"): engine.advance()


def test_config_inheritance_cycle_is_an_error(tmp_path):
    # Runtime test fixture creation; no config is rewritten by the actual CLI.
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"): load_config(tmp_path / "a.yaml")
