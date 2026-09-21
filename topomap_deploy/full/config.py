"""Configuration-only inspection: no torch, simulator, models, or API calls."""

from pathlib import Path


def load_config(path, _parents=()):
    from omegaconf import OmegaConf
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("configuration inheritance cycle")
    cfg = OmegaConf.load(path)
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = OmegaConf.merge(load_config(path.parent / parent, (*_parents, path)), cfg)
    return cfg


def validate_adaptive_policy(cfg):
    from src.dual_dynamic_navigation.config import validate_baseline_backend
    validate_baseline_backend(cfg)
    if cfg.get("navigation_backend") != "hgr_dual_topo_adaptive":
        raise ValueError("this framework targets the user-selected Adaptive-hybrid configuration")
    # Contract of the selected frozen configuration, not a new policy. Refuse
    # accidental inheritance of Route-only or all-target Verified semantics.
    expected = {
        "stage": "adaptive", "preserve_hgr_stop": False,
        "semantic_selection_mode": "original_hgr",
        "enable_candidate_annotations": False,
        "enable_revisit_deduplication": True, "enable_revisit_cost_gate": False,
        "enable_persistent_intent": True, "enable_incremental_preemption": False,
        "max_frontier_intent_motion_steps": 3,
        "enable_terminal_verification": True, "enable_selective_verification": True,
        "historical_confirmation_views": 1, "enable_unscored_frontier_fallback": False,
    }
    policy = cfg.get("dual_dynamic_navigation", {})
    for name, value in expected.items():
        if policy.get(name) != value:
            raise ValueError(f"Adaptive-hybrid requires {name}={value!r}")
    if list(policy.get("selective_verification_goal_types", [])) != ["image", "description"]:
        raise ValueError("Adaptive-hybrid verifies only selected historical image/description REVISIT tasks")


def inspect_framework(path):
    cfg = load_config(path)
    validate_adaptive_policy(cfg)
    hypothesis = cfg.get("hypothesis", {})
    for name in ("enable_hypothesis_refinement", "enable_vlm_hypothesis_prediction", "enable_cascade_deletion"):
        if not hypothesis.get(name, False):
            raise ValueError(f"full HGR requires {name}")
    if cfg.get("full_framework", {}).get("execution_enabled", False):
        raise ValueError("configuration inspection requires execution disabled; use explicit simulation opt-in separately")
    return cfg
