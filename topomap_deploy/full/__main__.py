"""Inspect the full-topomap framework; intentionally has no run/robot mode."""

import argparse
import json

from .config import inspect_framework
from .original import ORIGINAL_SYMBOLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/full/adaptive_mac.yaml")
    parser.add_argument("--describe", action="store_true", required=True)
    args = parser.parse_args()
    cfg = inspect_framework(args.config)
    print(json.dumps({
        "status": "ADAPTERS_IMPLEMENTED_NOT_RUNTIME_VALIDATED",
        "algorithm": cfg.navigation_backend,
        "vlm_model": cfg.vlm_model,
        "policy_owner": "src.dual_dynamic_navigation.unified_navigator.HypothesisAwareNavigator",
        "original_bindings": {key: f"{module}.{symbol}" for key, (module, symbol) in ORIGINAL_SYMBOLS.items()},
        "implemented_adapters": {
            name: cfg.full_framework[name] for name in
            ("scene_adapter", "sensor_adapter", "motion_adapter", "adaptive_lifecycle")
        },
        "pending_runtime_validation": [
            "local YOLO/SAM/LAION-CLIP and YOLO text-CLIP weights + native dependencies",
            "camera calibration / model import checks / source-equivalence replay",
            "full Adaptive closed loop and watchdog/fault tests",
            "semantic scene (default primitive fixture is not a YOLO benchmark)",
        ],
        "model_calls": 0,
        "motion_enabled": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
