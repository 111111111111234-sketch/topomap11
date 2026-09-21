"""Explicit assembly of full Adaptive + external Scene + MuJoCo/MPC ports.

Importing this module never loads a model, starts a simulator or sends a VLM
request. The checked-in configuration keeps execution disabled. Local model
files and a separate opt-in are required for a later, unvalidated simulation.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .config import validate_adaptive_policy
from .contracts import IntegrationPending
from .lifecycle import OriginalAdaptiveLifecycle
from .observations import CapturedViews
from .original import OriginalStageCalls, load_original_symbols
from .sim_io import MujocoPorts, SimulationLimits, load_mpc
from .workflow import FullTopomapWorkflow


@dataclass
class OriginalModels:
    detector: object
    sam: object
    clip: object
    clip_preprocess: object
    clip_tokenizer: object


def local_file(root, value, label):
    if not value:
        raise IntegrationPending(f"Local {label} path must be supplied; downloads are not automatic")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(f"Local {label} file does not exist: {path}")
    return path.resolve()


def load_local_models(cfg, root):
    """Explicit local-only bootstrap for the original pinned model stack.

    YOLO-World uses its own OpenAI CLIP text encoder, separate from the
    ConceptGraph LAION OpenCLIP image/text model. Preload its local checkpoint
    so YOLO.set_classes cannot implicitly download ViT-B/32 weights.
    """
    options = cfg.full_framework
    paths = {name: local_file(root, options.get(name), name)
             for name in ("yolo_weights", "sam_weights", "clip_weights", "yolo_text_weights")}
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import clip as yolo_clip
    import open_clip
    from ultralytics import YOLOWorld, SAM
    device = str(options.model_device)
    detector = YOLOWorld(str(paths["yolo_weights"]))
    detector.to(device)
    if not hasattr(detector.model, "set_classes"):
        raise RuntimeError("Expected the original Ultralytics WorldModel class interface")
    detector.model.clip_model = yolo_clip.load(str(paths["yolo_text_weights"]), device=device)[0].eval()
    sam = SAM(str(paths["sam_weights"]))
    sam.to(device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(paths["clip_weights"]), device=device)
    return OriginalModels(detector, sam, model.eval(), preprocess, open_clip.get_tokenizer("ViT-B-32"))


def assemble_core(cfg, *, initial_frame, models, root, symbols=None):
    """Construct original memory and geometry from measured calibration only."""
    from omegaconf import OmegaConf
    validate_adaptive_policy(cfg)
    initial_frame.validate()
    symbols = load_original_symbols() if symbols is None else symbols
    from src.conceptgraph.utils.general_utils import ObjectClasses
    if not isinstance(models.detector, symbols["YOLOWorld"]) or not isinstance(models.sam, symbols["SAM"]):
        raise TypeError("Full perception requires original YOLOWorld and SAM models")
    if getattr(models.detector.model, "clip_model", None) is None:
        raise IntegrationPending("YOLO-World text encoder must be loaded locally before Scene.set_classes")
    options = cfg.full_framework
    graph_cfg = OmegaConf.load(local_file(root, cfg.concept_graph_config_path, "ConceptGraph config"))
    classes_path = local_file(root, options.class_names_path, "detection vocabulary")
    classes = ObjectClasses.from_classes(classes_path.read_text().splitlines(),
                                        graph_cfg.bg_classes, graph_cfg.skip_bg)
    views = CapturedViews(yaw_tolerance_deg=float(options.frontier_view_tolerance_deg))
    scene = symbols["Scene"].from_external(cfg=cfg, graph_cfg=graph_cfg,
        obj_classes=classes, intrinsics=initial_frame.intrinsics, observation_provider=views,
        detection_model=models.detector, sam_predictor=models.sam, clip_model=models.clip,
        clip_preprocess=models.clip_preprocess, clip_tokenizer=models.clip_tokenizer,
        device=options.model_device)
    position, _ = OriginalStageCalls.legacy_capture(initial_frame)
    scene.start_position = position.copy()
    bounds = np.asarray(options.map_bounds_xyz_m, dtype=float)
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or not (bounds[:, 1] > bounds[:, 0]).all()
            or not np.isclose(bounds[2, 0], options.floor_height_m)):
        raise ValueError("explicit finite TSDF bounds must start at the configured floor height")
    planner = symbols["TSDFPlanner"](vol_bnds=bounds.copy(), voxel_size=cfg.tsdf_grid_size,
        floor_height=float(options.floor_height_m), pts_init=position,
        init_clearance=cfg.init_clearance * 2, save_visualization=False,
        hypothesis_graph_cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True))
    critic = symbols["SemanticCritic"](cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True),
                                       hypothesis_graph=planner.hypothesis_graph)
    critic.set_clip_model(models.clip, models.clip_preprocess)
    policy = OmegaConf.to_container(cfg.dual_dynamic_navigation, resolve=True)
    topology_cfg = OmegaConf.to_container(cfg.active_topology, resolve=True)
    lifecycle = OriginalAdaptiveLifecycle(
        navigator=symbols["HypothesisAwareNavigator"](policy),
        place_topology=symbols["PlaceTopology"](
            place_spacing_m=cfg.active_topology.place_topology.place_spacing_m,
            voxel_size=cfg.tsdf_grid_size, stage="route_only"),
        persistent_topology=symbols["PersistentBeliefTopology"](topology_cfg, voxel_size=cfg.tsdf_grid_size),
        max_motion_steps=int(options.max_motion_steps))
    symbols["configure_vlm_runtime"](cfg)
    return OriginalStageCalls(scene=scene, planner=planner, critic=critic, cfg=cfg,
                              lifecycle=lifecycle, symbols=symbols)


@dataclass
class SimulationSession:
    workflow: FullTopomapWorkflow
    core: OriginalStageCalls
    ports: MujocoPorts

    def close(self):
        try:
            self.core.cancel_goal("session_closed")
        finally:
            self.ports.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def start_simulation(cfg, *, root, lightnav_dir, events, allow_unvalidated=False, models=None):
    """Future explicit simulation entry, NEVER a real robot launcher.

    Both guards must be enabled by the caller. No autorun on import/config
    inspection. This function was implemented but not executed in this task.
    """
    from omegaconf import OmegaConf
    validate_adaptive_policy(cfg)
    options = cfg.full_framework
    if not options.execution_enabled or not allow_unvalidated:
        raise IntegrationPending("Simulation execution disabled; explicit unvalidated-simulation opt-in required")
    key_name = str(cfg.vlm_api_key_env)
    if not os.environ.get(key_name):
        raise IntegrationPending(f"Set {key_name} locally before requesting full VLM execution")
    limits = SimulationLimits(**OmegaConf.to_container(options.simulation, resolve=True))
    if float(options.execution_timeout_s) <= limits.route_timeout_s:
        raise ValueError("workflow timeout must leave time for route lease expiry and physical stopping")
    xml_path = (None if not options.get("scene_xml_path")
                else local_file(root, options.scene_xml_path, "MuJoCo scene"))
    if models is None:
        models = load_local_models(cfg, root)
    from ..sim_world import World
    core_holder = {}
    ports = MujocoPorts(world_factory=lambda: World(xml_path=xml_path),
        controller_factory=lambda: load_mpc(lightnav_dir, limits),
        geometry_provider=lambda: None if not core_holder else core_holder["core"].lifecycle.geometry,
        limits=limits)
    try:
        ports.start()
        core = assemble_core(cfg, initial_frame=ports.latest, models=models, root=root)
        core_holder["core"] = core
        workflow = FullTopomapWorkflow(core, ports, ports, events,
            execution_timeout_s=float(options.execution_timeout_s),
            terminal_tolerance_m=float(options.terminal_tolerance_m))
        events({"event": "assembled", "status": "ADAPTERS_IMPLEMENTED_NOT_RUNTIME_VALIDATED",
                "algorithm": cfg.navigation_backend, "localization": "ideal_simulator_pose",
                "body": "planar_proxy", "semantic_scene": "external_xml" if xml_path else "primitive_fixture"})
        return SimulationSession(workflow, core, ports)
    except BaseException:
        ports.close()
        raise
