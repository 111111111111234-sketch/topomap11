"""Single-robot HTTP inference service; heavy model imports stay off the robot."""

import argparse
import hmac
import json
import logging
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .protocol import MAX_MESSAGE_BYTES, Observation
from .runtime import Navigator, YoloDetector


def validate_config(cfg):
    for name in ("robot_radius_m", "arrival_distance_m", "approach_distance_m", "path_horizon_m"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 10:
            raise ValueError(f"invalid {name}")
    if not 0 < cfg["target_confidence"] <= 1:
        raise ValueError("invalid target_confidence")
    if not isinstance(cfg["confirmation_frames"], int) or cfg["confirmation_frames"] < 2:
        raise ValueError("confirmation_frames must be at least two")
    if cfg["arrival_distance_m"] <= cfg["approach_distance_m"]:
        raise ValueError("arrival distance must exceed object approach distance")
    if not cfg["classes"] or len(set(cfg["classes"])) != len(cfg["classes"]):
        raise ValueError("classes must be unique and nonempty")
    return cfg


class Service:
    def __init__(self, detector, navigator_factory, classes):
        self.detector, self.factory, self.classes = detector, navigator_factory, classes
        self.lock = threading.Lock()
        self.session = None
        self.navigator = None

    def dispatch(self, path, data):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("planner busy; only one request may be in flight")
        try:
            if path == "/reset":
                goal = str(data.get("goal", "")).strip().lower()
                if goal not in self.classes:
                    raise ValueError("goal must be one configured object category, not an arbitrary instruction")
                navigator = self.factory(self.detector, goal)
                self.navigator, self.session = navigator, uuid.uuid4().hex
                return {"session": self.session, "goal": goal}
            if path not in ("/step", "/stop"):
                raise ValueError("unknown endpoint")
            if self.session is None or data.get("session") != self.session:
                raise ValueError("invalid or expired session; reset required")
            if path == "/stop":
                self.navigator, self.session = None, None
                return {"stopped": True}
            return self.navigator.step(Observation.decode(data["observation"]))
        finally:
            self.lock.release()


def make_handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.reply(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/health":
                self.reply(404, {"error": "not found"})
                return
            self.reply(200, {"ready": True, "protocol": 1, "classes": service.classes,
                             "backend": "hgr_ros_grid", "robots_per_process": 1})

        def do_POST(self):
            if not self.authorized():
                return
            self.connection.settimeout(15.)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_MESSAGE_BYTES:
                    self.reply(413, {"error": "invalid request size"})
                    return
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("incomplete request")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("request must be a JSON object")
                self.reply(200, service.dispatch(self.path, data))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.reply(409, {"error": str(exc)})
            except Exception:
                logging.exception("inference failed")
                self.reply(500, {"error": "inference failed; inspect server log"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--check-config", action="store_true", help="validate config without loading models")
    args = parser.parse_args()
    cfg = validate_config(json.loads(Path(args.config).read_text()))
    if args.check_config:
        print("Configuration valid. This does not verify sensors, models, ROS, or hardware.")
        return
    token = os.environ.get("TOPOMAP_TOKEN", "")
    if args.host not in ("localhost", "127.0.0.1", "::1") and not token:
        parser.error("non-loopback binding requires TOPOMAP_TOKEN; prefer an SSH tunnel")
    if cfg["hypothesis"].get("enable_vlm_hypothesis_prediction") and not os.environ.get(cfg["vlm_api_key_env"]):
        parser.error(f"{cfg['vlm_api_key_env']} is not set")
    from omegaconf import OmegaConf
    from src.hypothesis_node_predictor import HypothesisNodePredictor
    from src.semantic_critic import SemanticCritic
    from src.vlm_runtime import configure_vlm_runtime
    from .semantics import HGRMemory
    logging.basicConfig(level=logging.INFO)
    configure_vlm_runtime(OmegaConf.create(cfg))
    detector = YoloDetector(cfg["yolo_weights"], cfg["classes"], cfg["device"])

    def factory(detector, goal):
        memory = HGRMemory(cfg["hypothesis"], HypothesisNodePredictor(cfg["hypothesis"]), SemanticCritic)
        return Navigator(detector, memory, goal, cfg)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(Service(detector, factory, cfg["classes"]), token))
    logging.info("Models loaded. Listening on %s:%s; no robot control in this process", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
