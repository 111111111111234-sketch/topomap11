"""Synchronize real RGB-D/TF/map observations; gate every autonomous command."""

import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener

from topomap_deploy.control import bounded_command, depth_obstacle, motion_allowed, validate_reply
from topomap_deploy.planning import traversable
from topomap_deploy.protocol import Grid, Observation, MAX_MESSAGE_BYTES
from src.route_guidance import visible


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def quaternion_matrix(q):
    values = np.array([q.x, q.y, q.z, q.w], float)
    if not np.isfinite(values).all() or abs(np.linalg.norm(values) - 1.) > .01:
        raise ValueError("invalid orientation quaternion")
    x, y, z, w = values / np.linalg.norm(values)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


class Bridge(Node):
    def __init__(self):
        super().__init__("topomap_bridge")
        defaults = {
            "server_url": "http://127.0.0.1:8060", "allow_motion": False,
            "rgb_topic": "/camera/color/image_rect", "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/color/camera_info", "map_topic": "/rtabmap/map",
            "odom_topic": "/odom", "sync_slop_s": .04, "sensor_timeout_s": .6,
            "tf_wait_s": .08,
            "map_timeout_s": 3., "plan_timeout_s": 1.5, "max_result_age_s": 2.,
            "request_timeout_s": 60., "robot_radius_m": .4,
            "linear_limit": .2, "angular_limit": .4,
        }
        self.params = {name: self.declare_parameter(name, value).value for name, value in defaults.items()}
        for key in defaults:
            if isinstance(defaults[key], float) and (not math.isfinite(self.params[key]) or self.params[key] <= 0):
                raise ValueError(f"{key} must be finite and positive")
        if self.params["linear_limit"] > .3 or self.params["angular_limit"] > .6:
            raise ValueError("initial Go2 deployment limits are at most 0.3 m/s and 0.6 rad/s")
        self.cv = CvBridge()
        self.tf = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.tf, self)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.goal = ""
        self.episode = 1
        self.sequence = 0
        self.session = None
        self.enabled = False
        self.running = False
        self.obstacle = True
        self.info = None
        self.grid = None
        self.latest = None
        self.image_queue = deque(maxlen=10)
        self.sensor_at = self.odom_at = self.map_at = self.plan_at = -math.inf
        self.path_world = None
        self.map_from_odom = None
        self.last_status = ""
        self.last_error = ""
        self.request_at = -math.inf
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "vln/status", latched)
        self.detail_pub = self.create_publisher(String, "topomap/status", latched)
        self.response_pub = self.create_publisher(String, "vln/response", 10)
        self.shadow_pub = self.create_publisher(String, "topomap/shadow_response", 10)
        self.enable_pub = self.create_publisher(Bool, "mpc/enable", 1)
        self.mode_pub = self.create_publisher(String, "vln/mode", latched)
        self.command_pub = self.create_publisher(TwistStamped, "topomap/safe_cmd_vel", 1)
        self.create_subscription(CameraInfo, self.params["camera_info_topic"], self.on_info, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, self.params["map_topic"], self.on_map, latched)
        self.create_subscription(Odometry, self.params["odom_topic"], self.on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "topomap/goal", self.on_goal, 10)
        self.create_subscription(TwistStamped, "topomap/raw_cmd_vel", self.on_command, 1)
        self.create_service(SetBool, "topomap/enable_motion", self.on_enable)
        self.rgb_sub = Subscriber(self, Image, self.params["rgb_topic"], qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.params["depth_topic"], qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, self.params["sync_slop_s"])
        self.sync.registerCallback(self.on_images)
        self.create_timer(.05, self.tick)
        self.mode_pub.publish(String(data="objnav"))
        self.status("IDLE", "waiting for a category on /topomap/goal; motion starts disabled")

    def status(self, state, reason):
        self.running = state == "RUNNING"
        self.status_pub.publish(String(data=state if state in ("IDLE", "RUNNING") else "ERROR"))
        detail = json.dumps({"state": state, "reason": reason, "armed": self.enabled,
                             "allow_motion": self.params["allow_motion"], "episode": self.episode})
        if detail != self.last_status:
            self.detail_pub.publish(String(data=detail))
            self.get_logger().info(detail)
            self.last_status = detail

    def stop(self):
        self.running = False
        self.path_world = None
        self.brake()

    def brake(self):
        self.enable_pub.publish(Bool(data=False))
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_link"
        self.command_pub.publish(command)

    def on_info(self, msg):
        self.info = msg

    def on_map(self, msg):
        try:
            if msg.header.frame_id != "map":
                raise ValueError("occupancy grid must use frame_id=map")
            age = (self.get_clock().now().nanoseconds - stamp_ns(msg.header.stamp)) / 1e9
            if not 0 <= age <= self.params["map_timeout_s"]:
                raise ValueError("stale/future map timestamp")
            rotation = quaternion_matrix(msg.info.origin.orientation)
            if not np.allclose(rotation[2], [0, 0, 1], atol=.001):
                raise ValueError("only planar occupancy maps are supported")
            origin = [msg.info.origin.position.x, msg.info.origin.position.y,
                      math.atan2(rotation[1, 0], rotation[0, 0])]
            self.grid = Grid(np.asarray(msg.data).reshape(msg.info.height, msg.info.width),
                             msg.info.resolution, origin)
            self.map_at = time.monotonic()
            # An obstacle update invalidates a previously issued route immediately.
            if self.path_world is not None and not self.path_clear():
                self.stop()
                self.status("BLOCKED", "map update invalidated the current route")
        except (ValueError, TypeError) as exc:
            self.grid = None
            self.stop()
            self.status("ERROR", str(exc))

    def on_odom(self, msg):
        now = self.get_clock().now().nanoseconds
        if (msg.header.frame_id != "odom" or msg.child_frame_id != "base_link"
                or not 0 <= (now - stamp_ns(msg.header.stamp)) / 1e9 <= self.params["sensor_timeout_s"]):
            self.odom_at = -math.inf
            self.stop()
            return
        self.odom_at = time.monotonic()

    def transform(self, target, source, stamp):
        transform = self.tf.lookup_transform(target, source, stamp, timeout=Duration(seconds=0.)).transform
        matrix = np.eye(4)
        matrix[:3, :3] = quaternion_matrix(transform.rotation)
        matrix[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite transform")
        return matrix

    def on_images(self, rgb_msg, depth_msg):
        # Let capture-time odometry/TF arrive before resolving transforms. Do
        # not wait inside a subscription callback: that would starve TF itself.
        self.image_queue.append((time.monotonic(), rgb_msg, depth_msg))

    def process_images(self, rgb_msg, depth_msg):
        try:
            if self.info is None or self.grid is None:
                raise ValueError("waiting for CameraInfo and occupancy map")
            if (not rgb_msg.header.frame_id or rgb_msg.header.frame_id != depth_msg.header.frame_id
                    or self.info.header.frame_id != rgb_msg.header.frame_id):
                raise ValueError("RGB, registered depth and CameraInfo must share the RGB optical frame")
            if (self.info.width, self.info.height) != (rgb_msg.width, rgb_msg.height):
                raise ValueError("CameraInfo resolution differs from RGB")
            now_ns = self.get_clock().now().nanoseconds
            stamp = stamp_ns(rgb_msg.header.stamp)
            if not 0 <= (now_ns - stamp) / 1e9 <= self.params["sensor_timeout_s"]:
                raise ValueError("stale/future image timestamp; check clocks")
            rgb = np.asarray(self.cv.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8"))
            raw_depth = np.asarray(self.cv.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
            if depth_msg.encoding == "16UC1":
                depth = raw_depth.astype(np.float32) * .001
            elif depth_msg.encoding == "32FC1":
                depth = raw_depth.astype(np.float32)
            else:
                raise ValueError("depth must be 16UC1 millimetres or 32FC1 metres")
            if depth.shape != rgb.shape[:2] or depth.size > 1280 * 720:
                raise ValueError("use aligned RGB-D at matching resolution, at most 1280x720")
            if np.count_nonzero(np.isfinite(depth) & (depth > .15) & (depth < 8.)) < depth.size * .05:
                raise ValueError("insufficient valid depth")
            k = np.asarray(self.info.p).reshape(3, 4)[:, :3]
            if k[0, 0] <= 0:
                if any(abs(x) > 1e-8 for x in self.info.d):
                    raise ValueError("rectified RGB and registered depth are required")
                k = np.asarray(self.info.k).reshape(3, 3)
            capture = Time.from_msg(rgb_msg.header.stamp)
            camera = self.transform("map", rgb_msg.header.frame_id, capture)
            base = self.transform("map", "base_link", capture)
            correction = self.transform("map", "odom", capture)
            if self.map_from_odom is not None:
                delta = np.linalg.inv(self.map_from_odom) @ correction
                if (np.linalg.norm(delta[:2, 3]) > .3 or
                        abs(math.atan2(delta[1, 0], delta[0, 0])) > .25):
                    self.on_goal(String(data=""))
                    self.enabled = False
                    self.status("ERROR", "localization jumped; inspect map, then resend goal and re-arm")
            self.map_from_odom = correction
            self.obstacle = depth_obstacle(depth, k, np.linalg.inv(base) @ camera)
            if self.obstacle:
                self.stop()
                self.status("BLOCKED", "depth obstacle in the forward stop volume")
            self.sequence += 1
            self.latest = Observation(self.sequence, stamp, rgb.copy(), depth.copy(), k.copy(), camera, base, self.grid)
            self.sensor_at = time.monotonic()
            self.last_error = ""
        except Exception as exc:
            self.sensor_at = -math.inf
            self.stop()
            if str(exc) != self.last_error:
                self.status("ERROR", str(exc))
                self.last_error = str(exc)

    def on_goal(self, msg):
        self.stop()
        self.enabled = False
        self.episode += 1
        self.goal = msg.data.strip().lower()
        self.session = None
        self.plan_at = -math.inf
        self.status("IDLE", "goal updated; new session pending" if self.goal else "goal cleared")

    def on_enable(self, request, response):
        if request.data and not self.params["allow_motion"]:
            response.success, response.message = False, "launch has allow_motion=false (shadow only)"
            return response
        if request.data and (not self.goal or self.latest is None):
            response.success, response.message = False, "set a goal and wait for valid sensor data first"
            return response
        self.enabled = request.data
        if not self.enabled:
            self.stop()
        response.success, response.message = True, "armed" if self.enabled else "stopped"
        return response

    def post(self, path, data):
        payload = json.dumps(data, allow_nan=False).encode()
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("request exceeds 16 MiB; reduce map/image resolution")
        headers = {"Content-Type": "application/json"}
        if os.environ.get("TOPOMAP_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["TOPOMAP_TOKEN"]
        request = Request(self.params["server_url"].rstrip("/") + path, payload, headers)
        with urlopen(request, timeout=self.params["request_timeout_s"]) as response:
            return json.loads(response.read(MAX_MESSAGE_BYTES))

    def path_clear(self):
        if self.path_world is None or self.grid is None or self.latest is None:
            return False
        mask = traversable(self.grid, self.params["robot_radius_m"])
        points = np.vstack([self.latest.map_from_base[:2, 3], self.path_world])
        cells = self.grid.world_to_cell(points)
        return all(visible(mask, a, b) for a, b in zip(cells, cells[1:]))

    def allowed(self):
        now = time.monotonic()
        return (self.params["allow_motion"] and not self.obstacle
                and now - self.map_at <= self.params["map_timeout_s"]
                and motion_allowed(enabled=self.enabled, running=self.running, now=now,
                    sensor_at=self.sensor_at, odom_at=self.odom_at, plan_at=self.plan_at,
                    sensor_timeout=self.params["sensor_timeout_s"], plan_timeout=self.params["plan_timeout_s"]))

    def on_command(self, msg):
        age = (self.get_clock().now().nanoseconds - stamp_ns(msg.header.stamp)) / 1e9
        if not self.allowed() or msg.header.frame_id != "base_link" or not 0 <= age <= .25:
            self.brake()
            return
        v, w = bounded_command(msg.twist.linear.x, msg.twist.angular.z,
                               self.params["linear_limit"], self.params["angular_limit"])
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_link"
        command.twist.linear.x, command.twist.angular.z = v, w
        self.command_pub.publish(command)

    def tick(self):
        now = time.monotonic()
        selected = None
        while self.image_queue and now - self.image_queue[0][0] >= self.params["tf_wait_s"]:
            _, rgb_msg, depth_msg = self.image_queue.popleft()
            selected = (rgb_msg, depth_msg)
        if selected is not None:
            self.process_images(*selected)
        if not self.allowed():
            self.brake()
            if self.running and self.enabled and self.params["allow_motion"]:
                self.path_world = None
                self.status("BLOCKED", "motion gate: stale sensor/odom/map/path or depth obstacle")
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            episode, kind, observation, sent_at = self.pending
            try:
                reply = future.result()
                if episode != self.episode:
                    return  # in-flight result from a cancelled/replaced goal
                if kind == "reset":
                    self.session = reply["session"]
                else:
                    validate_reply(reply, observation.seq, observation.stamp_ns)
                    age = (self.get_clock().now().nanoseconds - observation.stamp_ns) / 1e9
                    if not 0 <= age <= self.params["max_result_age_s"]:
                        raise ValueError("planner result is too old; robot remains stopped")
                    message = dict(reply, episode=episode, frame_id="base_link")
                    self.shadow_pub.publish(String(data=json.dumps(message)))
                    if reply["state"] == "RUNNING":
                        yaw = math.atan2(observation.map_from_base[1, 0], observation.map_from_base[0, 0])
                        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                        self.path_world = (np.asarray(reply["waypoints"])[:, :2] @ rotation.T
                                           + observation.map_from_base[:2, 3])
                        if not self.path_clear():
                            raise ValueError("latest map does not certify returned path")
                        self.plan_at = now
                        self.status("BLOCKED" if self.obstacle else "RUNNING",
                                    "depth obstacle" if self.obstacle else reply["reason"])
                        if self.allowed():
                            self.enable_pub.publish(Bool(data=True))
                            self.response_pub.publish(String(data=json.dumps(message)))
                    else:
                        self.stop()
                        self.status(reply["state"], reply["reason"])
                        if reply["state"] == "ARRIVED":
                            self.goal = ""
                            self.enabled = False
            except Exception as exc:
                self.stop()
                self.status("ERROR", str(exc))
        if self.future is not None or not self.goal or now - self.request_at < .5:
            return
        if (self.latest is None or now - self.sensor_at > self.params["sensor_timeout_s"]
                or now - self.odom_at > self.params["sensor_timeout_s"]
                or now - self.map_at > self.params["map_timeout_s"]):
            return
        self.request_at = now
        if self.session is None:
            self.pending = (self.episode, "reset", None, now)
            self.future = self.pool.submit(self.post, "/reset", {"goal": self.goal})
        else:
            observation = self.latest
            self.pending = (self.episode, "step", observation, now)
            self.future = self.pool.submit(self.send_observation, self.session, observation)

    def send_observation(self, session, observation):
        # JPEG/depth serialization belongs on the worker, not the control timer.
        return self.post("/step", {"session": session, "observation": observation.encode()})

    def destroy_node(self):
        self.enabled = False
        self.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
