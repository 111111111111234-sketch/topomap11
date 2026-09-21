"""Exercise actual bridge callbacks with ROS transport/message substitutes."""

import importlib.util
import json
import sys
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from topomap_deploy.protocol import Grid, Observation

NOW = 10.


def vector(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def stamp(seconds=10):
    return NS(sec=seconds, nanosec=0)


class Message:
    def __init__(self, data=None):
        self.data = data
        self.header = NS(stamp=stamp(), frame_id="base_link")
        self.twist = NS(linear=vector(), angular=vector())


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=int(NOW * 1e9), to_msg=lambda: stamp()))

    def get_logger(self):
        return NS(info=lambda msg: None)


@pytest.fixture
def bridge_module(monkeypatch):
    fake_modules = {
        "rclpy": NS(), "rclpy.node": NS(Node=FakeNode),
        "rclpy.duration": NS(Duration=lambda **kwargs: None),
        "rclpy.time": NS(Time=NS(from_msg=lambda msg: msg)),
        "rclpy.qos": NS(DurabilityPolicy=NS(TRANSIENT_LOCAL=1), ReliabilityPolicy=NS(RELIABLE=1),
                        QoSProfile=lambda **kwargs: None, qos_profile_sensor_data=None),
        "cv_bridge": NS(CvBridge=object),
        "message_filters": NS(ApproximateTimeSynchronizer=object, Subscriber=object),
        "geometry_msgs.msg": NS(TwistStamped=Message),
        "nav_msgs.msg": NS(OccupancyGrid=Message, Odometry=Message),
        "sensor_msgs.msg": NS(CameraInfo=Message, Image=Message),
        "std_msgs.msg": NS(Bool=Message, String=Message), "std_srvs.srv": NS(SetBool=Message),
        "tf2_ros": NS(Buffer=object, TransformListener=object),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "robot_deploy/src/topomap_go2/topomap_go2/node.py"
    spec = importlib.util.spec_from_file_location("go2_bridge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "time", NS(monotonic=lambda: NOW))
    return module


def node(module, allow_motion=True):
    obj = module.Bridge.__new__(module.Bridge)
    obj.params = dict(allow_motion=allow_motion, sensor_timeout_s=.6, map_timeout_s=3.,
                      plan_timeout_s=1.5, max_result_age_s=2., robot_radius_m=.4,
                      linear_limit=.2, angular_limit=.4, tf_wait_s=.08)
    obj.enabled, obj.running, obj.obstacle = True, True, False
    obj.sensor_at = obj.odom_at = obj.map_at = obj.plan_at = NOW
    obj.goal, obj.episode, obj.sequence, obj.session = "chair", 1, 1, "session"
    obj.future = None
    obj.image_queue = deque(maxlen=10)
    obj.path_world = None
    obj.request_at = NOW
    obj.map_from_odom = None
    obj.last_error = obj.last_status = ""
    obj.grid = Grid(np.zeros((80, 80), np.int8), .1, [-4, -4, 0])
    k = np.array([[50., 0, 30], [0, 50, 20], [0, 0, 1.]])
    optical = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, .5], [0, 0, 0, 1.]])
    obj.latest = Observation(1, int(NOW * 1e9), np.zeros((40, 60, 3), np.uint8),
                             np.full((40, 60), 2., np.float32), k, optical, np.eye(4), obj.grid)
    for name in ("status_pub", "detail_pub", "response_pub", "shadow_pub", "enable_pub", "command_pub"):
        setattr(obj, name, Publisher())
    return obj


def pending_result(obj, episode=1, age=0., state="RUNNING"):
    obs = obj.latest
    obs.stamp_ns = int((NOW - age) * 1e9)
    result = {"seq": obs.seq, "capture_stamp_ns": obs.stamp_ns, "state": state,
              "stop": state == "ARRIVED", "reason": "fixture",
              "waypoints": [[0., 0., 0.], [.2, 0., 0.]] if state == "RUNNING" else []}
    obj.future = Future()
    obj.future.set_result(result)
    obj.pending = (episode, "step", obs, NOW)


def test_shadow_result_never_reaches_mpc(bridge_module):
    obj = node(bridge_module, allow_motion=False)
    pending_result(obj)
    obj.tick()
    assert len(obj.shadow_pub.messages) == 1
    assert not obj.response_pub.messages
    assert all(not m.data for m in obj.enable_pub.messages)
    assert all(m.twist.linear.x == 0. for m in obj.command_pub.messages)


def test_fresh_result_preserves_capture_time_and_drives_only_when_armed(bridge_module):
    obj = node(bridge_module)
    pending_result(obj)
    obj.tick()
    result = json.loads(obj.response_pub.messages[-1].data)
    assert result["capture_stamp_ns"] == int(NOW * 1e9)
    assert result["frame_id"] == "base_link" and result["episode"] == 1
    command = Message()
    command.twist.linear.x, command.twist.angular.z = 9., -9.
    obj.on_command(command)
    assert obj.command_pub.messages[-1].twist.linear.x == .2
    assert obj.command_pub.messages[-1].twist.angular.z == -.4


@pytest.mark.parametrize("episode,age", [(2, 0), (1, 3)])
def test_cancelled_or_old_inflight_reply_cannot_rearm(bridge_module, episode, age):
    obj = node(bridge_module)
    obj.running = False
    pending_result(obj, episode=episode, age=age)
    obj.tick()
    assert not obj.response_pub.messages
    assert not any(m.data for m in obj.enable_pub.messages)


def test_plan_expiry_sends_zero_and_disables_mpc(bridge_module):
    obj = node(bridge_module)
    obj.plan_at = NOW - 2
    obj.tick()
    assert obj.enable_pub.messages[-1].data is False
    assert obj.command_pub.messages[-1].twist.linear.x == 0
    assert not obj.running


def test_goal_change_disarms_and_resets_session(bridge_module):
    obj = node(bridge_module)
    obj.on_goal(Message(data="sofa"))
    assert obj.goal == "sofa" and obj.episode == 2
    assert not obj.enabled and obj.session is None and not obj.running


def test_mpc_cannot_bypass_local_depth_obstacle(bridge_module):
    obj = node(bridge_module)
    obj.obstacle = True
    pending_result(obj)
    obj.tick()
    assert not obj.response_pub.messages
    command = Message()
    command.twist.linear.x = .2
    obj.on_command(command)
    assert obj.command_pub.messages[-1].twist.linear.x == 0


def test_odom_timestamp_and_command_frame_are_checked(bridge_module):
    obj = node(bridge_module)
    message = Message()
    message.header.frame_id = "odom"
    message.child_frame_id = "base_link"
    message.header.stamp = stamp(1)
    obj.on_odom(message)
    assert not obj.allowed()
    message = Message()
    message.header.frame_id = "map"
    message.twist.linear.x = .2
    obj.on_command(message)
    assert obj.command_pub.messages[-1].twist.linear.x == 0


def test_map_obstacle_invalidates_issued_path_and_old_map_is_rejected(bridge_module):
    obj = node(bridge_module)
    obj.path_world = np.array([[0., 0.], [.2, 0.]])
    cells = np.zeros((80, 80), np.int8)
    cells[39:42, 41] = 100
    msg = NS(header=NS(frame_id="map", stamp=stamp()), data=cells.ravel().tolist(),
             info=NS(height=80, width=80, resolution=.1,
                     origin=NS(position=vector(-4, -4), orientation=NS(x=0, y=0, z=0, w=1))))
    obj.on_map(msg)
    assert obj.path_world is None and not obj.running
    assert obj.command_pub.messages[-1].twist.linear.x == 0.
    msg.header.stamp = stamp(1)
    obj.on_map(msg)
    assert obj.grid is None


def image_fixture(obj):
    rgb = NS(header=NS(frame_id="camera_optical", stamp=stamp()), width=60, height=40,
             array=np.zeros((40, 60, 3), np.uint8))
    depth = NS(header=NS(frame_id="camera_optical", stamp=stamp()), encoding="16UC1",
               array=np.full((40, 60), 2000, np.uint16))
    obj.info = NS(header=NS(frame_id="camera_optical"), width=60, height=40,
                  p=np.column_stack([obj.latest.intrinsics, np.zeros(3)]).ravel().tolist())
    obj.cv = NS(imgmsg_to_cv2=lambda msg, **kwargs: msg.array)
    obj.transform = lambda target, source, stamp: obj.latest.map_from_camera if source == "camera_optical" else np.eye(4)
    return rgb, depth


def test_real_image_callback_converts_millimetres_and_rejects_unregistered_depth(bridge_module):
    obj = node(bridge_module)
    rgb, depth = image_fixture(obj)
    obj.process_images(rgb, depth)
    np.testing.assert_allclose(obj.latest.depth, 2.)
    assert obj.sensor_at == NOW and not obj.obstacle
    depth.header.frame_id = "unregistered_depth_optical"
    obj.process_images(rgb, depth)
    assert not obj.allowed() and not obj.running


def test_localization_jump_cancels_task_and_clears_inflight_authority(bridge_module):
    obj = node(bridge_module)
    rgb, depth = image_fixture(obj)
    obj.map_from_odom = np.eye(4)
    jump = np.eye(4)
    jump[0, 3] = 1.
    obj.transform = lambda target, source, stamp: (jump if source == "odom" else
                    obj.latest.map_from_camera if source == "camera_optical" else np.eye(4))
    obj.process_images(rgb, depth)
    assert obj.goal == "" and not obj.enabled and obj.session is None and obj.episode == 2


def test_invalid_quaternion_is_rejected(bridge_module):
    with pytest.raises(ValueError):
        bridge_module.quaternion_matrix(NS(x=0, y=0, z=0, w=0))


def test_images_wait_for_capture_time_tf_without_blocking_callbacks(bridge_module):
    obj = node(bridge_module)
    rgb, depth = image_fixture(obj)
    received = []
    obj.process_images = lambda rgb, depth: received.append((rgb, depth))
    obj.on_images(rgb, depth)
    obj.tick()
    assert not received
    obj.image_queue[0] = (NOW - .1, rgb, depth)
    obj.tick()
    assert received == [(rgb, depth)]
