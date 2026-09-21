"""Pure control checks shared by the ROS bridge and CPU tests."""

import math

import numpy as np


def depth_obstacle(depth, intrinsics, base_from_camera, distance=.75, half_width=.3):
    """Immediate forward stop volume in base_link; supplements the SLAM map.

    The vertical band excludes nominal ground beneath a standing Go2. It is not
    a substitute for terrain sensing or the vendor's obstacle avoidance.
    """
    vv, uu = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
    z = depth[::4, ::4].ravel()
    valid = np.isfinite(z) & (z > .15) & (z < 4.)
    if not valid.any():
        return True
    rays = np.linalg.solve(intrinsics, np.vstack([uu.ravel()[valid], vv.ravel()[valid], np.ones(valid.sum())]))
    points = (base_from_camera[:3, :3] @ (rays * z[valid]) + base_from_camera[:3, 3:4]).T
    inside = ((points[:, 0] > 0) & (points[:, 0] < distance)
              & (np.abs(points[:, 1]) < half_width)
              & (points[:, 2] > -.15) & (points[:, 2] < .6))
    return bool(inside.sum() >= 3)


def validate_reply(reply, sequence, stamp):
    if type(reply.get("seq")) is not int or type(reply.get("capture_stamp_ns")) is not int:
        raise ValueError("reply sequence and timestamp must be integers")
    if reply.get("seq") != sequence or reply.get("capture_stamp_ns") != stamp:
        raise ValueError("reply does not match the pending observation")
    state = reply.get("state")
    if state not in ("RUNNING", "BLOCKED", "ARRIVED") or type(reply.get("stop")) is not bool:
        raise ValueError("invalid planner state")
    if reply["stop"] != (state == "ARRIVED"):
        raise ValueError("inconsistent stop flag")
    points = reply.get("waypoints")
    if not isinstance(points, list) or len(points) > 100:
        raise ValueError("invalid waypoints")
    if state == "RUNNING" and not points:
        raise ValueError("RUNNING reply has no path")
    if state != "RUNNING" and points:
        raise ValueError("non-running reply must not contain a path")
    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("invalid waypoint")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
            raise ValueError("non-finite waypoint")
        if math.hypot(point[0], point[1]) > 2. or abs(point[2]) > math.pi + 1e-5:
            raise ValueError("waypoint outside local motion envelope")
    return reply


def motion_allowed(*, enabled, running, now, sensor_at, odom_at, plan_at,
                   sensor_timeout, plan_timeout):
    return (enabled and running and
            0 <= now - sensor_at <= sensor_timeout and
            0 <= now - odom_at <= sensor_timeout and
            0 <= now - plan_at <= plan_timeout)


def bounded_command(v, w, v_limit, w_limit):
    if not all(math.isfinite(x) for x in (v, w, v_limit, w_limit)) or min(v_limit, w_limit) <= 0:
        return 0., 0.
    return max(-v_limit, min(v_limit, v)), max(-w_limit, min(w_limit, w))
