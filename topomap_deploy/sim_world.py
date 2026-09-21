"""MuJoCo contact dynamics for a planar navigation proxy, not a Go2 gait model."""

import math

import mujoco
import numpy as np

WIDTH, HEIGHT = 320, 240
DT = .005


def world_xml(obstacle=True):
    block = '<geom name="obstacle" type="box" pos="2.1 0 .45" size=".35 .55 .45" rgba=".15 .35 .65 1"/>' if obstacle else ''
    return f'''<mujoco model="topomap_navigation_proxy">
      <compiler angle="radian"/>
      <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/><map znear=".02" zfar="20"/></visual>
      <default><geom friction=".8 .01 .001" solref=".005 1"/></default>
      <worldbody>
        <light pos="1 0 5" dir="0 0 -1" diffuse=".9 .9 .9"/>
        <geom name="floor" type="plane" size="9 9 .1" rgba=".72 .72 .68 1"/>
        <geom name="north_wall" type="box" pos="2 2.9 1" size="4 .1 1" rgba=".8 .8 .82 1"/>
        <geom name="south_wall" type="box" pos="2 -2.9 1" size="4 .1 1" rgba=".8 .8 .82 1"/>
        <geom name="east_wall" type="box" pos="5.9 0 1" size=".1 3 1" rgba=".8 .8 .82 1"/>
        <geom name="west_wall" type="box" pos="-1.9 0 1" size=".1 3 1" rgba=".8 .8 .82 1"/>
        {block}
        <geom name="red_marker" type="box" pos="4.6 .8 .6" size=".2 .25 .6" rgba=".95 .025 .025 1"/>
        <body name="proxy" pos="0 0 .25">
          <joint name="x" type="slide" axis="1 0 0" damping="1"/>
          <joint name="y" type="slide" axis="0 1 0" damping="1"/>
          <joint name="yaw" type="hinge" axis="0 0 1" damping=".3"/>
          <inertial pos="0 0 0" mass="10" diaginertia=".4 .6 .7"/>
          <geom name="robot_body" type="box" size=".30 .18 .13" rgba=".04 .12 .13 1"/>
          <geom type="box" pos="0 0 .14" size=".2 .13 .03" rgba=".05 .75 .65 1" contype="0" conaffinity="0"/>
          <camera name="rgbd" pos=".15 0 .35" xyaxes="0 -1 0 .342 0 .940" fovy="90"/>
        </body>
      </worldbody>
      <actuator>
        <velocity name="vx" joint="x" kv="80" forcerange="-40 40"/>
        <velocity name="vy" joint="y" kv="80" forcerange="-40 40"/>
        <velocity name="w" joint="yaw" kv="15" forcerange="-8 8"/>
      </actuator>
    </mujoco>'''


class World:
    def __init__(self, obstacle=True, *, xml_path=None):
        # External scenes must retain the proxy/joint/camera names and actuator
        # contract. No scene geometry is exported to the navigation algorithm.
        self.model = (mujoco.MjModel.from_xml_path(str(xml_path)) if xml_path is not None
                      else mujoco.MjModel.from_xml_string(world_xml(obstacle)))
        if (self.model.nq, self.model.nv, self.model.nu) != (3, 3, 3):
            raise ValueError("World requires the three-DOF planar proxy, not a quadruped model")
        for index, name in enumerate(("x", "y", "yaw")):
            joint = self.model.joint(name).id
            if (self.model.jnt_dofadr[joint] != index
                    or self.model.actuator_trnid[index, 0] != joint):
                raise ValueError("External scene changed the x/y/yaw actuator contract")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.body_id = self.model.body("proxy").id
        self.geom_id = self.model.geom("robot_body").id
        self.camera_id = self.model.camera("rgbd").id
        self.contacts = 0
        self.steps = 0
        self.max_penetration = 0.
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self):
        return float(self.data.time)

    def pose(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        xy = self.data.xpos[self.body_id, :2]
        return np.array([xy[0], xy[1], math.atan2(rotation[1, 0], rotation[0, 0])])

    def speed(self):
        """Measured generalized velocity, not the actuator target."""
        return float(np.linalg.norm(self.data.qvel[:2])), float(abs(self.data.qvel[2]))

    def step(self, command, seconds=.1):
        for _ in range(round(seconds / DT)):
            yaw = self.pose()[2]
            # The ONLY state advancement is mj_step. Commands are actuator
            # velocity targets with finite force, not qpos/pose assignments.
            self.data.ctrl[:] = [command[0] * math.cos(yaw), command[0] * math.sin(yaw), command[1]]
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            for contact in self.data.contact:
                if self.geom_id in (contact.geom1, contact.geom2) and contact.dist < 0:
                    self.contacts += 1
                    self.max_penetration = max(self.max_penetration, -float(contact.dist))

    def observe(self):
        self.renderer.update_scene(self.data, camera="rgbd")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="rgbd")
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        fy = HEIGHT / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id]) / 2))
        k = np.array([[fy, 0, WIDTH/2], [0, fy, HEIGHT/2], [0, 0, 1.]])
        camera = np.eye(4)
        camera[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        camera[:3, 3] = self.data.cam_xpos[self.camera_id]
        base = np.eye(4)
        base[:3, :3] = self.data.xmat[self.body_id].reshape(3, 3)
        base[:3, 3] = self.data.xpos[self.body_id]
        return rgb, depth, k, camera, base

    def overhead(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [2., 0., 0.]
        camera.distance = 9.
        camera.azimuth = 90.
        camera.elevation = -90.
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def target_distance_for_evaluation(self):
        # Privileged target coordinates are evaluator-only, never sent to the
        # detector, mapper, HGR memory, planner, or controller. RGB-D estimates
        # a visible surface, so evaluate base-to-box surface distance, not center.
        target_id = self.model.geom("red_marker").id
        target = self.data.geom_xpos[target_id, :2]
        half_size = self.model.geom_size[target_id, :2]
        delta = np.maximum(np.abs(target - self.pose()[:2]) - half_size, 0.)
        return float(np.linalg.norm(delta))

    def close(self):
        self.renderer.close()
