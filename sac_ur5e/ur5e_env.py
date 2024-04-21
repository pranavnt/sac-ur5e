import mujoco
import random
import time
import mujoco.viewer
from dm_control import mujoco as dm_mujoco
from ur5e_ik import qpos_from_site_pose

class ReachingUR5eEnv:
    def __init__(self, render=False):
        self.model = mujoco.MjModel.from_xml_path("./mujoco_menagerie/universal_robots_ur5e/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.physics = dm_mujoco.Physics.from_xml_path("./mujoco_menagerie/universal_robots_ur5e/scene.xml")
        self.render = render
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def step(self):
        new_pos = (random.random()*5, random.random()*5, random.random()*5)
        ik_result = qpos_from_site_pose(
            self.physics,
            site_name="attachment_site",
            target_pos=new_pos,
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            max_steps=100
        )
        self.data.qpos[:6] = ik_result.qpos[:6]
        mujoco.mj_step(self.model, self.data)
        if self.render:
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
                self.viewer.sync()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.render:
            self.viewer.reset()

    def close(self):
        if self.render:
            self.viewer.close()