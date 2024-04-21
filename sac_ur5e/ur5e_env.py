import gymnasium as gym
import random
from gymnasium import spaces
import numpy as np
import mujoco
from dm_control import mujoco as dm_mujoco
from ur5e_ik import qpos_from_site_pose

class ReachingUR5eEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("./mujoco_menagerie/universal_robots_ur5e/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.physics = dm_mujoco.Physics.from_xml_path("./mujoco_menagerie/universal_robots_ur5e/scene.xml")
        self.render_mode = render

        # Define the observation space
        obs_low = np.full(6, -np.inf)
        obs_high = np.full(6, np.inf)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Define the action space
        action_low = np.array([-5, -5, -5])
        action_high = np.array([5, 5, 5])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.target_pos = np.array([0.5, 0.5, 0.5])

    def step(self, action):
        new_pos = action
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

        observation = self.data.qpos[:6]
        dist_to_target = np.linalg.norm(observation[:3] - self.target_pos)
        reward = -dist_to_target
        if reward > -0.5:
            done = True
            print("DONEEE")
            self.reset()
            self.target_pos = np.array([random.random(), random.random(), random.random()])
        else:
            done = False
        info = {}
        truncated = False

        print(reward)

        return observation, reward, done, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        observation = self.data.qpos[:6]
        return observation, {}

    def render(self):
        if self.render_mode:
            mujoco.mj_forward(self.model, self.data)
            mujoco.mjr_render(self.model, self.data)

    def close(self):
        pass