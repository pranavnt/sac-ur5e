import gymnasium as gym
from ur5e_env import ReachingUR5eEnv

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1", render_mode="human")

model = SAC("MlpPolicy", env, verbose=1)

env = ReachingUR5eEnv(render=True)

