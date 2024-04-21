import gymnasium as gym
from ur5e_env import ReachingUR5eEnv
from stable_baselines3 import SAC

# Create the environment
env = ReachingUR5eEnv(render=True)

# Create the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Set the number of training timesteps
total_timesteps = 100000

# Training loop
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("sac_reaching_ur5e")

# Evaluate the trained model
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

# Close the environment
env.close()