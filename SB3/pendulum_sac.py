import  gym
from stable_baselines3 import SAC

# SAC only for single environment
env = gym.make("Pendulum-v0")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_pendulum")
