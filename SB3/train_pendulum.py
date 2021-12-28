import gym
import numpy as np
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = make_vec_env("Pendulum-v0", n_envs=1, seed=0)

# The noise objeccts for DDPG / TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)
#model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
#model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

model.learn(total_timesteps=16000)

model.save("pendulum")
