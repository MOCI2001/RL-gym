from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=1) # DQN only for single environment

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000, log_interval=4)
model.save("dqn_cartpole")
