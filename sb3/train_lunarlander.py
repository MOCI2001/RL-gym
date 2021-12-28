## pip install Box2D
## pip install box2d-py
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v2", n_envs=1, seed=0)

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(0.5e6))
model.save("lunarlander")
