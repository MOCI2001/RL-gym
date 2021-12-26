## pip install Box2D
## pip install box2d-py
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v2", n_envs=64) # using 16 environments to train

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e6))
model.save("a2c_lunarlander")
