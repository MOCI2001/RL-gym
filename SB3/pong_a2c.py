from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(0.5e6))
model.save("a2c_pong")
