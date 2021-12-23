from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
model = A2C.load("a2c_pong")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
