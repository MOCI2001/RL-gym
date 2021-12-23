from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('Breakout-v4', n_envs=1)
model = PPO.load("ppo_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
