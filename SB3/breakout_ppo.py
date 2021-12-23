from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('Breakout-v4', n_envs=4)

## Create & Train Agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e6))
model.save("ppo_breakout")
