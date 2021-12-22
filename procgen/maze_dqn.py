# !pip install stable-baselines3
# !pip install procgen

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack

env = gym.make('procgen:procgen-maze-v0')
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("dqn_maze")

## To reload model & replay
del model
model = DQN.load("dqn_maze")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
