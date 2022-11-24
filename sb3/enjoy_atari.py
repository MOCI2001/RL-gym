# Usage: python enjoy_atari.py Breakout-v5
# Env_Name: listed in Env_Name.txt

import sys
import gym
from stable_baselines3 import DQN, A2C, PPO, SAC, DDPG, TD3
#from stable_baselines3.common.env_util import make_atari_env
#from stable_baselines3.common.vec_env import VecFrameStack

from ale_py import ALEInterface
ale = ALEInterface()

env_name = sys.argv[1]

env = gym.make('ALE/'+env_name)
#env = make_atari_env(env_name+"-v0", n_envs=4, seed=0) # n_envs=1 for DQN, SAC, DDPG, TD3
#env = make_atari_env(env_name+"NoFrameskip-v4", n_envs=4, seed=0) # n_envs=1 for DQN, SAC, DDPG, TD3
#env = make_atari_env(env_name)
#env = VecFrameStack(env,n_stack=4)

model = A2C.load(env_name)
#model = DQN.load(env_name)
#model = PPO.load(env_name)
#model = SAC.load(env_name)
#model = DDPG.load(env_name)
#model = TD3.load(env_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
