# Usage :  python enjoy.py CartPole-v0
# Usage :  python enjoy.py Pendulum-v1
# Usage :  python enjoy.py LunarLander-v2

import sys
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env_name = sys.argv[1]
env = make_vec_env(env_name, n_envs=1, seed=0)

model = A2C.load(env_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
