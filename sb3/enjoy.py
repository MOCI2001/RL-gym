# Usage :  python enjoy.py CartPole
# Usage :  python enjoy.py Pendulum
# Usage :  python enjoy.py LunarLander

import sys
import gym
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

env_name = sys.argv[1]

if env_name=="LunarLander":
    env = make_vec_env(env_name+"-v2", n_envs=1, seed=0) # For LunarLander
else:
    env = make_vec_env(env_name+"-v0", n_envs=1, seed=0) # For CartPole, Pendulum

model = A2C.load(env_name)
#model = DQN.load(env_name)
#model = PPO.load(env_name)
#model = SAC.load(env_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
