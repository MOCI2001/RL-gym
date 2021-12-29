# Usage :  python enjoy.py CartPole
# Usage :  python enjoy.py Pendulum
# Usage :  python enjoy.py LunarLander

import sys
import gym
from stable_baselines3 import A2C

env_name = sys.argv[1]

if env_name=="LunarLander":
    env = gym.make(env_name+"-v2") # For LunarLander
else:
    env = gym.make(env_name+"-v0") # For CartPole, Pendulum

model = A2C.load(env_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
