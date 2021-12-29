## pip install stable-baselines3
## pip install Box2D box2d-py  (For LunarLander)
# Usage: python train.py CartPole 320000
# Usage: python train.py Pendulum 32000
# Usage: python train.py LunarLander 500000

import sys
import gym
import numpy as np
from stable_baselines3 import A2C

env_name = sys.argv[1]
timesteps= sys.argv[2]

if env_name=="LunarLander":
    env = gym.make(env_name+"-v2") # For LunarLander
else:
    env = gym.make(env_name+"-v0") # For CartPole, Pendulum

model = A2C('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=int(timesteps))

model.save(env_name)
