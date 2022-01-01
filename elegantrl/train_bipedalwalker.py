## pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git
import sys
import gym
from PIL import Image
from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.envs.Gym import build_env
gym.logger.set_level(40) # Block warning

env_name = 'BipedalWalker-v3' #sys.argv[1]
env = build_env(env_name)

agent= AgentTD3()
args = Arguments(env,agent)
args.eval_times1 = 2 **3
args.eval_times2 = 2 **5

args.gamma = 0.98
args.target_step = args.env.max_step

train_and_evaluate(args) # the training process will terminate once it reaches the target reward

