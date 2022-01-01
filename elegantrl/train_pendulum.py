## pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git
# Impor Packages
import sys
import gym
from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.Gym import build_env
gym.logger.set_level(40) # Block warning

env_name = 'Pendulum-v0' # sys.argv[1]
env = build_env(env_name)

agent= AgentPPO()
args = Arguments(env,agent)

args.gamma = 0.98
args.netdm = 2 **8
args.worker_num = 2
args.reward_scale = 2 **-2
args.target_step = 200 *16 # max_step = 200
args.eval_gap = 2 **5

# Train & Evaluate the Agent
train_and_evaluate(args) # the training process will terminate once it reaches the target reward
