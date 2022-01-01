## pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git
# Impor Packages
import sys
import gym
from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentSAC import AgentModSAC
#from elegantrl.envs.Gym import build_env

env_name = 'AntBulletEnv-v0' # sys.argv[1]
env = build_env(env_name)

agent= AgentModSAC()
args = Arguments(env, agent)
GPU_ID = 0

args.learner_gpus = (GPU_ID, )
args.agent.if_use_act_target = False
args.gamma = 0.98
args.net_dim = 2 **9
args.max_memo = 2 **22
args.repeat_times = 2 **1
args.reward_scale = 2 **-2
args.batch_size = args.net_dim * 2
args.target_step = args.env.max_step * 2

args.eval_gap = 2 **8
args.eval_times1 = 2 **1
args.eval_times2 = 2 **4
args.break_step = int(8e7)
args.if_allow_break = False

args.worker_num = 4

train_and_evaluate(args)
