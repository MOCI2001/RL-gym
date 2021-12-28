import sys
from stable_baselines3 import DQN, A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env_id = sys.argv[1]

env_name = {'breakout':'BreakoutNoFrameskip-v4', 'qbert':'QbertNoFrameskip-v4'} 

env = make_atari_env(env_name[env_id], n_envs=1, seed=0)
#env = make_atari_env(env_name[env_id], n_envs=4, seed=0)
#env = VecFrameStack(env, n_stack=4)

model = A2C.load(env_id)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
