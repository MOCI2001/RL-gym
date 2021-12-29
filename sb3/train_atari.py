## pip install stable-baselines3
## pip install gym[atari] (For Windows, pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari-py)
# Usage: python train.py Pong 16000
# Env_Name: listed in Env_Name.txt

import sys
from stable_baselines3 import DQN, A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
#from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise # for DDPG, TD3

env_name = sys.argv[1]
timesteps = sys.argv[2]

#env = make_atari_env(env_name+"-v0", n_envs=16, seed=0)           # n_envs=1 for DQN, SAC, DDPG, TD3
env = make_atari_env(env_name+"NoFrameskip-v4", n_envs=16, seed=0) # n_envs=1 for DQN, SAC, DDPG, TD3
env = VecFrameStack(env, n_stack=4)                                # not for DQN, SAC, DDPG, TD3

# For DDPG, TD3 (n_envs=1)
#n_actions = env.action_space.shape[-1] 
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = A2C('MlpPolicy', env, verbose=1)
#model = DQN('MlpPolicy', env, verbose=1)
#model = PPO('MlpPolicy', env, verbose=1)
#model = SAC('MlpPolicy', env, verbose=1)
#model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
#model = TD3('MlpPolicy',  env, action_noise=action_noise, verbose=1)

model.learn(total_timesteps=int(timesteps))

model.save(env_name)
