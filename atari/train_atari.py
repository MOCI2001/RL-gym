# Usage: python train_atari.py pong 100000
import sys
from envID_v4 import envIDs
from stable_baselines3 import DQN, A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
#from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

game_name = sys.argv[1]
timesteps = sys.argv[2]
env_id = envIDs[game_name]
print(env_id)

env = make_atari_env(env_id, n_envs=16, seed=0)
env = VecFrameStack(env, n_stack=4)

## For DDPG, TD3
#n_actions = env.action_space.shape[-1]
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = A2C('MlpPolicy', env, verbose=1)
#model = DQN('MlpPolicy', env, verbose=1)
#model = PPO('MlpPolicy', env, verbose=1)
#model = SAC('MlpPolicy', env, verbose=1)
#model = DDPG('MlpPolicy', env, verbose=1)
#model = TD3('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=int(timesteps))

model.save(game_name)
