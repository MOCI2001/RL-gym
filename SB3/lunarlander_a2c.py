## apt install wig
## apt install box2d box2d-kengz
#import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

#env = gym.make('LunarLander-v2')
env = make_vec_env("LunarLander-v2", n_envs=4) # using 4 environments to train

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e5))
model.save("a2c_lunarlander")
