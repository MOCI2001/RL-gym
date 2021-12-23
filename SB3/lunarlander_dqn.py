## apt install wig
## apt install box2d box2d-kengz
import gym
from stable_baselines3 import DQN

env = gym.make('LunarLander-v2')

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e5))
model.save("dqn_lunarlander")
