import gym
from stable_baselines3 import DQN

env = gym.make ('LunarLander-v2')

model = DQN.load("dqn_lunarlander")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
