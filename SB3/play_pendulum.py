import gym
from stable_baselines3 import DDPG, SAC, TD3

env = gym.make('Pendulum-v0')

model = DDPG.load("ddpg_pendulum")
#model = SAC.load("sac_pendulum")
#model = TD3.load("td3_pendulum")

obs = env.reset()
#while True:
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
