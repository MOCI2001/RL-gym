import gym
from stable_baselines3 import A2C, DQN, PPO

env = gym.make("CartPole-v1")

#model = DQN.load("cartpole")
#model = A2C.load("cartpole")
model = PPO.load("cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
