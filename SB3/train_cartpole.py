import gym
from stable_baselines3 import DQN, A2C, PPO

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)
#model = A2C("MlpPolicy", env, verbose=1)
#model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=25000)

model.save("cartpole")
