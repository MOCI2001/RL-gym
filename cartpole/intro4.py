# assign your own action space
import gym
from gym import spaces

env = gym.make('CartPole-v0')
env.action_space = spaces.Discrete(1) # Set it to only 1 elements {0}

for i_episode in range(5): #how many episodes you want to run
    observation = env.reset() #reset() returns initial observation

    for t in range(200):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
