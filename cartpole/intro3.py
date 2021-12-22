# Space
# check Action and Observation space
import gym
env = gym.make('CartPole-v0')
## Check dimension of spaces ##
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
## Check range of spaces ##
"""
print(env.action_space.high)-
You'll get error if you run this, because 'Discrete' object has no attribute 'high'
"""
print(env.observation_space.high)
print(env.observation_space.low)
