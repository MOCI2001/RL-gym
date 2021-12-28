### use OpenAI GYM to run one of Atari games (https://gym.openai.com/envs/#atari)
# pip install gym['atari']
# python gym_atari.py
import sys
import gym
from envID_v0 import envIDs

game_name = sys.argv[1]
env_id = envIDs[game_name]
print(env_id)

env = gym.make(env_id)
env.reset()

for _ in range(1000): # play 1000 frames
    env.step(env.action_space.sample()) # random action
    env.render()
env.close()
