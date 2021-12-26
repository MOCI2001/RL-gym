import sys
import gym
game_name = sys.argv[1]
env = gym.make(game_name)
state = env.reset()
print(state.shape)

