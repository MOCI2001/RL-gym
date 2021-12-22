# Observation
# reset will restart the environment
import gym
  
env = gym.make('CartPole-v0')
  
for i_episode in range(5):    #how many episodes you want to run
    observation = env.reset() #reset() returns initial observation
  
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
