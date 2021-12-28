import gym
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("Pendulum-v0", n_envs=1, seed=0)

model = SAC.load("pendulum")
#model = DDPG.load("pendulum")
#model = TD3.load("pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
