from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=1)

model = DQN.load("dqn_cartpole")
#model = A2C.load("a2c_cartpole")
#model = PPO.load("ppo_cartpole")

obs = env.reset()
#while True:
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
