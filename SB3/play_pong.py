from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('PongNoFrameskip-v4', n_envs=4)
env = VecFrameStack(env, n_stack=4)

model = A2C.load("a2c_pong")
#model = PPO.load("ppo_pong")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
