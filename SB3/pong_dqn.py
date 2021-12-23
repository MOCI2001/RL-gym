from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)

model.save("dqn_pong")

env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)

del model # remove to demonstrate saving and loading
model = DQN.load("dqn_pong")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
