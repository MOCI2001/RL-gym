from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

model.save("a2c_pong")

## load model and continue the training
#model = A2C.load("a2c_pong", verbose=1)
#env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
#model.set_env(env)
#model.learn(int(0.5e6))
#model.save("a2c_pong_2")

#env = model.get_env()
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
del model # remove to demonstrate saving and loading

model = A2C.load("a2c_pong")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
