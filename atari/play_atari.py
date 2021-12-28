import sys
from envID_v4 import envIDs
from stable_baselines3 import DQN, A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

game_name = sys.argv[1]
env_id = envIDs[game_name]
print(env_id)

#env = make_atari_env(env_id, n_envs=1, seed=0)
env = make_atari_env(env_id, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

model = A2C.load(game_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
