# Usage: python enjoy_gif.py CartPole
# Usage: python enjoy_gif.py Pendulum
# Usage: python enjoy_gif.py LunarLander
 
import sys
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from PIL import Image

env_name = sys.argv[1]

if env_name =="LunarLander":
    env = gym.make(env_name+"-v2") # For LunarLander
else if env_name=="CartPole" or env_name=="Pendulum":
    env = gym.make(env_name+"-v0") # For CartPole, Pendulum
else:
    env = make_vec_env(env_name+"-v0", n_envs=4, seed=0) 
    
model = A2C.load(env_name)

obs = env.reset()
images = []
img = env.render(mode='rgb_array')
for i in range(160): #480 
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    img = env.render(mode='rgb_array')
    images.append(Image.fromarray(img))

images[0].save(env_name+'.gif', save_all=True, append_images=images[1:], optimize=False, duration=30, loop=0)
