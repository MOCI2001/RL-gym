import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
#import imageio
#import numpy as np
from PIL import Image

env = make_vec_env("LunarLander-v2", n_envs=1, seed=0)

model = A2C.load("lunarlander")

obs = env.reset()
images = []
img = env.render(mode='rgb_array')
for i in range(480):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    img = env.render(mode='rgb_array')
    #images.append(img)
    images.append(Image.fromarray(img))

#imageio.mimsave('lunarlander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
images[0].save('lunarlander.gif', save_all=True, append_images=images[1:], optimize=False, duration=30, loop=0)
