#import imageio
#import numpy as np
from PIL import Image
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v2", n_envs=1)

model = A2C.load("a2c_lunarlander")

images = []
obs = env.reset()
img = env.render(mode='rgb_array')

for i in range(500):
    #images.append(img)
    images.append(Image.fromarray(img))
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    img = env.render(mode='rgb_array')

#imageio.mimsave('lunarlander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
images[0].save('lunarlander_a2c.gif', save_all=True, append_images=images[1:], optimize=False, duration=30, loop=0)
