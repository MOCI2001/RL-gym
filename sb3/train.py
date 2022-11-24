## pip install stable-baselines3
## pip install Box2D box2d-py  (For LunarLander)
# Usage: python train.py CartPole-v0 160000
# Usage: python train.py Pendulum-v1 1000000
# Usage: python train.py LunarLander-v2 640000
import time
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.vec_env  import VecFrameStack

env_name = sys.argv[1]
timesteps= sys.argv[2]

if env_name=="LunarLander":
    env = make_vec_env(env_name, n_envs=4)
else:
    env = make_vec_env(env_name, n_envs=1)

model = A2C('MlpPolicy', env, verbose=1)

start_t = time.time()
model.learn(total_timesteps=int(timesteps))
end_t   = time.time()
print("Execution Time = {} sec".format(int(end_t-start_t)))

model.save(env_name)
