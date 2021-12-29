## pip install stable-baselines3
## pip install Box2D box2d-py  (For LunarLander)
# Usage: python train.py CartPole 160000
# Usage: python train.py Pendulum 1000000
# Usage: python train.py LunarLander 640000
import time
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.vec_env  import VecFrameStack

env_name = sys.argv[1]
timesteps= sys.argv[2]

if env_name=="LunarLander":
    env = make_vec_env(env_name+"-v2", n_envs=4, seed=0)
    #env = VecFrameStack(env, n_stack=4)
else: # "CartPole" or "Pendulum"
    env = make_vec_env(env_name+"-v0", n_envs=1, seed=0)

model = A2C('MlpPolicy', env, verbose=1)

start_t = time.time()
model.learn(total_timesteps=int(timesteps))
end_t   = time.time()
print("Execution Time = {} sec".format(int(end_t-start_t)))

model.save(env_name)
