## !pip install highway-env
## !pip install stable-baselines3
import time
import gym
import highway_env
import numpy as np
from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("parking-v0")

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

start_t = time.time()
model = DDPG("MultiInputPolicy", env, action_noise=action_noise,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                max_episode_length=100,
                online_sampling=True,),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256,256,256]),
            )
model.learn(int(1e6))
model.save("her_ddpg_parking")

end_t = time.time()
print("Total Execution Time = ", end_t - start_t)
