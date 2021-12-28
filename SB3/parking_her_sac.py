## !pip install highway-env
## !pip install stable-baselines3
import gym
import highway_env
from stable_baselines3 import HerReplayBuffer, SAC

env = gym.make("parking-v0")

model = SAC("MultiInputPolicy", env, 
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
model.save("her_sac_parking")
