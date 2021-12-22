# RL-Gym

## required packages
`pip install stable_baselines3`<br>
`pip install gym[atari]`<br>
`pip install gym-super-mario-bros`<br>

## cartpole 
Gym cartpole sample codes
* random_action.py
* q_learn.py
* dqn.py
* ddqn.py
`python q_learning.py`

## mario
RL SuperMarioBros using DDQN<br>
`cd mario`<br>
To train : (epochs=40000)<br>
`python main.py`<br>
To replay: ("trained_mario.chkpt")<br>
`python replay.py`<br>

## SB3
RL Humanoid examples:
* humanoid_a2c.py
* humanoid_ddpg.py
* humanoid_ppo.py
* humanoidrun_a2c.py

RL inverted-pendulum examples:
* invertedpendulum_a2c.py
* invertedpendulum_ddpg.py
* invertedpendulum_sac.py
* invertedpendulum_td3.py
