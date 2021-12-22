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
* 
`python q_learning.py`<br>

## mario
[Tutorial: Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br>
Repro [Github](https://github.com/yuansongFeng/MadMario/)<br>
`cd mario`<br>

*Training time is around 80 hours on CPU and 20 hours on GPU.*<br>
To train : (epochs=40000)<br>
`python main.py`<br>

To replay: ("trained_mario.chkpt")<br>
`python replay.py`<br>
![](https://pytorch.org/tutorials/_images/mario.gif)

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
