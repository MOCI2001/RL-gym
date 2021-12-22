# RL-Gym

## required packages
`pip install stable_baselines3`<br>
`pip install gym[atari]`<br>
`pip install gym-super-mario-bros`<br>
`git clone https://github.com/rkuo2000/RL-Gym`<br>
`cd RL-Gym`<br>

## cartpole 
Gym cartpole sample codes<br>
`cd cartpole`

* random_action.py
* q_learn.py
* dqn.py
* ddqn.py

`python q_learning.py`

## SB3
`cd SB3`

* cartpole_a2c.py
* cartpole_dqn.py
* cartpole_ppo.py
* pendulum_ddpg.py
* pendulum_sac.py
* pendulum_td3.py

`python pendulum_ddpg.py`

## mario
[Tutorial: Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br>
Repro [Github](https://github.com/yuansongFeng/MadMario/)<br>
`cd mario`

*Training time is around 80 hours on CPU and 20 hours on GPU.*<br>
To train : (epochs=40000)<br>
`python main.py`

To replay: ("trained_mario.chkpt")<br>
`python replay.py`

![](https://pytorch.org/tutorials/_images/mario.gif)
