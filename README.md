# RL-Gym

## required packages
`pip install stable_baselines3`<br>
`pip install gym[atari]`<br>
`git clone https://github.com/rkuo2000/RL-Gym`<br>
`cd RL-Gym`<br>

## cartpole 
Gym cartpole sample codes<br>
* random_action.py
* q_learn.py
* dqn.py
* ddqn.py

`cd cartpole`<br>
`python q_learning.py`

## stock
`cd stock`<br>
`python stock_dqn.py GOOGL`<br>
`python play_stock.py GOOGL`<br>

## SB3
`cd SB3`<br>

**DQN, A2C, PPO**<br>
`python train_cartple.py`<br>
`python play_cartpole.py`<br>

**DDPG, SAC, TD3**<br>
`python train_pendulum.py`<br>
`python play_pendulum.py`<br>


* pong_a2c.py
* breakout_ppo.py
* lunarlander_a2c.py
* 
**LunarLander**<br>
`python train_lunarlander.py`<br> 
`python play_lunarlander.py`<br>
`python gif_lunarlander.py`<br>

## mario
[Tutorial: Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br>
Repro [Github](https://github.com/yuansongFeng/MadMario/)<br>
`pip install gym-super-mario-bros`<br>
`cd mario`

*Training time is around 80 hours on CPU and 20 hours on GPU.*<br>
To train : (epochs=40000)<br>
`python main.py`

To replay: ("trained_mario.chkpt")<br>
`python replay.py`

![](https://pytorch.org/tutorials/_images/mario.gif)
