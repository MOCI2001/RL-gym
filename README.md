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

## SB3
`cd SB3`<br>

* cartpole_a2c.py
* cartpole_dqn.py
* cartpole_ppo.py

&emsp;`python cartple_a2c.py`<br>
&emsp;`python play_cartpole.py`<br>

* pendulum_ddpg.py
* pendulum_sac.py
* pendulum_td3.py

&emsp;`python pendulum_ddpg.py`<br>
&emsp;`python play_pendulum.py`<br>

* pong_a2c.py
* breakout_ppo.py
* lunarlander_dqn.py

&emsp;You can use Kaggle to train above game agents, then download the .zip to play on PC<br>
&emsp;`python pong_a2c.py`<br> 
&emsp;`python play_pong.py`<br>

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
