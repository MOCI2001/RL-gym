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
`python random_action.py`<br>
`python q_learning.py`<br>
`python dqn.py`<br>

## sb3
`cd sb3`<br>

### CartPole, Pendulum, LunarLander
**Train**<br>
`python train.py CartPole 32000`<br>
`python train.py Pendulum 32000`<br>
`python train.py LunarLander 500000`<br>

**Enjoy**<br>
`python enjoy.py CartPole`<br>
`python enjoy.py Pendulum`<br>
`python enjoy.py LunarLander`<br>
 
**Enjoy + Gif**<br>
`python enjoy_gif.py LunarLander`<br>
![](./assets/lunarlander.gif)

### Atari
Env Name listed in Env_Name.txt<br>

`python train_atari.py Pong 10000000`<br>
`python enjoy_atari.py Pong`<br>

## [mario](https://github.com/yuansongFeng/MadMario/)
[Tutorial: Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br>
`pip install gym-super-mario-bros`<br>
`cd mario`

*Training time is around 80 hours on CPU and 20 hours on GPU.*<br>
To train : (epochs=40000)<br>
`python main.py`

To replay: ("trained_mario.chkpt")<br>
`python replay.py`

![](https://pytorch.org/tutorials/_images/mario.gif)

## stock
`cd stock`<br>
`python stock_dqn.py GOOGL`<br>
`python play_stock.py GOOGL`<br>
