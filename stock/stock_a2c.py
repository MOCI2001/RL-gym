### Reinforcement Learning Stock Trading using A2C
import sys
import time
import math
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam

stock_name = sys.argv[1] # 'GOOGL', 'MSFT', 'AAPL'
model_name = stock_name+'_A2C'
window_size = 50 # working days

## Create Agent
class A2CAgent:
    def __init__(self, state_size, is_loadmodel=False, model_name=""):
        self.state_size = state_size 
        self.action_size = 3 # sit, buy, sell
        self.value_size = 1
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.load_model = is_loadmodel
        ## Policay Gradient hyperparameters
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.actor  = self.build_actor()
        self.critic = self.build_critic()
        if self.load_model:
            self.actor.load_weights(model_name+"_actor.h5")
            self.critic.load_weights(model_name+"_critic.h5")
    def build_actor(self):
        actor = Sequential()
        actor.add(Input(shape=(self.state_size,1)))
        actor.add(LSTM(units=50,return_sequences=True))
        actor.add(LSTM(units=50))        
        #actor.add(Dense(units=64, input_dim=self.state_size, activation="relu", kernel_initializer="he_uniform"))
        #actor.add(Dense(units=32, activation="relu", kernel_initializer="he_uniform"))
        #actor.add(Dense(units=8,  activation="relu", kernel_initializer="he_uniform"))
        actor.add(Dense(self.action_size, activation="softmax"))
        actor.summary()
        actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr= self.actor_lr))
        return actor 
    def build_critic(self): 
        critic = Sequential()
        critic.add(Input(shape=(self.state_size,1)))
        critic.add(LSTM(units=50,return_sequences=True))
        critic.add(LSTM(units=50))        
        #critic.add(Dense(units=64, input_dim=self.state_size, activation="relu", kernel_initializer="he_uniform"))
        #critic.add(Dense(units=32, activation="relu", kernel_initializer="he_uniform"))
        #critic.add(Dense(units=8, activation="relu", kernel_initializer="he_uniform"))
        critic.add(Dense(self.value_size, activation="linear", kernel_initializer="he_uniform"))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr= self.critic_lr))
        return critic
    def get_action(self, state):
        state = state.reshape(-1,self.state_size,1)
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1,self.value_size)) # Initialize the policy targets matrix
        advantages = np.zeros((1,self.action_size)) # Initialize the advantages matrix
        state = state.reshape(-1,self.state_size,1)
        value = self.critic.predict(state)[0] # Get value for this state
        next_state = next_state.reshape(-1,self.state_size,1)
        next_value = self.critic.predict(next_state)[0] # Get value for the next state

        # update the advantages and value tables if done
        if done:
            advantages[0][action] = reward - value # Basically, what do we gain by choosing the action, will it improve or worsen the advantage
            target[0][0] = reward # Fill in the target value to see if we can still improve it in the policy making
        else:
            advantages[0][action] = reward + self.discount_factor*(next_value) - value # If not yet done, then simply update for the current step.
            target[0][0] = reward + self.discount_factor*next_value
        # Once we are done with the episode, we then update the weights
        self.actor.fit(state,advantages,epochs=1,verbose=0)
        self.critic.fit(state,target,epochs=1,verbose=0)

## Basic Functions
def formatPrice(n):
    return("-Rs." if n<0 else "Rs.")+"{0:.2f}".format(abs(n))

def getStockDataVec(key):
    vec = []
    dat = []
    lines = open(key+".csv","r").read().splitlines()
    for line in lines[1:]: # open, hight, low, close, volume
        vec.append(float(line.split(",")[4])) 
        dat.append(line.split(",")[0])
    return vec, dat 

def sigmoid(x):
    return 1/(1+math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

## Train Agent

agent = A2CAgent(window_size)
data, date  = getStockDataVec(stock_name)
l = len(data) -1
print(stock_name, l)

data = data[:52*5] # reduce data to latest 52 weeks (5 working days per week)
date = date[:52*5] 
l = len(data) - 1
print(l)

data.reverse() # first data is from the latest date, so reverse it
date.reverse()

batch_size = 32
num_episode= 10

start_t = time.time()
for e in range(num_episode):
    print("Episode " + str(e+1) + "/" + str(num_episode))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        action = agent.get_action(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1: # buy
            agent.inventory.append(data[t])
            print(date[t]+" Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print(date[t]+" Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.train_model(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
    if e % 10 == 0:
        agent.actor.save_weights(model_name+'_actor.h5')
        agent.critic.save_weights(model_name+'_critic.h5')
    agent.actor.save_weights(model_name+'_actor.h5')
    agent.critic.save_weights(model_name+'_critic.h5')

end_t = time.time()
print("Excecution Time = ", end_t - start_t)
