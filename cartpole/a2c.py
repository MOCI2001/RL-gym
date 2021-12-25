### Reinforcement Learning Stock Trading using A2C
import math
import numpy as np
import random
from collections import deque
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class A2CAgent:
    def __init__(self, state_size, action_size, is_loadmodel=False):
        self.state_size = state_size 
        self.action_size = action_size
        self.value_size = 1
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.load_model = is_loadmodel
        ## Policay Gradient hyperparameters
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.actor  = self.build_actor()
        self.critic = self.build_critic()
        if self.load_model:
            self.actor.load_weights("a2c_actor.h5")
            self.critic.load_weights("a2c_critic.h5")
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(units=64, input_dim=self.state_size, activation="relu", kernel_initializer="he_uniform"))
        actor.add(Dense(units=32, activation="relu", kernel_initializer="he_uniform"))
        actor.add(Dense(units=8,  activation="relu", kernel_initializer="he_uniform"))
        actor.add(Dense(self.action_size, activation="softmax"))
        actor.summary()
        actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr= self.actor_lr))
        return actor 
    def build_critic(self): 
        critic = Sequential()
        critic.add(Dense(units=64, input_dim=self.state_size, activation="relu", kernel_initializer="he_uniform"))
        critic.add(Dense(units=32, activation="relu", kernel_initializer="he_uniform"))
        critic.add(Dense(units=8, activation="relu", kernel_initializer="he_uniform"))
        critic.add(Dense(self.value_size, activation="linear", kernel_initializer="he_uniform"))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr= self.critic_lr))
        return critic
    def get_action(self, state):
        state = state.reshape(-1, self.state_size)
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1,self.value_size)) # Initialize the policy targets matrix
        advantages = np.zeros((1,self.action_size)) # Initialize the advantages matrix
        state = state.reshape(-1, self.state_size)
        value = self.critic.predict(state)[0] # Get value for this state
        next_state = next_state.reshape(-1,self.state_size)
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

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size= env.action_space.n
    agent = A2CAgent(state_size, action_size)
    done = False
    batch_size = 32

    EPISODES = 1000

    for e in range(EPISODES):
        state = env.reset()
        for t in range(500):
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            agent.train_model(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}"
                        .format(e, EPISODES, t))
                break
        if e % 10 == 0:
            agent.actor.save_weights('a2c_actor.h5')
            agent.critic.save_weights('a2c_critic.h5')
        agent.actor.save_weights('a2c_actor.h5')
        agent.critic.save_weights('a2c_critic.h5')

