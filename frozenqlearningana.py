# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 14:41:36 2024
#https://www.dideo.tv/v/yt/V65phXUGb4I/code-frozen-game-using-reinforcement-learning-%7C-openai-gym-%7C-python-project
@author: DigiNarin
"""

import gym
import numpy as np
import random

env = gym.make('FrozenLake-v1',is_slippery = False )
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

qtable = np.zeros((state_space_size, action_space_size))
print(qtable)

total_episodes = 10000        # Total episodes
learning_rate = 0.2           # Learning rate
max_steps = 100                # Max steps per episode
gamma = 0.99                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001             # Exponential decay rate for exploration prob




rewards = []

for episode in range(total_episodes):
    state = env.reset()[0]
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        
         if random.uniform(0,1) > epsilon:
              action = np.argmax(qtable[state,:])
         else:
              action = env.action_space.sample()
             
            
      
        # Take the action (index) that have the maximum expected future reward given that state
         
        
         obs, reward, terminated, truncated , info = env.step(action)
         new_state = obs
         max_new_state = np.max(qtable[new_state,:])
         print("state", state)
         print("action", action)
         print("new_state", new_state)
         print("qtable", qtable)
         qtable[state,action] = qtable[state,action] + learning_rate*(reward+gamma*max_new_state-qtable[state,action])
         total_rewards += reward

         state = new_state
         if terminated:
               break
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_rewards)
    print("score:"  ,str(sum(rewards)/total_episodes))
    print(qtable)

