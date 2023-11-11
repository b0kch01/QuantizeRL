#!/usr/bin/env python
# coding: utf-8

# # PPO Base Implementation
# This will be the baseline implementation for comparing with the other methods.

# In[1]:


SEED = 1234
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPOCHS = 10
CLIP_EPSILON = 0.2
BATCH_SIZE = 5


# In[2]:


import numpy as np
import random

import torch
from torch.nn import Linear, Dropout, LeakyReLU, Softmax, Sequential, Sigmoid, ReLU
from torch.optim import Adam
from torch.nn import MSELoss

import gym

from collections import deque


# In[3]:


env = gym.make('CartPole-v1')
device = torch.device("mps")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


# ## Network Architecture

# **PolicyNetwork**:
# - Input: State
# - Output: Action distribution (0-1)
# - 2 Hidden layers with LeakyReLU activation
# 
# **ValueNetwork**:
# - Input: State
# - Output: Value
# - 2 Hidden layers with LeakyReLU activation

# In[4]:


class PolicyNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(PolicyNetwork, self).__init__()
    self.model = Sequential(
      Linear(input_dim, hidden_dim),
      ReLU(),
      Linear(hidden_dim, hidden_dim),
      ReLU(),
      Linear(hidden_dim, 2),
      Softmax(dim=1) # using this so my code can work with other environments
    )

  def forward(self, state):
    return self.model(state)
 
 
  def stochastic_action(self, state):
    r"""Returns an action sampled from the policy network."""
    
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    probs = self.forward(state).cpu()
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)
  
  def deterministic_action(self, state):
    r"""Returns an action with the highest probability."""
    
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    probs = self.forward(state).cpu()
    action = torch.argmax(probs)
    return action.item(), probs[0][action].item()

  
class ValueNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim) -> None:
    super(ValueNetwork, self).__init__()
    self.model = Sequential(
      Linear(input_dim, hidden_dim),
      ReLU(),
      Linear(hidden_dim, hidden_dim),
      ReLU(),
      Linear(hidden_dim, 1)
    )
  
  def forward(self, state):
    return self.model(state)
  


# In[5]:


_observation_size = env.observation_space.shape[0]

policy_net = PolicyNetwork(_observation_size, 64).to(device)
value_net  = ValueNetwork(_observation_size, 64).to(device)

policy_optimizer = Adam(policy_net.parameters(), lr=LEARNING_RATE)
value_optimizer  = Adam(value_net.parameters(), lr=LEARNING_RATE)
criterion = MSELoss()


# In[6]:


def compute_returns(rewards):
  returns = [0]*len(rewards)
  R = 0
  for i in reversed(range(len(rewards))):
    R = rewards[i] + GAMMA * R
    returns[i] = R
  return torch.tensor(returns)


# In[20]:


def ppo_step():
  state, _ = env.reset()
  done = False
  states, actions, log_probs_old, rewards = [], [], [], []

  # capture entire episode
  while not done:
    action, log_prob = policy_net.stochastic_action(state)
    next_state, reward, done, *_ = env.step(action)
    
    log_probs_old.append(log_prob)
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    
    state = next_state
  
  # Calculate returns and value estimation
  returns = compute_returns(rewards).to(device)
  
  values = value_net(torch.from_numpy(np.array(states)).to(device))
  advantages = returns - values.squeeze()

  states = torch.from_numpy(np.array(states)).detach().to(device)
  actions = torch.tensor(actions).detach().to(device)
  log_probs_old = torch.stack(log_probs_old).detach().to(device)
  advantages = advantages.detach().to(device)
  returns = returns.detach().to(device)

  for _ in range(EPOCHS):
      for i in range(0, len(states), BATCH_SIZE):
          # Grab a batch of data
          batch_states = states[i:i+BATCH_SIZE]
          batch_actions = actions[i:i+BATCH_SIZE]
          batch_log_probs_old = log_probs_old[i:i+BATCH_SIZE]
          batch_advantages = advantages[i:i+BATCH_SIZE]
          batch_returns = returns[i:i+BATCH_SIZE]
          
          # Calculate new log probabilities
          new_action_probs = policy_net(batch_states)
          new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)))
          
          # rho is the ratio between new and old log probabilities
          ratio = (new_log_probs - batch_log_probs_old).exp()
          
          policy_loss = -torch.min(
            # rho * A
            ratio * batch_advantages,
            # Clipped rho * A
            torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * batch_advantages 
          ).mean()

          policy_optimizer.zero_grad()
          policy_loss.backward()
          policy_optimizer.step()
          
          value_loss = criterion(value_net(batch_states),
                                 batch_returns.unsqueeze(-1))

          value_optimizer.zero_grad()
          value_loss.backward()
          value_optimizer.step()

  print(f"Rewards: {sum(rewards)}")
      
  


# In[21]:


for i in range(1):
  print(f"Episode {i}", end=" ")
  ppo_step()


# In[9]:


def evaluate():
  # evaluate and collect rewards
  rewards = []
  
  for _ in range(10):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
      action, _ = policy_net.deterministic_action(state)
      state, reward, done, *_ = env.step(action)
      total_reward += reward
    rewards.append(total_reward)
    
  print(f"Average reward: {sum(rewards)/len(rewards)}")
    


# In[10]:


evaluate()


# In[ ]:




