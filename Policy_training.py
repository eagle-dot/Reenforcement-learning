import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Simple two-layer neural network
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Output probabilities over actions using Softmax
        return F.softmax(self.fc2(x), dim=-1)

def train_one_episode(policy, optimizer, env, gamma=0.99):
    saved_log_probs = []
    rewards = []
    
    state, _ = env.reset()
    done = False
    
    # 1. Roll out a full episode (Collect Trajectory)
    while not done:
        state_tensor = torch.from_numpy(state).float()
        action_probs = policy(state_tensor)
        
        # Create a categorical distribution to sample from
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Save the log probability: log pi(a_t | s_t)
        saved_log_probs.append(dist.log_prob(action))
        
        # Take a step in the environment
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)
        
    # 2. Compute the Rewards-to-go (G_t)
    returns = []
    G = 0
    # Loop backwards to efficiently calculate G_t via causality
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
        
    # Standardize returns to stabilize variance
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # 3. Compute the Policy Gradient Objective Loss
    policy_loss = []
    for log_prob, G_t in zip(saved_log_probs, returns):
        # We put a minus sign because PyTorch optimizers minimize loss, 
        # but we want to maximize the expected return (Gradient Ascent)
        policy_loss.append(-log_prob * G_t)
        
    # Sum up all the token step updates for the episode trajectory
    policy_loss = torch.stack(policy_loss).sum()
    
    # 4. Perform Backpropagation weight update
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    return sum(rewards)