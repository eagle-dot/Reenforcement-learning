import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


# This code uses standard Gymnasium (the modern successor to OpenAI Gym) to solve the classic CartPole-v1 environment, where an agent learns to balance a pole on a moving cart.

# =====================================================================
# 1. ARCHITECTURE DEFINITION
# =====================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor Network: Outputs a probability distribution over actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic Network: Outputs a single scalar value V(s) estimating expected future rewards
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        state_value = self.critic(state)
        return probs, state_value

# =====================================================================
# 2. TRAINING LOOP
# =====================================================================
def main():
    # Initialize Environment (CartPole-v1)
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Instantiate Model and Optimizer
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    gamma = 0.99
    max_episodes = 1000
    
    print(f"Starting training on CartPole-v1... Target score is 500.")
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert environment state to PyTorch tensor
            state_tensor = torch.FloatTensor(state)
            
            # 1. Forward Pass: Get Action Probabilities and State Value Estimation
            probs, value = model(state_tensor)
            
            # Sample an action from the policy distribution
            dist = Categorical(probs)
            action = dist.sample()
            
            # Step the environment using the chosen action
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
            
            # Convert next state to tensor
            next_state_tensor = torch.FloatTensor(next_state)
            
            # 2. Get next state value estimation from Critic (V(s_t+1))
            _, next_value = model(next_state_tensor)
            
            # 3. Calculate TD Target and Advantage (The Feedback)
            # If the episode is over, there is no future value (next_value = 0)
            td_target = reward + (gamma * next_value * (1 - int(done)))
            advantage = td_target - value
            
            # 4. Calculate Actor Loss
            log_prob = dist.log_prob(action)
            # We use detach() on advantage here because we only want to update the Actor's 
            # weights based on this value, without backpropagating Actor gradients into the Critic.
            actor_loss = -log_prob * advantage.detach()
            
            # 5. Calculate Critic Loss (Mean Squared Error between prediction and actual outcome)
            critic_loss = F.mse_loss(value, td_target.detach())
            
            # 6. Combined Backpropagation
            # The 0.5 scales down the critic updates so it doesn't overpower the policy changes
            total_loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Move to the next state
            state = next_state
            
        # Log progress every 20 episodes
        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Total Reward: {episode_reward:5.1f}")
            
        # CartPole-v1 maximizes at 500 steps. If we steadily hit 500, we consider it solved.
        if episode_reward >= 500:
            print(f"\nEnvironment Solved in {episode} episodes! Final reward: {episode_reward}")
            break
            
    env.close()

if __name__ == "__main__":
    main()