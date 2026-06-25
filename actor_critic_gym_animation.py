import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# ── Network ───────────────────────────────────────────────────────────────────
# One network, two independent heads (no shared layers).
# Actor  → probability distribution over actions (what to do)
# Critic → single scalar estimating how good the current state is V(s)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # state(4) → 128 → ReLU → action_dim → Softmax
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # state(4) → 128 → ReLU → 1  (no activation: raw value estimate)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        # Run both heads in one call; caller decides which output(s) to use
        return self.actor(state), self.critic(state)

def main():
    # ── Setup ─────────────────────────────────────────────────────────────────
    train_env = gym.make('CartPole-v1')
    # Read dims from env so network sizes are never hardcoded (state_dim=4, action_dim=2)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    gamma = 0.99    # discount factor: future rewards worth 99% of immediate ones

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Training the agent with batch trajectory smoothing...")
    for episode in range(600):
        state, _ = train_env.reset()
        done = False
        episode_reward = 0

        # Buffers to hold an entire episode's rollouts for a single, stable gradient update
        log_probs = []
        values    = []
        rewards   = []
        entropies = []

        # ── Rollout: play one full episode ────────────────────────────────────
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs, value = model(state_tensor)
            dist   = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = train_env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))  # log P(action | state)
            values.append(value)                     # critic's V(s) estimate
            rewards.append(reward)                   # actual reward from env
            entropies.append(dist.entropy())         # randomness of the distribution

            episode_reward += reward
            state = next_state

        # ── Compute discounted returns (backwards pass) ────────────────────────
        # G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
        # Walking backwards and inserting at 0 keeps results in forward order
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns    = torch.FloatTensor(returns)
        values     = torch.cat(values).squeeze()
        log_probs  = torch.stack(log_probs)
        entropies  = torch.stack(entropies)

        # ── Advantage: how much better/worse the action was vs. expectation ───
        # Positive → action exceeded the critic's expectation; negative → fell short
        advantages = returns - values

        # ── Loss terms ────────────────────────────────────────────────────────
        # Actor: push up log-prob of high-advantage actions; entropy bonus discourages
        #        premature certainty and keeps exploration alive
        actor_loss  = (-log_probs * advantages.detach()).mean() - 0.01 * entropies.mean()
        # Critic: reduce the gap between predicted V(s) and actual return G
        critic_loss = F.mse_loss(values, returns)
        # 0.5 weight stops the critic loss from dominating the actor updates
        total_loss  = actor_loss + 0.5 * critic_loss

        # ── Gradient update ───────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()

        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Total Reward: {episode_reward:.1f} steps")

        # Early stop: ≥470 steps means the pole stayed up for almost the full 500-step limit
        if episode_reward >= 470:
            print(f"\n--> Success! Solved at episode {episode} ({episode_reward} steps). Launching Animation!")
            break

    train_env.close()

    # ── Animation ─────────────────────────────────────────────────────────────
    # Second env with render_mode='human' opens a visual window.
    # Critic output is discarded (_) — only the actor's action probs are needed here.
    print("\nOpening visual window...")
    visual_env = gym.make('CartPole-v1', render_mode='human')
    for test_episode in range(50):
        state, _ = visual_env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():   # inference only — no gradient tracking needed
                probs, _ = model(state_tensor)
            dist   = Categorical(probs)
            action = dist.sample().item()
            state, reward, terminated, truncated, _ = visual_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        print(f"Watch Episode {test_episode+1} | Lasted for {episode_reward} steps!")
    visual_env.close()

if __name__ == "__main__":
    main()
