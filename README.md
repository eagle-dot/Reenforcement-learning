# Reinforcement Learning — Study Examples

A progression of RL concepts implemented from scratch, starting with tabular methods and building up to deep Actor-Critic on OpenAI Gymnasium.

---

## Files

### `Bellman.py` — Q-Learning: Single Bellman Update
The most basic building block of Q-learning. A hardcoded 3-state grid world demonstrates one complete Bellman update step by step.

Step 1 — Compute the target (what the Q-value should be):
    bellman_target = immediate_reward + (gamma * max_future_q)


Step 2 — Nudge the Q-table toward that target (the actual update):

    td_error    = bellman_target - old_q_value   # how wrong we were
    new_q_value = old_q_value + (alpha * td_error)


alpha controls how big a step we take. If alpha = 1, we replace the old value entirely


**Concepts:** Q-table, Bellman equation, TD error, learning rate (α), discount factor (γ)

---

### `blackJack.py` — Q-Learning Environment: Blackjack
A hand-coded Blackjack environment (no external libraries). Runs a single game using a simple threshold policy: hit if below 17, stick otherwise.

**Concepts:** Environment/agent interface, state representation, reward signals, episode termination

NO training!!!!

---

### `mc1.py` — Monte Carlo Control: Blackjack
Trains a Q-table on Blackjack using constant-alpha Monte Carlo over 500,000 episodes. Prints the learned policy (Hit/Stick) and expected reward per state as two grids.

**Concepts:** Monte Carlo control, first-visit updates, ε-soft exploration, Q-table convergence

**State space:** Player sum (12–21) × Dealer showing (1–10) × Usable ace (yes/no)  
**Output:** Two grids per ace type — policy (H/S) and expected reward (max Q-value)

> Player sums start at 12 because hitting on 11 or below can never bust — there is no decision to learn below that threshold.

---

### `Policy_training.py` — REINFORCE (Vanilla Policy Gradient)
Implements the REINFORCE algorithm with a simple two-layer neural network. Trains on CartPole-v1 without a critic — the full Monte Carlo return G is used directly as the learning signal.

**Concepts:** Policy gradient, log-probability, rewards-to-go, return standardisation, gradient ascent

**Network:** `state(4) → 128 → ReLU → 2 → Softmax`

---

### `actor_critic_gym.py` — Actor-Critic (TD / Online)
Adds a Critic network to estimate V(s) at every step. Uses single-step TD targets instead of full Monte Carlo returns — updates happen after every action, not at episode end.

**Concepts:** Actor-Critic, TD target, advantage estimate `A = TD_target − V(s)`, online updates

**Key difference from REINFORCE:** The critic provides a baseline at each timestep rather than waiting until the episode finishes.

---

### `actor_critic_gym_animation.py` — Actor-Critic with Animation
Batch trajectory version of Actor-Critic. Collects a full episode rollout before updating, then launches a visual window once CartPole is solved.

**Concepts:** Batch trajectory updates, entropy bonus, gradient clipping (`clip_grad_norm_`), Gymnasium `render_mode='human'`

**Key differences from `actor_critic_gym.py`:**

| Feature | `actor_critic_gym.py` | `actor_critic_gym_animation.py` |
|---------|----------------------|--------------------------------|
| Update style | Online (single-step TD) | Batch (full episode rollout) |
| Entropy bonus | No | Yes — keeps exploration alive |
| Gradient clipping | No | Yes — `clip_grad_norm_` max=1.0 |
| Animation | No | Yes — opens visual window on solve |

**Solve condition:** Episode reward ≥ 470 steps (out of 500 max)

---

## Concept Progression

```
Bellman.py                       →  manual Bellman update (no learning loop)
blackJack.py                     →  environment interface + single episode
mc1.py                           →  tabular Q-learning over 500k episodes
Policy_training.py               →  neural network policy, no value baseline (REINFORCE)
actor_critic_gym.py              →  neural policy + value baseline (TD, online)
actor_critic_gym_animation.py    →  batch rollout + entropy + gradient clip + animation
```

---

## Dependencies

```
pip install torch numpy gymnasium pygame
```

`pygame` is required by Gymnasium for the CartPole animation window (`render_mode='human'`).
