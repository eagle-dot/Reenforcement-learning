import numpy as np

# ── Environment ───────────────────────────────────────────────────────────────
# A minimal 3-state grid world.  Actions: 0 = Move Left, 1 = Move Right.
num_states  = 3
num_actions = 2

# ── Q-Table ───────────────────────────────────────────────────────────────────
# Rows = states, columns = actions.
# Values are arbitrary initial guesses — in real training these start at 0
# or random and improve through repeated experience.
q_table = np.array([
    [2.0, 5.0],   # State 0: [Left, Right]
    [1.5, 3.0],   # State 1: [Left, Right]
    [8.0, 4.0]    # State 2: [Left, Right]
])

print("--- Initial Q-Table ---")
print(q_table)
print("-" * 30)

# ── Hyperparameters ───────────────────────────────────────────────────────────
alpha = 0.5   # Learning rate: how far we move toward the new estimate (0=ignore, 1=replace)
gamma = 0.9   # Discount factor: how much we value future rewards vs. immediate ones

# ── Experience sample ─────────────────────────────────────────────────────────
# One (state, action, reward, next_state) tuple — the atom of Q-learning.
current_state   = 1      # agent starts here
action_taken    = 1      # chose Move Right
immediate_reward = 10.0  # reward the environment gave back
next_state      = 2      # state the agent landed in

# ── Bellman update (2-step process) ──────────────────────────────────────────

# STEP A: Compute the Bellman target — what the Q-value *should* be.
# Pure arithmetic using real experience; no learning happens yet.
# Combines immediate reward with discounted best future value we currently know.
old_q_value  = q_table[current_state, action_taken]    # our current guess
max_future_q = np.max(q_table[next_state])             # best option from next state
bellman_target = immediate_reward + (gamma * max_future_q)

# STEP B: Update the Q-table — nudge the old value toward that target.
# We never jump straight to the target because the target is itself an estimate
# (max_future_q is also a guess). Alpha blending keeps learning stable.
td_error    = bellman_target - old_q_value             # how wrong the old guess was
new_q_value = old_q_value + (alpha * td_error)         # move alpha% of the way there
q_table[current_state, action_taken] = new_q_value

# ── Diagnostics ───────────────────────────────────────────────────────────────
print(f"Agent's current state: {current_state}")
print(f"Action taken: {action_taken} (Old Q-value guess was: {old_q_value})")
print(f"Received immediate reward: {immediate_reward}")
print(f"Transitioned to next state: {next_state}")
print(f"Best available Q-value guess in the next state: {max_future_q}\n")

print(f"-> Calculated Bellman Target: {bellman_target:.2f}  (Formula: {immediate_reward} + {gamma} * {max_future_q})")
print(f"-> Temporal Difference Error: {td_error:.2f}  (Formula: {bellman_target:.2f} - {old_q_value})")
print(f"-> Nudging Old Q-value by alpha ({alpha}) * TD Error: {new_q_value:.2f}\n")

print("--- Updated Q-Table (after Step 1) ---")
print(q_table)
print("Note how only the cell at index [1, 1] changed!")

# ── Step 2: same action from the new state ────────────────────────────────────
# Agent is now in state 2, takes action 1 (Move Right) again.
# The environment returns reward 0 and keeps the agent in state 2.
print("\n" + "=" * 30)
print("STEP 2")
print("=" * 30)

current_state    = 2      # arrived here after step 1
action_taken     = 1      # Move Right again
immediate_reward = 0    # no reward this step (environment decision)
next_state       = 2      # stays in state 2 (environment decision)

old_q_value  = q_table[current_state, action_taken]
max_future_q = np.max(q_table[next_state])

bellman_target = immediate_reward + (gamma * max_future_q)
td_error       = bellman_target - old_q_value
new_q_value    = old_q_value + (alpha * td_error)
q_table[current_state, action_taken] = new_q_value

print(f"Agent's current state: {current_state}")
print(f"Action taken: {action_taken} (Old Q-value guess was: {old_q_value})")
print(f"Received immediate reward: {immediate_reward}")
print(f"Transitioned to next state: {next_state}")
print(f"Best available Q-value guess in the next state: {max_future_q}\n")

print(f"-> Calculated Bellman Target: {bellman_target:.2f}  (Formula: {immediate_reward} + {gamma} * {max_future_q})")
print(f"-> Temporal Difference Error: {td_error:.2f}  (Formula: {bellman_target:.2f} - {old_q_value})")
print(f"-> Nudging Old Q-value by alpha ({alpha}) * TD Error: {new_q_value:.2f}\n")

print("--- Updated Q-Table (after Step 2) ---")
print(q_table)
print("Note how only the cell at index [2, 1] changed!")
