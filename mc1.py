import random
import numpy as np

class BlackjackSimulation:
    def __init__(self):
        # Hyperparameters from the video screen
        self.epsilon = 0.1
        self.alpha = 1 / 5000
        self.gamma = 1.0  # Standard for Blackjack (no discounting)
        
        # Q-Table Dimensions:
        # Player Sum (12 to 21 -> 10 states)
        # Dealer Showing (1 to 10 -> 10 states)
        # Usable Ace (0 or 1 -> 2 states)
        # Actions (0 = Stick, 1 = Hit -> 2 actions)
        self.Q = np.zeros((10, 10, 2, 2)) 
        
    def _draw_card(self):
        card = random.randint(1, 13)
        return 10 if card > 10 else (11 if card == 1 else card)

    def _score(self, hand):
        total = sum(hand)
        while total > 21 and 11 in hand:
            hand[hand.index(11)] = 1
            total = sum(hand)
        return total, (11 in hand)

    def get_state_indices(self, player_sum, dealer_showing, usable_ace):
        """Maps raw game values to 0-indexed Q-table coordinates."""
        return (player_sum - 12, dealer_showing - 1, int(usable_ace))

    def get_action(self, state_idx):
        """Implements the epsilon-soft policy rule from the video."""
        if random.random() < self.epsilon:
            return random.randint(0, 1) # Explore
        return np.argmax(self.Q[state_idx]) # Exploit

    def play_episode(self):
        """Simulates one complete episode under the current policy."""
        # Initialize hands
        player_hand = [self._draw_card(), self._draw_card()]
        dealer_hand = [self._draw_card(), self._draw_card()]

        while self._score(player_hand)[0] < 12:
            player_hand.append(self._draw_card())

        episode_history = []
        done = False

        # --- Player's Turn ---
        while not done:
            p_sum, u_ace = self._score(player_hand)
            d_show = dealer_hand[0] if dealer_hand[0] <= 10 else 1  # Ace drawn as 11 -> map back to 1
            
            state_idx = self.get_state_indices(p_sum, d_show, u_ace)
            action = self.get_action(state_idx)
            
            # Record state-action pair visited
            episode_history.append((state_idx, action))

            if action == 1:  # HIT
                player_hand.append(self._draw_card())
                if self._score(player_hand)[0] > 21:
                    return episode_history, -1.0 # Player busted immediately
            else:
                done = True

        # --- Dealer's Turn ---
        d_sum, _ = self._score(dealer_hand)
        while d_sum < 17:
            dealer_hand.append(self._draw_card())
            d_sum, _ = self._score(dealer_hand)

        # Final Evaluation
        p_sum, _ = self._score(player_hand)
        if d_sum > 21 or p_sum > d_sum:
            return episode_history, 1.0
        elif p_sum < d_sum:
            return episode_history, -1.0
        return episode_history, 0.0

    def train(self, num_episodes=500000):
        """Runs the main Constant-alpha MC algorithm loop."""
        print(f"Starting simulation of {num_episodes} episodes...")
        
        for ep in range(1, num_episodes + 1):
            # 1. Generate an episode following the policy
            history, final_reward = self.play_episode()
            
            # 2. Constant-alpha updates for each step in the episode
            # (First-visit tracking per episode to match video theory)
            visited_step = set()
            for state_idx, action in history:
                if (state_idx, action) not in visited_step:
                    visited_step.add((state_idx, action))
                    
                    # Target G_m is simply final_reward since gamma = 1
                    old_Q = self.Q[state_idx][action]
                    
                    # Update rule shown on video slide: Q <- Q + alpha * (G - Q)
                    self.Q[state_idx][action] = old_Q + self.alpha * (final_reward - old_Q)
            
            if ep % 100000 == 0:
                print(f"Completed {ep}/{num_episodes} episodes...")

    def print_policy(self):
        """Prints a visual map of the resulting learned policy and expected rewards."""
        for ace_type, title in [(1, "USEABLE ACE"), (0, "NO USEABLE ACE")]:
            print(f"\n=== {title} ===")

            # --- Action grid (H / S) ---
            print("\n  Policy (H=Hit, S=Stick)")
            print("  P_Sum\\Dealer ", " ".join(f"{i:2d}" for i in range(1, 11)))
            for p_idx in reversed(range(10)):
                row_str = []
                for d_idx in range(10):
                    best_action = np.argmax(self.Q[p_idx, d_idx, ace_type])
                    row_str.append(" H" if best_action == 1 else " S")
                print(f"    {p_idx + 12:2d}      ", "".join(row_str))

            # --- Reward grid (max Q-value per cell) ---
            print("\n  Expected Reward (max Q-value)")
            print("  P_Sum\\Dealer ", "  ".join(f"{i:5d}" for i in range(1, 11)))
            for p_idx in reversed(range(10)):
                row_str = []
                for d_idx in range(10):
                    best_q = np.max(self.Q[p_idx, d_idx, ace_type])
                    row_str.append(f"{best_q:+.2f}")
                print(f"    {p_idx + 12:2d}      ", "  ".join(row_str))

# --- Run the Simulation ---
if __name__ == "__main__":
    sim = BlackjackSimulation()
    # Run 500,000 episodes to see the tables converge
    sim.train(num_episodes=500000)
    sim.print_policy()