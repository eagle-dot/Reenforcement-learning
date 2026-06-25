import random

class BlackjackEnv:
    def __init__(self):
        # Actions: 0 = Stick (Stop taking cards), 1 = Hit (Take another card)
        self.action_space = [0, 1]
        self.reset()

    def _draw_card(self):
        """Draws a card with replacement. Face cards count as 10, Ace as 11."""
        card = random.randint(1, 13)
        if card > 10:
            return 10
        elif card == 1:
            return 11
        return card

    def _score(self, hand):
        """Calculates the total score of a hand, handling usable aces."""
        total = sum(hand)
        usable_ace = False
        
        # If we bust but have an Ace counting as 11, convert it to a 1
        while total > 21 and 11 in hand:
            hand[hand.index(11)] = 1
            total = sum(hand)
            
        if 11 in hand:
            usable_ace = True
            
        return total, usable_ace

    def reset(self):
        """Resets the game and deals the initial hands."""
        # Deal 2 cards to player, 2 cards to dealer
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]

        # Force hands to be under 21 at the start (standard RL benchmark rule)
        while self._score(self.player_hand)[0] < 12:
            self.player_hand.append(self._draw_card())

        player_sum, usable_ace = self._score(self.player_hand)
        
        # The agent only sees the dealer's first card
        dealer_showing = self.dealer_hand[0] 
        
        return (player_sum, dealer_showing, usable_ace)

 
    """
    return (player_sum, dealer_showing, usable_ace), reward, true/false
    The state is a 3-tuple: (player_sum, dealer_showing, usable_ace)
    Each dimension is independent — any combination is a valid state. So you count every possible combination by multiplying.
    total states = (# player_sum values) × (# dealer_showing values) × (# usable_ace values)
             =        20             ×           10               ×         2
             = 400
             
    """
    def step(self, action):
        """Executes one step in the environment based on the agent's action."""
        if action == 1:  # HIT
            self.player_hand.append(self._draw_card())
            player_sum, usable_ace = self._score(self.player_hand)
            
            if player_sum > 21:  # Player busts
                return (player_sum, self.dealer_hand[0], usable_ace), -1.0, True
            else:
                return (player_sum, self.dealer_hand[0], usable_ace), 0.0, False

        else:  # STICK (Dealer's turn)
            player_sum, _ = self._score(self.player_hand)
            dealer_sum, _ = self._score(self.dealer_hand)

            # Dealer hits until reaching 17 or higher
            while dealer_sum < 17:
                self.dealer_hand.append(self._draw_card())
                dealer_sum, _ = self._score(self.dealer_hand)

            # Determine the reward mapping
            if dealer_sum > 21: # Dealer busts
                reward = 1.0
            elif player_sum > dealer_sum:
                reward = 1.0
            elif player_sum < dealer_sum:
                reward = -1.0
            else:
                reward = 0.0  # Draw / Push

            # Terminal step representation
            final_player_sum, final_usable_ace = self._score(self.player_hand)
            return (final_player_sum, self.dealer_hand[0], final_usable_ace), reward, True

# --- Simulation Execution ---
if __name__ == "__main__":
    env = BlackjackEnv()
    
    print("--- Executing a Sample Game Interaction ---")
    state = env.reset()
    done = False
    
    print(f"Initial State -> Player Sum: {state[0]}, Dealer Shows: {state[1]}, Usable Ace: {state[2]}")
    print(f"Player Hand Details: {env.player_hand}")
    
    while not done:
        # Simple policy: Hit if below 17, otherwise stick
        action = 1 if state[0] < 17 else 0
        action_name = "HIT" if action == 1 else "STICK"
        
        state, reward, done = env.step(action)
        print(f"\nAction Chosen: {action_name}")
        print(f"New State -> Player Sum: {state[0]}, Dealer Shows: {state[1]}, Usable Ace: {state[2]}")
        
    print("\n--- Game Over ---")
    print(f"Final Player Hand: {env.player_hand} (Score: {sum(env.player_hand)})")
    print(f"Final Dealer Hand: {env.dealer_hand} (Score: {sum(env.dealer_hand)})")
    print(f"Resulting Reward: {reward}")