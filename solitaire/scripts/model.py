import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import random
from enum import Enum

# Load config
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class Suit(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Card:
    def __init__(self, suit, rank, face_up=False):
        self.suit = suit
        self.rank = rank
        self.face_up = face_up

    def color(self):
        # 0 (Red): Hearts, Diamonds
        # 1 (Black): Clubs, Spades
        return 0 if self.suit in [Suit.HEARTS.value, Suit.DIAMONDS.value] else 1

    def __repr__(self):
        suits = ['H', 'D', 'C', 'S']
        ranks = ['_', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return f"{ranks[self.rank]}{suits[self.suit]}" if self.face_up else "[?]"

class SolitaireEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.verbose = False  # Ustaw True by zobaczyć warnings o powtórzeniach
        
        # Config
        self.max_steps = config['environment']['max_steps']
        self.rewards = config['environment']['reward_scaling']
        
        # Observation Space
        # Tableau: 7 piles, max 20 cards. Features: [Present(0/1), Suit(0-3), Rank(1-13), FaceUp(0/1)]
        self.max_tableau_height = config['environment']['observation_space']['max_tableau_height']
        
        self.observation_space = spaces.Dict({
            'tableau': spaces.Box(low=0, high=1, shape=(7, self.max_tableau_height, 4), dtype=np.float32),
            'foundations': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32), # Top rank for each suit
            'waste': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32), # [Present, Suit, Rank]
            'stock': spaces.Box(low=0, high=1, shape=(52, 3), dtype=np.float32) # [Present, Suit, Rank] - Full visibility
        })
        
        # Action Space
        # 0: Draw Stock
        # 1-7: Waste -> Tableau 0-6
        # 8-11: Waste -> Foundation 0-3
        # 12-53: Tableau i -> Tableau j (7*6 = 42 permutations)
        # 54-81: Tableau i -> Foundation j (7*4 = 28 permutations)
        # 82-85: Stock -> Foundation 0-3 (Direct move)
        # 86: Surrender (Give up)
        self.action_space = spaces.Discrete(87)
        
        # State history for detecting loops
        self.state_history = []
        self.max_history = 50  # Pamiętaj ostatnie 50 stanów
        self.repeat_penalty = -50.0  # Kara za powtórzenie stanu
        
        self.current_action_mask = None
        self.reset()

    def action_masks(self):
        """
        Returns a boolean mask of valid actions for the current state.
        Uses cached mask if available to save computation.
        """
        if self.current_action_mask is None:
            self.current_action_mask = self._calculate_action_mask()
        return self.current_action_mask

    def _calculate_action_mask(self):
        """
        Internal method to calculate valid actions.
        """
        mask = np.zeros(87, dtype=bool)
        
        # 0: Draw Stock
        # Valid if stock is not empty OR (stock is empty AND waste is not empty)
        if self.stock or self.waste:
            mask[0] = True
            
        # 1-7: Waste -> Tableau 0-6
        if self.waste:
            card = self.waste[-1]
            # Symmetry breaking: if multiple empty columns, only allow moving King to the first one
            empty_cols = [c for c in range(7) if not self.tableau[c]]
            
            for i in range(7):
                if self._can_move_to_tableau(card, i):
                    # If it's a King and target is empty, ensure it's the first empty column
                    if card.rank == 13 and not self.tableau[i]:
                        if empty_cols and i != empty_cols[0]:
                            continue
                    mask[1 + i] = True
                    
        # 8-11: Waste -> Foundation 0-3
        if self.waste:
            card = self.waste[-1]
            for i in range(4):
                if self._can_move_to_foundation(card, i):
                    mask[8 + i] = True
                    
        # 12-53: Tableau i -> Tableau j
        # idx = action - 12; src = idx // 6; dst = idx % 6 (skip self)
        
        # Pre-calculate current state parts for simulation
        curr_tableau = [tuple((c.suit, c.rank, c.face_up) for c in pile) for pile in self.tableau]
        
        for action_idx in range(12, 54):
            idx = action_idx - 12
            src_idx = idx // 6
            dst_idx = idx % 6
            if dst_idx >= src_idx: dst_idx += 1
            
            if self.tableau[src_idx]:
                pile = self.tableau[src_idx]
                
                # Find first face up
                first_face_up = -1
                for k, c in enumerate(pile):
                    if c.face_up:
                        first_face_up = k
                        break
                
                if first_face_up != -1:
                    # Check if any card from first_face_up to top can start a chain on dst
                    # We need to find the EXACT move that 'step' would perform to simulate it correctly
                    # 'step' moves the DEEPEST valid substack (first found from bottom)
                    
                    move_card_idx = -1
                    for k in range(first_face_up, len(pile)):
                        card = pile[k]
                        if self._can_move_to_tableau(card, dst_idx):
                            move_card_idx = k
                            break
                    
                    if move_card_idx != -1:
                        # 🛑 HEURISTIC: Block moving a stack from the bottom of a pile to an empty pile.
                        # This is a useless move (just swapping empty columns or moving a King for no reason).
                        if move_card_idx == 0 and not self.tableau[dst_idx]:
                            continue  # Skip this action (leave it masked as False)

                        # Simulate the move to check for loops
                        # 1. Create copies of src and dst piles
                        src_tuple = curr_tableau[src_idx]
                        dst_tuple = curr_tableau[dst_idx]
                        
                        # Moving cards
                        moving_cards = src_tuple[move_card_idx:]
                        new_src = src_tuple[:move_card_idx]
                        new_dst = dst_tuple + moving_cards
                        
                        # Handle flip of new top card in src
                        if new_src:
                            # If the new top card was face down, it would be flipped face up
                            # We need to reflect this in the hash
                            last_card = new_src[-1]
                            if not last_card[2]: # if not face_up
                                new_src = new_src[:-1] + ((last_card[0], last_card[1], True),)
                        
                        # Construct new tableau tuple
                        new_tableau = list(curr_tableau)
                        new_tableau[src_idx] = new_src
                        new_tableau[dst_idx] = new_dst
                        new_tableau_tuple = tuple(new_tableau)
                        
                        # Check hash
                        sim_hash = self._get_state_hash(tableau=new_tableau_tuple)
                        
                        if sim_hash not in self.state_history:
                            mask[action_idx] = True

        # 54-81: Tableau i -> Foundation j
        for action_idx in range(54, 82):
            idx = action_idx - 54
            src_idx = idx // 4
            f_idx = idx % 4
            
            if self.tableau[src_idx]:
                card = self.tableau[src_idx][-1]
                if self._can_move_to_foundation(card, f_idx):
                    mask[action_idx] = True

        # 82-85: Stock -> Foundation 0-3
        if self.stock:
            card = self.stock[-1]
            for i in range(4):
                if self._can_move_to_foundation(card, i):
                    mask[82 + i] = True
        
        # 86: Surrender
        # Always available
        mask[86] = True
                    
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Deck
        self.deck = [Card(s, r) for s in range(4) for r in range(1, 14)]
        random.shuffle(self.deck)
        
        # Deal
        self.tableau = [[] for _ in range(7)]
        self.foundations = [0] * 4 # Top rank for each suit (0=Empty)
        self.waste = []
        self.stock = self.deck # Remaining cards
        
        for i in range(7):
            for j in range(i + 1):
                card = self.stock.pop()
                if j == i:
                    card.face_up = True
                self.tableau[i].append(card)
        
        # Track cards that have been seen in waste (for stock observation)
        self.seen_stock_cards = set()  # Set of (suit, rank) tuples
        
        self.steps = 0
        self.score = 0
        
        # Reset state history
        self.state_history = []
        
        # Calculate initial mask
        self.current_action_mask = self._calculate_action_mask()
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Tableau
        tableau_obs = np.zeros((7, self.max_tableau_height, 4), dtype=np.float32)
        for i, pile in enumerate(self.tableau):
            for j, card in enumerate(pile):
                if j >= self.max_tableau_height: break
                # [Present, Suit, Rank, FaceUp]
                # Normalize: Suit / 3.0, Rank / 13.0
                tableau_obs[i, j] = [1.0, card.suit / 3.0, card.rank / 13.0, 1.0 if card.face_up else 0.0]
        
        # Foundations
        # Normalize: Rank / 13.0
        foundations_obs = np.array([r / 13.0 for r in self.foundations], dtype=np.float32)
        
        # Waste
        waste_obs = np.zeros((3,), dtype=np.float32)
        if self.waste:
            top_card = self.waste[-1]
            waste_obs = np.array([1.0, top_card.suit / 3.0, top_card.rank / 13.0], dtype=np.float32)
            
        # Stock
        # Show all cards in stock (reversed so index 0 is the next card to draw)
        # BUT only show cards that have been seen before (revealed through waste)
        stock_obs = np.zeros((52, 3), dtype=np.float32)
        for i, card in enumerate(reversed(self.stock)):
            if (card.suit, card.rank) in self.seen_stock_cards:
                # [Present, Suit, Rank]
                stock_obs[i] = [1.0, card.suit / 3.0, card.rank / 13.0]
            else:
                # Card not yet seen - show as present but unknown (0, 0)
                stock_obs[i] = [1.0, 0.0, 0.0]
        
        return {
            'tableau': tableau_obs,
            'foundations': foundations_obs,
            'waste': waste_obs,
            'stock': stock_obs
        }

    def _get_state_hash(self, tableau=None, foundations=None, waste=None, stock=None):
        """
        Tworzy hash stanu gry do sprawdzenia powtórzeń.
        Może przyjąć opcjonalne argumenty do symulacji stanu.
        """
        t = tableau if tableau is not None else tuple(tuple((c.suit, c.rank, c.face_up) for c in pile) for pile in self.tableau)
        f = foundations if foundations is not None else tuple(self.foundations)
        w = waste if waste is not None else tuple((c.suit, c.rank) for c in self.waste)
        s = stock if stock is not None else tuple((c.suit, c.rank) for c in self.stock)
        
        state_tuple = (t, f, w, s)
        return hash(state_tuple)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {'is_success': False}
        
        # Penalty for time/step
        reward += self.rewards.get('time_penalty', -0.01)
        
        # ✅ CHECK FOR STATE REPETITION BEFORE ACTION
        current_state_hash = self._get_state_hash()
        if current_state_hash in self.state_history:
            # Stan się powtórzył - karamy model
            repeat_count = self.state_history.count(current_state_hash)
            repeat_penalty = self.repeat_penalty * (1 + repeat_count * 0.5)  # Rosnąca kara
            reward += repeat_penalty
            
            # Jeśli stan powtarza się zbyt często (np. 3 razy), przerywamy grę
            if repeat_count >= 3:
                terminated = True
                reward += -100.0 # Dodatkowa kara za doprowadzenie do pętli
                info['reason'] = "Loop Detected"
                
            if self.verbose:
                print(f"⚠️ State repetition detected (#{repeat_count}) - penalty: {repeat_penalty}")
        
        # Dodaj bieżący stan do historii
        self.state_history.append(current_state_hash)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        valid_move = False
        
        # 0: Draw Stock
        if action == 0:
            if self.stock:
                card = self.stock.pop()
                card.face_up = True
                self.waste.append(card)
                # Mark this card as seen
                self.seen_stock_cards.add((card.suit, card.rank))
                valid_move = True
            elif self.waste:
                # Recycle waste
                self.stock = self.waste[::-1]
                for c in self.stock: c.face_up = False
                self.waste = []
                reward += self.rewards.get('recycle_waste_penalty', -5.0)
                valid_move = True
        
        # 1-7: Waste -> Tableau 0-6
        elif 1 <= action <= 7:
            t_idx = action - 1
            if self.waste:
                card = self.waste[-1]
                if self._can_move_to_tableau(card, t_idx):
                    self.tableau[t_idx].append(self.waste.pop())
                    reward += self.rewards.get('move_waste_to_tableau', 3.0)
                    valid_move = True
        
        # 8-11: Waste -> Foundation 0-3
        elif 8 <= action <= 11:
            f_idx = action - 8
            if self.waste:
                card = self.waste[-1]
                if self._can_move_to_foundation(card, f_idx):
                    self.waste.pop()
                    self.foundations[f_idx] = card.rank
                    reward += self.rewards.get('move_to_foundation', 10.0)
                    valid_move = True

        # 12-53: Tableau i -> Tableau j
        elif 12 <= action <= 53:
            idx = action - 12
            src_idx = idx // 6
            dst_idx = idx % 6
            if dst_idx >= src_idx: dst_idx += 1
            
            if self.tableau[src_idx]:
                moved = False
                pile = self.tableau[src_idx]
                first_face_up = -1
                for i, c in enumerate(pile):
                    if c.face_up:
                        first_face_up = i
                        break
                
                if first_face_up != -1:
                    for i in range(first_face_up, len(pile)):
                        card = pile[i]
                        if self._can_move_to_tableau(card, dst_idx):
                            moving_cards = pile[i:]
                            self.tableau[dst_idx].extend(moving_cards)
                            self.tableau[src_idx] = pile[:i]
                            
                            # Flip new top of src if needed
                            flipped = False
                            if self.tableau[src_idx] and not self.tableau[src_idx][-1].face_up:
                                self.tableau[src_idx][-1].face_up = True
                                reward += self.rewards.get('flip_tableau_card', 5.0)
                                flipped = True
                            
                            # Reward logic for tableau moves
                            # Base reward for moving (can be 0.0 or negative to discourage useless moves)
                            move_reward = self.rewards.get('move_tableau_to_tableau', 0.0)
                            
                            # If we didn't flip a card and didn't empty a column, this might be a useless "lateral" move.
                            # Apply a small penalty to discourage "dancing" between valid spots.
                            if not flipped and self.tableau[src_idx]:
                                move_reward -= 0.5
                                
                            reward += move_reward
                            valid_move = True
                            moved = True
                            break
        
        # 54-81: Tableau i -> Foundation j
        elif 54 <= action <= 81:
            idx = action - 54
            src_idx = idx // 4
            f_idx = idx % 4
            
            if self.tableau[src_idx]:
                card = self.tableau[src_idx][-1]
                if self._can_move_to_foundation(card, f_idx):
                    self.tableau[src_idx].pop()
                    self.foundations[f_idx] = card.rank
                    
                    if self.tableau[src_idx] and not self.tableau[src_idx][-1].face_up:
                        self.tableau[src_idx][-1].face_up = True
                        reward += self.rewards.get('flip_tableau_card', 5.0)
                        
                    reward += self.rewards.get('move_to_foundation', 10.0)
                    valid_move = True

        # 82-85: Stock -> Foundation 0-3
        elif 82 <= action <= 85:
            f_idx = action - 82
            if self.stock:
                card = self.stock[-1]
                if self._can_move_to_foundation(card, f_idx):
                    self.stock.pop()
                    self.foundations[f_idx] = card.rank
                    reward += self.rewards.get('move_to_foundation', 10.0)
                    valid_move = True

        # 86: Surrender
        elif action == 86:
            reward += self.rewards.get('surrender_penalty', -50.0)
            terminated = True
            valid_move = True
            if self.verbose:
                print("🏳️ Surrendered")

        if not valid_move:
            reward += self.rewards.get('invalid_move_penalty', -1.0)

        # Update action mask for the new state
        self.current_action_mask = self._calculate_action_mask()

        # Check Win
        if all(f == 13 for f in self.foundations):
            reward += self.rewards.get('win_bonus', 1000.0)
            terminated = True
            info['is_success'] = True
            
        # Check Deadlock (No moves left)
        if not terminated and not any(self.current_action_mask):
            terminated = True
            # Opcjonalnie: mała kara za przegraną przez brak ruchów?
            # Ale sam brak nagrody za wygraną jest już wystarczającą "karą" (opportunity cost)
            
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def _can_move_to_tableau(self, card, t_idx):
        dest_pile = self.tableau[t_idx]
        if not dest_pile:
            return card.rank == 13 # King only on empty
        
        top_card = dest_pile[-1]
        return top_card.color() != card.color() and top_card.rank == card.rank + 1

    def _can_move_to_foundation(self, card, f_idx):
        # f_idx corresponds to suit? Usually foundations are specific to suits or auto-sorted.
        # Let's assume f_idx 0=Hearts, 1=Diamonds, etc. to simplify logic, 
        # OR allow any empty foundation to take an Ace, then it becomes that suit.
        # But our state is just "Top Rank". We need to know which suit is where.
        # To simplify: Fixed foundations. Foundation 0 is Hearts, 1 Diamonds, etc.
        
        target_suit = f_idx
        if card.suit != target_suit:
            return False
            
        current_rank = self.foundations[f_idx]
        return card.rank == current_rank + 1

    def render(self):
        if self.render_mode == "human":
            print(f"\nStep: {self.steps}")
            print(f"Foundations: {self.foundations}")
            print(f"Waste: {self.waste[-1] if self.waste else 'Empty'}")
            print("Tableau:")
            for i, pile in enumerate(self.tableau):
                print(f" {i}: {pile}")

def make_env():
    return SolitaireEnv
