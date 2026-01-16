import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import yaml
import random
import pygame
import sys

# Dodaj katalog scripts do ścieżki
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.training_utils import get_difficulty_multiplier

# Wczytanie konfiguracji
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class MinesweeperEnv(gym.Env):
    """
    Środowisko Minesweeper (Saper) dla Gymnasium.
    
    Obserwacja:
    - Słownik z kluczem 'image' zawierającym planszę (4 kanały).
    - Kanał 0 (State):
        0.0: Padding (tło poza planszą)
        1.0-9.0: Odkryte cyfry (1=0 min, 9=8 min)
        10.0: Zakryte pole (Fog)
        
    Akcja:
    - Discrete(max_grid_size * max_grid_size)
    - Indeks pola do odkrycia: y * max_width + x
    """
    
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=None, mines_count=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 40  # Pixel size per cell
        self.screen = None   # Pygame surface
        
        self.min_grid_size = config['environment']['min_grid_size']
        self.max_grid_size = config['environment']['max_grid_size']
        self.min_mines_fraction = config['environment']['min_mines_fraction']
        self.max_mines_fraction = config['environment']['max_mines_fraction']
        
        # Nowe mechanizmy nagród
        self.progressive_config = config['environment']['reward_scaling'].get('progressive_reveal_bonus', {})
        self.milestones = config['environment']['reward_scaling'].get('milestones', {})
        self.efficiency_config = config['environment']['reward_scaling'].get('efficiency_bonus', {})
        
        # Inicjalizacja wymiarów (domyślne lub zadane)
        self.grid_size = grid_size if grid_size else self.min_grid_size
        self._fixed_grid_size = grid_size is not None  # Flaga: czy rozmiar jest sztywny
        
        # --- VIEWPORT CONFIG ---
        self.viewport_size = config['environment']['viewport_size']
        self.cursor_x = 0
        self.cursor_y = 0
        
        self.current_mines_count = mines_count if mines_count else 10
        self.max_steps = self.max_grid_size * self.max_grid_size * 2  # Limit kroków (2x plansza)
        
        # Action space (Cursor Mode):
        # 0: Up, 1: Down, 2: Left, 3: Right
        # 4: Reveal, 5: Flag
        self.action_space = spaces.Discrete(6)
        
        # Observation space
        # Używamy Dict, aby było kompatybilne z MultiInputPolicy
        # 4 Kanały: [StateMap, Flags, LogicMine, LogicSafe]
        # TERAZ: Wymiar stały 'viewport_size'
        # + VECTOR: 4 liczby (NormX, NormY, DirToUnknownX, DirToUnknownY)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0.0, 
                high=10.0, 
                shape=(5, self.viewport_size, self.viewport_size), 
                dtype=np.float16
            ),
            'vector': spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(4,), 
                dtype=np.float32
            )
        })
        
        self.board = None           # Prawdziwa plansza (0: puste, 1: mina)
        self.revealed = None        # Maska odkrytych pól (bool)
        self.flags = None           # Maska flag (bool)
        self.flag_interactions = None # Licznik interakcji flagowania per pole
        self.neighbor_counts = None # Pre-kalkulowane sąsiedztwo
        self.first_move = True
        self.game_over = False
        
        self.steps = 0
        self.invalid_action_count = 0  # Licznik invalid actions
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Obsługa opcji zmiany rozmiaru przy resecie
        if options:
            self.grid_size = options.get('grid_size', self.grid_size)
            
        # Losowanie rozmiaru tylko jeśli NIE jest ustalony na sztywno
        if not self._fixed_grid_size and (options is None or 'grid_size' not in options):
            self.grid_size = random.randint(self.min_grid_size, self.max_grid_size)
            
        # Ustalanie liczby min
        mines_fraction = random.uniform(self.min_mines_fraction, self.max_mines_fraction)
        self.current_mines_count = int(self.grid_size * self.grid_size * mines_fraction)
        self.current_mines_count = max(1, min(self.current_mines_count, self.grid_size*self.grid_size - 9)) # Zawsze zostaw miejsce na start
        
        # Inicjalizacja plansz
        # Padding dla max_grid_size
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.revealed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.flags = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.flag_interactions = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.neighbor_counts = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self.first_move = True
        self.game_over = False
        self.steps = 0
        self.invalid_action_count = 0
        self.steps_since_interaction = 0
        
        # Progressive bonus tracking
        self.reveals_count = 0
        self.milestones_achieved = set()
        
        # Difficulty multiplier for current grid
        self.difficulty_multiplier = get_difficulty_multiplier(self.grid_size, config['environment']['reward_scaling'])
        
        # Reset kursora na środek
        self.cursor_x = self.grid_size // 2
        self.cursor_y = self.grid_size // 2
        
        return self._get_obs(), {}

    def _place_mines(self, safe_x, safe_y):
        """Rozmieszcza miny, gwarantując, że (safe_x, safe_y) jest bezpieczne (i jego sąsiedzi, dla ułatwienia 'startu')"""
        positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        
        # Usuń pole startowe i jego sąsiadów z puli min, żeby pierwszy ruch był '0' (duże ułatwienie, standard w strzelankach)
        safe_zone = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = safe_y + dy, safe_x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if (ny, nx) in positions:
                        positions.remove((ny, nx))
        
        # Losuj miny
        mine_positions = random.sample(positions, self.current_mines_count)
        for r, c in mine_positions:
            self.board[r, c] = 1
            
        # Policz sąsiadów
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == 1:
                    continue
                
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0: continue
                        ny, nx = r + dy, c + dx
                        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                            if self.board[ny, nx] == 1:
                                count += 1
                self.neighbor_counts[r, c] = count

    def step(self, action):
        self.steps += 1
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        reward_cfg = config['environment']['reward_scaling']
        should_terminate_invalid = reward_cfg.get('invalid_action_terminate', False)
        
        # --- IDLE CHECK ---
        # Jeśli agent tylko się rusza i nic nie robi przez N kroków -> Koniec
        # Dostosowane do move_step: hver ruch przesuwa o 1/3 viewport'u
        move_step = self.viewport_size // 3
        idle_limit = (self.grid_size // move_step) * 2  # 2x przezentacja mapy
        if self.steps_since_interaction > idle_limit:
            reward = reward_cfg.get('idle_penalty', -10.0)
            terminated = True
            info['result'] = 'idle_timeout'
            return self._get_obs(), reward, terminated, truncated, info
            
        # --- MOVEMENT ACTIONS (0-3) ---
        if action < 4:
            self.steps_since_interaction += 1
            
            moved = False
            move_step = self.viewport_size // 3  # Przesunięcie o 1/3 viewport'u
            
            # 0: Up, 1: Down, 2: Left, 3: Right
            if action == 0 and self.cursor_y >= move_step:
                self.cursor_y -= move_step
                moved = True
            elif action == 1 and self.cursor_y < self.grid_size - move_step:
                self.cursor_y += move_step
                moved = True
            elif action == 2 and self.cursor_x >= move_step:
                self.cursor_x -= move_step
                moved = True
            elif action == 3 and self.cursor_x < self.grid_size - move_step:
                self.cursor_x += move_step
                moved = True
                
            if moved:
                reward += reward_cfg.get('move_penalty', -0.01)
            else:
                # Wall hit
                reward += reward_cfg.get('invalid_action_penalty', -1.0)
                
            return self._get_obs(), reward, terminated, False, info
            
        # --- INTERACTION ACTIONS (4-5) ---
        # 4: Reveal, 5: Flag
        
        # Reset idle counter on attempted interaction
        self.steps_since_interaction = 0
        
        # All interactions happen at cursor_x, cursor_y
        x, y = self.cursor_x, self.cursor_y
            
        # Sprawdzenie czy pole już odkryte (wspólne dla Reveal i Flag)
        if self.revealed[y, x]:
            reward = reward_cfg['invalid_action_penalty']
            self.invalid_action_count += 1
            # W Cursor Mode: Nie przerywaj gry, po prostu kara
            # Agent musi nauczyć się nie klikać odkrytych
            return self._get_obs(), reward, terminated, False, info
            
        ### OBSŁUGA FLAGI ###
        if action == 5:
            # LIMIT FLAG = Liczba min (exploit prevention)
            current_flag_count = np.sum(self.flags)
            
            # Jeśli próbuje postawić flagę a już osiągnął limit -> INSTANT LOSE
            if not self.flags[y, x] and current_flag_count >= self.current_mines_count:
                reward = reward_cfg.get('explosion_penalty', -10.0)  # Duża kara
                terminated = True
                info['result'] = 'flag_limit_exceeded'
                return self._get_obs(), reward, terminated, False, info
            
            # Eksploit Fix: Limit interakcji z flagą na jednym polu
            self.flag_interactions[y, x] += 1
            if self.flag_interactions[y, x] > 4:
                # Zbyt częste zmienianie zdania = Invalid Action
                reward = reward_cfg['invalid_action_penalty']
                self.invalid_action_count += 1
                if should_terminate_invalid and self.invalid_action_count >= 10:
                    terminated = True
                    info['result'] = 'invalid_repeat_flag'
                return self._get_obs(), reward, terminated, False, info
            
            # Toggle flagi
            was_flagged = self.flags[y, x]
            self.flags[y, x] = not was_flagged
            
            # Action Cost (zniechęca do spamowania)
            reward += reward_cfg.get('flag_redundant_penalty', -0.05)
            
            # Nagradzanie/Karanie za flagowanie (Symetryczne)
            if self.flags[y, x]: # Postawiono flagę
                if self.board[y, x] == 1:
                    # Poprawne oflagowanie miny
                    reward += reward_cfg.get('flag_correct_reward', 0.5)
                else:
                    # Błędne oflagowanie bezpiecznego pola
                    reward += reward_cfg.get('flag_incorrect_penalty', -1.0)
            else:
                # Zdjęto flagę - ODWRACAMY nagrody/kary
                if self.board[y, x] == 1:
                    # Zdjęto flagę z miny -> odbieramy nagrodę
                    reward -= reward_cfg.get('flag_correct_reward', 0.5)
                else:
                    # Zdjęto flagę z bezpiecznego pola -> cofamy karę (naprawił błąd)
                    reward -= reward_cfg.get('flag_incorrect_penalty', -1.0)
                
            return self._get_obs(), reward, terminated, False, info
            
        ### OBSŁUGA ODKRYCIA (REVEAL) ###
        # Action == 4
        
        # Nie można odkryć oflagowanego pola
        if self.flags[y, x]:
            # Traktujemy to jako stratę ruchu lub invalid
            reward = reward_cfg.get('invalid_action_penalty', -0.1)
            return self._get_obs(), reward, terminated, False, info

        # Guess Factor Implementation
        # Jeśli ruch nie jest logicznie uzasadniony przez Solver, dodaj lekką karę
        if not self.first_move:
             # Sprawdź czy jakikolwiek sąsiad gwarantuje bezpieczeństwo
             is_safe_guaranteed = False
             
             # Sprawdzamy sąsiadów odkrytego pola (y, x)
             # Ale pole (y, x) jest jeszcze zakryte w 'neighbor_counts' logicznie
             # Więc musimy sprawdzić, czy któryś z JEGO ODKRYTYCH SĄSIADÓW ma (Val == CleanFlags)
             
             # Szybkie sprawdzenie (bez pętli po całej planszy, tylko lokalnie)
             for dy in [-1, 0, 1]:
                 for dx in [-1, 0, 1]:
                     if dy==0 and dx==0: continue
                     ny, nx = y+dy, x+dx
                     if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                         if self.revealed[ny, nx] and self.board[ny, nx] == 0:
                             val = self.neighbor_counts[ny, nx]
                             if val > 0:
                                 # Policz flagi wokół TEGO sąsiada (ny, nx)
                                 flags_cnt = 0
                                 for fdy in [-1, 0, 1]:
                                     for fdx in [-1, 0, 1]:
                                         if fdy==0 and fdx==0: continue
                                         fy, fx = ny+fdy, nx+fdx
                                         if 0 <= fy < self.grid_size and 0 <= fx < self.grid_size:
                                             if self.flags[fy, fx]:
                                                 flags_cnt += 1
                                 
                                 if val == flags_cnt:
                                     is_safe_guaranteed = True
                                     break
                 if is_safe_guaranteed: break
                 
             if not is_safe_guaranteed:
                 reward += reward_cfg.get('guess_penalty', 0.0)

        # Pierwszy ruch - generacja planszy
        if self.first_move:
            self._place_mines(x, y)
            self.first_move = False
            # Upewnij się, że startowe pole nie ma flagi (technicznie możliwe, jeśli agent dał flagę w 1 kroku)
            self.flags[y, x] = False 
            
        # Odkrycie pola
        if self.board[y, x] == 1:
            # BOOM
            self.game_over = True
            self.revealed[y, x] = True
            reward = reward_cfg['explosion_penalty']
            terminated = True
            info['result'] = 'loss'
        else:
            # Bezpieczne
            reward += self._reveal(x, y)
            
            # Milestone rewards - bonusy za osiągnięcie % postępu
            total_safe = self.grid_size * self.grid_size - self.current_mines_count
            revealed_fraction = np.sum(self.revealed) / total_safe
            
            for milestone_threshold, milestone_reward in self.milestones.items():
                # milestone_threshold to już float (0.1, 0.25, etc.)
                if revealed_fraction >= milestone_threshold and milestone_threshold not in self.milestones_achieved:
                    reward += milestone_reward * self.difficulty_multiplier
                    self.milestones_achieved.add(milestone_threshold)
                    info[f'milestone_{int(milestone_threshold*100)}pct'] = True
            
            # Sprawdzenie wygranej
            # Wygrana = wszystkie bezpieczne pola odkryte
            if np.sum(self.revealed) == total_safe:
                reward += reward_cfg['win_reward'] * self.difficulty_multiplier
                
                # Efficiency bonus - nagroda za mało kroków
                if self.efficiency_config.get('enabled', False):
                    threshold = self.efficiency_config.get('threshold_steps_per_reveal', 1.5)
                    bonus_value = self.efficiency_config.get('bonus_value', 10.0)
                    efficiency_ratio = self.steps / max(self.reveals_count, 1)
                    if efficiency_ratio < threshold:
                        reward += bonus_value * self.difficulty_multiplier
                        info['efficiency_bonus'] = True
                
                terminated = True
                info['result'] = 'win'
        
        # Sprawdź limit kroków (zapobiega nieskończonym grom przy invalid actions)
        if self.steps >= self.max_steps:
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, info

    def _reveal(self, x, y):
        """Rekurencyjne odkrywanie pustych pól. Zwraca skumulowaną nagrodę."""
        if self.revealed[y, x]:
            return 0
        
        self.revealed[y, x] = True
        self.reveals_count += 1
        
        # Progressive bonus - nagroda rośnie z każdym odkryciem
        base_reward = config['environment']['reward_scaling']['reveal_safe_reward']
        if self.progressive_config.get('enabled', False):
            increment = self.progressive_config.get('increment_per_reveal', 0.02)
            cap = self.progressive_config.get('max_multiplier', 3.0)
            multiplier = min(1.0 + self.reveals_count * increment, cap)
            reward = base_reward * multiplier * self.difficulty_multiplier
        else:
            reward = base_reward * self.difficulty_multiplier
        
        if self.neighbor_counts[y, x] == 0:
            # Jeśli 0 sąsiadów, odkryj wszystkich dookoła
            # Flood fill
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = cx + dx, cy + dy
                        
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if not self.revealed[ny, nx]:
                                # Auto-reveal zdejmuje flagi (jeśli tam były błędnie postawione)
                                self.flags[ny, nx] = False
                                self.revealed[ny, nx] = True
                                self.reveals_count += 1
                                
                                # Progressive bonus dla auto-reveal
                                base_auto_reward = config['environment']['reward_scaling']['reveal_safe_multiplier']
                                if self.progressive_config.get('enabled', False):
                                    increment = self.progressive_config.get('increment_per_reveal', 0.02)
                                    cap = self.progressive_config.get('max_multiplier', 3.0)
                                    multiplier = min(1.0 + self.reveals_count * increment, cap)
                                    reward += base_auto_reward * multiplier * self.difficulty_multiplier
                                else:
                                    reward += base_auto_reward * self.difficulty_multiplier
                                    
                                if self.neighbor_counts[ny, nx] == 0:
                                    stack.append((nx, ny))
        return reward

    def _count_neighbors_vectorized(self, input_map):
        """
        Szybkie liczenie sąsiadów (suma 3x3 - center) dla całej macierzy 2D.
        Input: 2D numpy array (bool or int)
        Output: 2D numpy array (int)
        """
        # Konwersja na int (0/1)
        map_int = input_map.astype(int)
        
        # Padding zerami dookoła (żeby krawędzie miały sąsiadów '0')
        padded = np.pad(map_int, pad_width=1, mode='constant', constant_values=0)
        
        # Sumowanie 8 przesunięć (szybsze niż scipy.convolve dla małych kerneli w czystym numpy)
        # Top Row
        tl = padded[:-2, :-2]
        tm = padded[:-2, 1:-1]
        tr = padded[:-2, 2:]
        # Mid Row
        ml = padded[1:-1, :-2]
        # center = padded[1:-1, 1:-1] # Skip center
        mr = padded[1:-1, 2:]
        # Bot Row
        bl = padded[2:, :-2]
        bm = padded[2:, 1:-1]
        br = padded[2:, 2:]
        
        return tl + tm + tr + ml + mr + bl + bm + br

    def action_masks(self):
        """
        Zwraca maskę poprawnych akcji.
        Action Space = [Up, Down, Left, Right, Reveal, Flag]
        """
        mask = np.zeros(6, dtype=bool)
        
        # Movement
        mask[0] = self.cursor_y > 0 # Up
        mask[1] = self.cursor_y < self.grid_size - 1 # Down
        mask[2] = self.cursor_x > 0 # Left
        mask[3] = self.cursor_x < self.grid_size - 1 # Right
        
        # Interaction (at cursor)
        is_revealed = self.revealed[self.cursor_y, self.cursor_x]
        is_flagged = self.flags[self.cursor_y, self.cursor_x]

        # Reveal: Valid if not revealed AND not flagged
        mask[4] = not is_revealed and not is_flagged
        
        # Flag: Valid if not revealed
        mask[5] = not is_revealed
        
        return mask

    def _get_obs(self):
        # 5 Channels: State, Flags, LogicM, LogicS, Needed
        # Output: V x V crop around cursor
        
        V = self.viewport_size
        half_v = V // 2
        
        # Working on current grid slice only to save computation
        H, W = self.grid_size, self.grid_size
        
        # --- 1. Compute FULL maps first (Optimization: only needed locally, but easier globally) ---
        # Logic helpers works on full board
        
        # Init maps (on grid_size)
        state_map = np.full((H, W), 10.0, dtype=np.float16) # Fog
        flag_map = self.flags[:H, :W].astype(np.float16)
        logic_m = np.zeros((H, W), dtype=np.float16)
        logic_s = np.zeros((H, W), dtype=np.float16)
        needed_map = np.zeros((H, W), dtype=np.float16)
        
        revealed_slice = self.revealed[:H, :W]
        vals_map = self.neighbor_counts[:H, :W]
        
        # Fill State Map
        state_map[revealed_slice] = vals_map[revealed_slice].astype(np.float16) + 1.0
        
        # Logic Computation (Vectorized on full board)
        hidden_slice = ~revealed_slice
        flags_bool = self.flags[:H, :W]
        
        neighbors_flags = self._count_neighbors_vectorized(flags_bool)
        neighbors_hidden_total = self._count_neighbors_vectorized(hidden_slice)
        
        info_mask = revealed_slice & (vals_map > 0)
        
        if np.any(info_mask):
            # Rule 1
            rule1_triggers = info_mask & (vals_map == neighbors_hidden_total)
            if np.any(rule1_triggers):
                triggers_dilated = self._count_neighbors_vectorized(rule1_triggers) > 0
                logic_m = (triggers_dilated & hidden_slice & ~flags_bool).astype(np.float16)

            # Rule 2
            rule2_triggers = info_mask & (vals_map == neighbors_flags)
            if np.any(rule2_triggers):
                triggers_dilated = self._count_neighbors_vectorized(rule2_triggers) > 0
                logic_s = (triggers_dilated & hidden_slice & ~flags_bool).astype(np.float16)
                
        # Needed Calculation
        needed = vals_map - neighbors_flags
        needed_norm = np.clip(needed, 0, 8).astype(np.float16) / 8.0
        needed_map[revealed_slice] = needed_norm[revealed_slice]
        
        # --- 2. CROP ---
        # Stack channels: (5, H, W)
        full_obs = np.stack([state_map, flag_map, logic_m, logic_s, needed_map], axis=0)
        
        # Pad with 0.0 (except State channel? Fog is 10.0, Pad is 0.0. Let's use 0.0 for pad)
        # We need padding of size 'half_v' on all sides
        # Using constant 0 (Pad) for all channels suits well
        padded_obs = np.pad(full_obs, ((0,0), (half_v, half_v), (half_v, half_v)), mode='constant', constant_values=0)
        
        # Crop coordinates
        # Original (cx, cy) -> In padded: (cx + half_v, cy + half_v)
        # Top-Left of crop: (cx, cy)
        y_start = self.cursor_y
        x_start = self.cursor_x 
        # range: [y_start : y_start+V]
        
        obs_crop = padded_obs[:, y_start : y_start+V, x_start : x_start+V]
        
        # --- 3. VECTOR (Global Stats) ---
        # 1. Normalized Position (-1..1 is better for NN, but 0..1 is fine)
        # Let's use -0.5 to 0.5 centered
        norm_x = (self.cursor_x / max(1, W-1)) - 0.5
        norm_y = (self.cursor_y / max(1, H-1)) - 0.5
        
        # 2. Direction to Nearest Unknown (Exploration Hint)
        dir_x, dir_y = 0.0, 0.0
        
        # Find unrevealed indices
        # Optimization: Don't scan everything if not needed?
        # Scan is fast enough for 16x16 or 32x32
        
        # We need coords of ~revealed
        # hidden_slice calculated earlier
        # ~revealed is simply bool
        
        # Argwhere returns (y, x)
        unrevealed_coords = np.argwhere(~self.revealed[:H, :W])
        
        if len(unrevealed_coords) > 0:
            # Calculate distances to cursor (Manhattan or Euclidean)
            # cursor is (self.cursor_y, self.cursor_x)
            
            # Vectors from cursor to targets
            # dy = target_y - cursor_y
            dy = unrevealed_coords[:, 0] - self.cursor_y
            dx = unrevealed_coords[:, 1] - self.cursor_x
            
            # Squared Euclidean distance
            dists_sq = dx*dx + dy*dy
            
            # Find nearest
            nearest_idx = np.argmin(dists_sq)
            
            # Get vector
            best_dy = dy[nearest_idx]
            best_dx = dx[nearest_idx]
            
            # Normalize vector to length 1 (or by max dimension)
            # Simply sign or normalized? Normalized gives smooth gradient
            dist = np.sqrt(dists_sq[nearest_idx])
            if dist > 0:
                dir_x = best_dx / dist
                dir_y = best_dy / dist
        
        vector_obs = np.array([norm_x, norm_y, dir_x, dir_y], dtype=np.float32)
                                    
        return {'image': obs_crop, 'vector': vector_obs}

    def render(self):
        if self.render_mode == "ansi":
            print(f"\n--- Minesweeper {self.grid_size}x{self.grid_size} | Mines: {self.current_mines_count} ---")
            print("   " + " ".join([str(i%10) for i in range(self.grid_size)]))
            for r in range(self.grid_size):
                line = []
                for c in range(self.grid_size):
                    if self.revealed[r, c]:
                        if self.board[r, c] == 1:
                            symbol = "X"
                        elif self.neighbor_counts[r, c] == 0:
                            symbol = "."
                        else:
                            symbol = str(self.neighbor_counts[r, c])
                    else:
                        symbol = "#"
                    line.append(symbol)
                print(f"{r:2d} " + " ".join(line))
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window_width = self.grid_size * self.cell_size
            self.window_height = self.grid_size * self.cell_size
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        
        if self.screen is None:
            # Create surface if not exists (for rgb_array or initial human)
            self.window_width = self.grid_size * self.cell_size
            self.window_height = self.grid_size * self.cell_size
            self.screen = pygame.Surface((self.window_width, self.window_height))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Colors
        COLOR_HIDDEN = (192, 192, 192)
        COLOR_REVEALED = (220, 220, 220)
        COLOR_BORDER_LIGHT = (255, 255, 255)
        COLOR_BORDER_DARK = (128, 128, 128)
        COLOR_TEXT = {
            1: (0, 0, 255),    # Blue
            2: (0, 128, 0),    # Green
            3: (255, 0, 0),    # Red
            4: (0, 0, 128),    # Dark Blue
            5: (128, 0, 0),    # Maroon
            6: (0, 128, 128),  # Cyan
            7: (0, 0, 0),      # Black
            8: (128, 128, 128) # Gray
        }
        COLOR_MINE = (0, 0, 0)
        COLOR_MINE_BG = (255, 0, 0)
        # New: Cursor Color
        COLOR_CURSOR = (255, 255, 0) # Yellow outline
        
        # Font
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont("Arial", int(self.cell_size * 0.7), bold=True)

        self.screen.fill(COLOR_HIDDEN)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                if self.revealed[y, x]:
                    # Revealed Cell
                    pygame.draw.rect(self.screen, COLOR_REVEALED, rect)
                    pygame.draw.rect(self.screen, COLOR_BORDER_DARK, rect, 1)
                    
                    if self.board[y, x] == 1:
                        # Mine
                        pygame.draw.rect(self.screen, COLOR_MINE_BG, rect)
                        pygame.draw.circle(self.screen, COLOR_MINE, rect.center, self.cell_size // 4)
                    elif self.neighbor_counts[y, x] > 0:
                        # Number
                        text_surf = font.render(str(self.neighbor_counts[y, x]), True, COLOR_TEXT.get(self.neighbor_counts[y, x], (0,0,0)))
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                        
                else:
                    # Hidden Cell (3D effect)
                    pygame.draw.rect(self.screen, COLOR_HIDDEN, rect)
                    # Bevels
                    pygame.draw.line(self.screen, COLOR_BORDER_LIGHT, rect.topleft, rect.topright, 3)
                    pygame.draw.line(self.screen, COLOR_BORDER_LIGHT, rect.topleft, rect.bottomleft, 3)
                    pygame.draw.line(self.screen, COLOR_BORDER_DARK, rect.bottomleft, rect.bottomright, 3)
                    pygame.draw.line(self.screen, COLOR_BORDER_DARK, rect.topright, rect.bottomright, 3)
                    
                    if self.flags[y, x]:
                        # Draw Flag
                        flag_color = (255, 0, 0)
                        # Triangle
                        p1 = (rect.centerx - 5, rect.centery + 5)
                        p2 = (rect.centerx + 5, rect.centery + 5)
                        p3 = (rect.centerx, rect.centery - 5)
                        pygame.draw.polygon(self.screen, flag_color, [p1, p2, p3])
                        # Pole
                        pygame.draw.line(self.screen, (0,0,0), (rect.centerx, rect.centery-5), (rect.centerx, rect.centery+10), 2)

        # --- DRAW CURSOR ---
        cx, cy = self.cursor_x * self.cell_size, self.cursor_y * self.cell_size
        cursor_rect = pygame.Rect(cx, cy, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, COLOR_CURSOR, cursor_rect, 4) # 4px thickness
        
        # Draw Viewport Boundary (Optional visualization of what AI sees)
        V = self.viewport_size
        half_v = V // 2
        # Calculate viewport top-left
        vx = (self.cursor_x - half_v) * self.cell_size
        vy = (self.cursor_y - half_v) * self.cell_size
        vw = V * self.cell_size
        vh = V * self.cell_size
        pygame.draw.rect(self.screen, (0, 255, 255), (vx, vy, vw, vh), 1) # Cyan thin line

        if self.render_mode == "human":
            self.window.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

def make_env(render_mode=None, grid_size=None, mines_count=None):
    def _init():
        return MinesweeperEnv(render_mode=render_mode, grid_size=grid_size, mines_count=mines_count)
    return _init
