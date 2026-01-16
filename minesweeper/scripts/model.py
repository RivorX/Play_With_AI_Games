import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import yaml
import random
import pygame

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
        
        # Inicjalizacja wymiarów (domyślne lub zadane)
        self.grid_size = grid_size if grid_size else self.min_grid_size
        self._fixed_grid_size = grid_size is not None  # Flaga: czy rozmiar jest sztywny
        
        self.current_mines_count = mines_count if mines_count else 10
        self.max_steps = self.max_grid_size * self.max_grid_size * 2  # Limit kroków (2x plansza)
        
        # Action space: flatten index of max grid * 2 (Reveal | Flag)
        # 0 -> HW-1: Reveal
        # HW -> 2HW-1: Flag
        self.total_cells = self.max_grid_size * self.max_grid_size
        self.action_space = spaces.Discrete(self.total_cells * 2)
        
        # Observation space
        # Używamy Dict, aby było kompatybilne z MultiInputPolicy
        # 4 Kanały: [StateMap, Flags, LogicMine, LogicSafe]
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0.0, 
                high=10.0, 
                shape=(5, self.max_grid_size, self.max_grid_size), 
                dtype=np.float16 # FP16 dla RAMu
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
        
        # Konwersja akcji
        max_w = self.max_grid_size
        total_cells = max_w * max_w
        
        is_flag_action = False
        if action >= total_cells:
            is_flag_action = True
            action_idx = action - total_cells
        else:
            action_idx = action
            
        y = action_idx // max_w
        x = action_idx % max_w
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        reward_cfg = config['environment']['reward_scaling']
        should_terminate_invalid = reward_cfg.get('invalid_action_terminate', False)
        
        # Sprawdzenie czy ruch jest wewnątrz aktualnej planszy
        if y >= self.grid_size or x >= self.grid_size:
            reward = reward_cfg['invalid_action_penalty']
            self.invalid_action_count += 1
            max_invalid = reward_cfg.get('max_invalid_actions', 3)
            
            if should_terminate_invalid and self.invalid_action_count >= max_invalid:
                terminated = True
                info['result'] = 'invalid'
            return self._get_obs(), reward, terminated, False, info
            
        # Sprawdzenie czy pole już odkryte (wspólne dla Reveal i Flag)
        if self.revealed[y, x]:
            reward = reward_cfg['invalid_action_penalty']
            self.invalid_action_count += 1
            if should_terminate_invalid and self.invalid_action_count >= 10:
                terminated = True
                info['result'] = 'invalid_repeat'
            return self._get_obs(), reward, terminated, False, info
            
        ### OBSŁUGA FLAGI ###
        if is_flag_action:
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
            
            # Sprawdzenie wygranej
            # Wygrana = wszystkie bezpieczne pola odkryte
            if np.sum(self.revealed) == (self.grid_size * self.grid_size - self.current_mines_count):
                reward += reward_cfg['win_reward']
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
        reward = config['environment']['reward_scaling']['reveal_safe_reward']
        
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
                                reward += config['environment']['reward_scaling']['reveal_safe_multiplier'] # Mniejsza nagroda za auto-reveal
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
        Action Space = [Reveal_H*W, Flag_H*W]
        """
        total_cells = self.max_grid_size * self.max_grid_size
        full_mask = np.zeros(total_cells * 2, dtype=bool)
        
        # 1. Mask for Reveal
        # Valid: Inside Grid, Not Revealed, Not Flagged
        reveal_mask = np.zeros((self.max_grid_size, self.max_grid_size), dtype=bool)
        reveal_mask[:self.grid_size, :self.grid_size] = ~self.revealed & ~self.flags
        
        # 2. Mask for Flags
        # Valid: Inside Grid, Not Revealed
        # (Można flagować/odflagowywać dowolnie, byle nie odkryte)
        flag_mask = np.zeros((self.max_grid_size, self.max_grid_size), dtype=bool)
        flag_mask[:self.grid_size, :self.grid_size] = ~self.revealed
        
        full_mask[:total_cells] = reveal_mask.flatten()
        full_mask[total_cells:] = flag_mask.flatten()
        
        return full_mask

    def _get_obs(self):
        # 5 Kanałów:
        # Ch 0: State Map (0=Pad, 1-9=Val, 10=Fog)
        # Ch 1: Flags (1.0 = Flagged)
        # Ch 2: Logic Mine (1.0 = Deduced Mine)
        # Ch 3: Logic Safe (1.0 = Deduced Safe)
        # Ch 4: Needed Mines ((Val - SurroundingFlags) / 8.0)
        
        # Working on current grid slice only to save computation
        H, W = self.grid_size, self.grid_size
        
        obs_grid = np.zeros((5, self.max_grid_size, self.max_grid_size), dtype=np.float16)
        
        # --- Channel 0 (State) & 1 (Flags) Init ---
        obs_grid[0, :H, :W] = 10.0 # Default Fog
        
        flags_slice = self.flags[:H, :W]
        if np.any(flags_slice):
             obs_grid[1, :H, :W] = flags_slice.astype(np.float16)
        
        revealed_slice = self.revealed[:H, :W]
        
        # Jeśli nic nie odkryte, zwracamy pustą planszę (z fogiem)
        if not np.any(revealed_slice):
            return {'image': obs_grid}

        # --- Vectorized Calculations ---
        
        # Wartości liczbowe na planszy (0..8)
        # neighbor_counts przechowuje rzeczywistą liczbę min (ground truth), ale my widzimy ją tylko tam, gdzie revealed
        vals_map = self.neighbor_counts[:H, :W]
        
        # Wypełnienie Ch 0 (Odkryte)
        # Mapowanie: 0->1.0, 1->2.0 ... 8->9.0
        # Używamy maskowania bo jest bardzo szybkie
        obs_grid[0, :H, :W][revealed_slice] = vals_map[revealed_slice].astype(np.float16) + 1.0

        # Mapy pomocnicze do logiki
        hidden_slice = ~revealed_slice # Zakryte (w tym flagi)
        
        # Liczenie sąsiadów operacjami macierzowymi (Vectorized)
        # Zastępuje pętle nested loop
        neighbors_flags = self._count_neighbors_vectorized(flags_slice)
        neighbors_hidden_total = self._count_neighbors_vectorized(hidden_slice)
        
        # --- LOGIC RULES (Heuristic Solver) ---
        # Interesują nas tylko pola odkryte, które są cyframi (>0)
        # Cells that provide info: Revealed AND Value > 0
        info_mask = revealed_slice & (vals_map > 0)
        
        if np.any(info_mask):
            # Rule 1: Val == HiddenNeighbors (All hidden are mines)
            # e.g. "3" with 3 hidden neighbors
            rule1_triggers = info_mask & (vals_map == neighbors_hidden_total)
            
            # Rule 2: Val == FlaggedNeighbors (All remaining hidden are safe)
            # e.g. "2" with 2 flags
            rule2_triggers = info_mask & (vals_map == neighbors_flags)
            
            # Propagacja (Dilation)
            # Jeśli pole triggeruje regułę, to WSZYSCY jego sąsiedzi są...
            # Używamy tej samej funkcji count_neighbors, bo jeśli count > 0 to znaczy że jest sąsiadem triggera
            
            # Apply Rule 1 -> Mines
            if np.any(rule1_triggers):
                # Gdzie sąsiedzi triggerów?
                triggers_dilated = self._count_neighbors_vectorized(rule1_triggers) > 0
                # Logic Mine = Sąsiad triggera AND Zakryty AND Nie Oflagowany (już wiemy, ale dla czystości)
                # (Zaznaczamy tylko tam, gdzie jeszcze nie ma flagi, żeby zasugerować ruch)
                logic_mine = triggers_dilated & hidden_slice & ~flags_slice
                obs_grid[2, :H, :W][logic_mine] = 1.0

            # Apply Rule 2 -> Safe
            if np.any(rule2_triggers):
                triggers_dilated = self._count_neighbors_vectorized(rule2_triggers) > 0
                # Logic Safe = Sąsiad triggera AND Zakryty AND Nie Oflagowany
                logic_safe = triggers_dilated & hidden_slice & ~flags_slice
                obs_grid[3, :H, :W][logic_safe] = 1.0
        
        # --- Channel 4 (Needed Mines) ---
        # (Val - NeighborsFlags) / 8.0
        # Obliczamy dla całej mapy, nakładamy maskę info_mask (lub revealed)
        needed = vals_map - neighbors_flags
        # Clip < 0 (gdy za dużo flag) i normalizacja
        needed = np.clip(needed, 0, 8).astype(np.float16) / 8.0
        
        # Zapisujemy tylko dla odkrytych (dla zakrytych to bez sensu)
        obs_grid[4, :H, :W][revealed_slice] = needed[revealed_slice]
                                    
        return {'image': obs_grid}

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
