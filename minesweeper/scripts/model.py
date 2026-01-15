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
    - Słownik z kluczem 'image' zawierającym planszę.
    - Wartości na planszy:
        -1: Zakryte pole
        0-8: Liczba min w sąsiedztwie
        -2: Poza planszą (dla paddingu do max_grid_size)
        
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
        self.current_mines_count = mines_count if mines_count else 10
        
        # Action space: flatten index of max grid
        self.action_space = spaces.Discrete(self.max_grid_size * self.max_grid_size)
        
        # Observation space
        # Używamy Dict, aby było kompatybilne z MultiInputPolicy
        # 3 Kanały: [Fog, Values, Valid]
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(3, self.max_grid_size, self.max_grid_size), 
                dtype=np.float32
            )
        })
        
        self.board = None           # Prawdziwa plansza (0: puste, 1: mina)
        self.revealed = None        # Maska odkrytych pól (bool)
        self.neighbor_counts = None # Pre-kalkulowane sąsiedztwo
        self.first_move = True
        self.game_over = False
        
        self.steps = 0
        
        # Pre-calculated padding mask (channel 2)
        # Will be updated in reset()
        self.valid_mask = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Obsługa opcji zmiany rozmiaru przy resecie
        if options:
            self.grid_size = options.get('grid_size', self.grid_size)
            
        # Losowanie rozmiaru jeśli nie ustalone na sztywno (dla treningu curricularnego)
        if options is None or 'grid_size' not in options:
            self.grid_size = random.randint(self.min_grid_size, self.max_grid_size)
            
        # Ustalanie liczby min
        mines_fraction = random.uniform(self.min_mines_fraction, self.max_mines_fraction)
        self.current_mines_count = int(self.grid_size * self.grid_size * mines_fraction)
        self.current_mines_count = max(1, min(self.current_mines_count, self.grid_size*self.grid_size - 9)) # Zawsze zostaw miejsce na start
        
        # Inicjalizacja plansz
        # Padding dla max_grid_size
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.revealed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.neighbor_counts = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Pre-calc Channel 2 (Valid Area)
        self.valid_mask = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.valid_mask[:self.grid_size, :self.grid_size] = 1.0
        
        self.first_move = True
        self.game_over = False
        self.steps = 0
        
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
        
        # Konwersja akcji na współrzędne
        # Actions are flat: y * max_width + x
        # Ale my operujemy na self.grid_size.
        # Jeśli akcja wskazuje poza self.grid_size -> kara i ignore.
        
        max_w = self.max_grid_size
        y = action // max_w
        x = action % max_w
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        reward_cfg = config['environment']['reward_scaling']
        should_terminate_invalid = reward_cfg.get('invalid_action_terminate', False)
        
        # Sprawdzenie czy ruch jest wewnątrz aktualnej planszy
        if y >= self.grid_size or x >= self.grid_size:
            reward = reward_cfg['invalid_action_penalty']
            if should_terminate_invalid:
                terminated = True
                info['result'] = 'invalid'
            return self._get_obs(), reward, terminated, False, info
            
        # Sprawdzenie czy pole już odkryte
        if self.revealed[y, x]:
            reward = reward_cfg['invalid_action_penalty']
            if should_terminate_invalid:
                terminated = True
                info['result'] = 'invalid_repeat'
            return self._get_obs(), reward, terminated, False, info
            
        # Pierwszy ruch - generacja planszy
        if self.first_move:
            self._place_mines(x, y)
            self.first_move = False
            
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
                                self.revealed[ny, nx] = True
                                reward += config['environment']['reward_scaling']['reveal_safe_multiplier'] # Mniejsza nagroda za auto-reveal
                                if self.neighbor_counts[ny, nx] == 0:
                                    stack.append((nx, ny))
        return reward

    def _get_obs(self):
        # 3 Kanały: [Fog, Values, Valid]
        obs_grid = np.zeros((3, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        
        # Channel 2: Valid Area (Pre-calculated in reset)
        obs_grid[2] = self.valid_mask
        
        # Przygotowanie widoków
        # Channel 0: Fog (1.0 = Unknown/Hidden, 0.0 = Revealed)
        # Domyślnie wszystko ukryte (1.0) tam gdzie valid
        obs_grid[0, :self.grid_size, :self.grid_size] = 1.0
        
        # Ustaw 0.0 tam gdzie revealed
        revealed_indices = np.where(self.revealed)
        obs_grid[0][revealed_indices] = 0.0
        
        # Channel 1: Values (0.0 - 1.0 representing 0-8 neighbors)
        # Tylko dla revealed. Values are 0-8. Normalize by dividing by 8.0
        # Bezpiecznie, bo revealed tylko tam gdzie valid
        values = self.neighbor_counts[revealed_indices].astype(np.float32) / 8.0
        obs_grid[1][revealed_indices] = values
        
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
