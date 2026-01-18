import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml
import os
from collections import deque

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class MinesweeperEnv(gym.Env):
    """
    🎮 Minesweeper Environment - AI learns to play Minesweeper
    
    Observation:
    - image: FIXED viewport_size x viewport_size (padded if grid_size < viewport_size)
    - scalars: remaining_cells, flags_used, mine_density, etc.
    
    Action:
    - Discrete(grid_size * grid_size): choose a cell to reveal
    
    For smaller grids: unused cells are marked with -1.0 (masked area)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=None):
        super().__init__()
        
        # Viewport size (FIXED for all grids)
        self.viewport_size = config['environment']['viewport_size']
        
        # Grid size (variable, but <= viewport_size)
        if grid_size is None:
            self.grid_size = np.random.randint(
                config['environment']['min_grid_size'],
                config['environment']['max_grid_size'] + 1
            )
        else:
            self.grid_size = grid_size
        
        # Validate
        if self.grid_size > self.viewport_size:
            raise ValueError(f"grid_size ({self.grid_size}) cannot be larger than viewport_size ({self.viewport_size})")
        
        # Mine configuration
        self.mine_density = config['environment']['mine_density']
        self.num_mines = int(self.grid_size * self.grid_size * self.mine_density)
        
        # Reward config
        self.reward_config = config['environment'].get('reward_scaling', {})
        self.reward_scaling_enabled = self.reward_config.get('enable', True)
        
        if self.reward_scaling_enabled:
            min_grid = config['environment']['min_grid_size']
            max_grid = config['environment']['max_grid_size']
            min_mult = self.reward_config.get('min_difficulty_multiplier', 1.0)
            max_mult = self.reward_config.get('max_difficulty_multiplier', 2.0)
            
            if max_grid > min_grid:
                progress = (self.grid_size - min_grid) / (max_grid - min_grid)
                self.difficulty_multiplier = min_mult + (max_mult - min_mult) * progress
            else:
                self.difficulty_multiplier = min_mult
        else:
            self.difficulty_multiplier = 1.0
        
        # Reward values
        self.base_safe_cell_reward = self.reward_config.get('base_safe_cell_reward', 1.0)
        self.base_mine_penalty = self.reward_config.get('base_mine_penalty', -10.0)
        self.base_victory_bonus = self.reward_config.get('base_victory_bonus', 100.0)
        self.milestones = self.reward_config.get('milestones', {})
        self.efficiency_config = self.reward_config.get('efficiency_bonus', {})
        
        # Tracking
        self.milestones_achieved = set()
        
        # Action space: FIXED size (viewport_size * viewport_size)
        # Invalid actions (outside grid_size) will be masked/penalized
        self.action_space = spaces.Discrete(self.viewport_size * self.viewport_size)
        
        # Observation space (FIXED viewport_size)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=-1.0,  # Allow -1.0 for masked areas
                high=1.0,
                shape=(self.viewport_size, self.viewport_size, 1),
                dtype=np.float32
            ),
            'remaining_cells': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'revealed_ratio': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'mine_density_norm': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'steps_per_cell': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'grid_size_norm': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Game state
        self.mine_grid = None      # True = mine
        self.number_grid = None    # Number of adjacent mines
        self.revealed = None       # True = revealed
        self.flagged = None        # True = flagged
        
        self.score = 0
        self.steps = 0
        self.total_reward = 0.0
        self.safe_cells_total = 0
        self.safe_cells_revealed = 0
        
        # Pre-allocated viewport array (FIXED SIZE)
        self.obs_array = np.zeros((self.viewport_size, self.viewport_size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'grid_size' in options:
            self.grid_size = options['grid_size']
            
            # Validate
            if self.grid_size > self.viewport_size:
                raise ValueError(f"grid_size ({self.grid_size}) cannot be larger than viewport_size ({self.viewport_size})")
            
            self.num_mines = int(self.grid_size * self.grid_size * self.mine_density)
        
        # Recalculate difficulty multiplier
        if self.reward_scaling_enabled:
            min_grid = config['environment']['min_grid_size']
            max_grid = config['environment']['max_grid_size']
            min_mult = self.reward_config.get('min_difficulty_multiplier', 1.0)
            max_mult = self.reward_config.get('max_difficulty_multiplier', 2.0)
            
            if max_grid > min_grid:
                progress = (self.grid_size - min_grid) / (max_grid - min_grid)
                self.difficulty_multiplier = min_mult + (max_mult - min_mult) * progress
            else:
                self.difficulty_multiplier = min_mult
        
        # Reset milestone tracking
        self.milestones_achieved = set()
        
        # Initialize grids
        self.mine_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.number_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.revealed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.flagged = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Place mines randomly
        mine_positions = np.random.choice(
            self.grid_size * self.grid_size, 
            size=self.num_mines, 
            replace=False
        )
        for pos in mine_positions:
            row = pos // self.grid_size
            col = pos % self.grid_size
            self.mine_grid[row, col] = True
        
        # Calculate numbers
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if not self.mine_grid[r, c]:
                    count = self._count_adjacent_mines(r, c)
                    self.number_grid[r, c] = count
        
        self.score = 0
        self.steps = 0
        self.total_reward = 0.0
        self.safe_cells_total = self.grid_size * self.grid_size - self.num_mines
        self.safe_cells_revealed = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        self.steps += 1
        
        # Decode action (based on viewport_size)
        row = action // self.viewport_size
        col = action % self.viewport_size
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # ✅ CHECK: Action outside actual grid (invalid action)
        if row >= self.grid_size or col >= self.grid_size:
            # Heavy penalty for choosing masked area
            reward = -1.0
            observation = self._get_obs()
            info = self._get_info()
            info['invalid_action'] = True
            return observation, reward, terminated, truncated, info
        
        # Check if cell is already revealed or flagged
        if self.revealed[row, col]:
            # Penalty for clicking already revealed cell
            reward = -0.1
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Reveal cell
        self.revealed[row, col] = True
        
        # Check if mine
        if self.mine_grid[row, col]:
            # Hit a mine - game over
            terminated = True
            mine_penalty = self.base_mine_penalty * self.difficulty_multiplier
            reward = mine_penalty
            info = self._get_info()
            info['termination_reason'] = 'mine'
            
            if self.render_mode == "human":
                self._render_frame()
            
            return self._get_obs(), reward, terminated, truncated, info
        
        # Safe cell revealed
        cells_revealed_before = self.safe_cells_revealed
        
        # If it's a 0, reveal all adjacent cells recursively
        if self.number_grid[row, col] == 0:
            self._flood_fill(row, col)
        else:
            self.safe_cells_revealed += 1
        
        cells_revealed_now = self.safe_cells_revealed - cells_revealed_before
        
        # Reward for revealing safe cells
        safe_cell_reward = self.base_safe_cell_reward * cells_revealed_now * self.difficulty_multiplier
        reward += safe_cell_reward
        
        # Check milestones
        current_progress = self.safe_cells_revealed / self.safe_cells_total
        
        for threshold_float, bonus in self.milestones.items():
            threshold = float(threshold_float)
            
            if current_progress >= threshold and threshold not in self.milestones_achieved:
                reward += bonus * self.difficulty_multiplier
                self.milestones_achieved.add(threshold)
                
                if threshold >= 0.99:
                    print(f"🏆🏆🏆 FULL BOARD CLEARED! Grid={self.grid_size}x{self.grid_size} 🏆🏆🏆")
        
        # Check victory
        if self.safe_cells_revealed >= self.safe_cells_total:
            terminated = True
            victory_bonus = self.base_victory_bonus * self.difficulty_multiplier
            reward += victory_bonus
            
            # Efficiency bonus
            if self.efficiency_config.get('enable', False):
                steps_per_cell = self.steps / max(self.safe_cells_revealed, 1)
                efficiency_threshold = self.efficiency_config.get('threshold', 1.5)
                efficiency_reward = self.efficiency_config.get('reward', 10.0)
                
                if steps_per_cell < efficiency_threshold:
                    reward += efficiency_reward * self.difficulty_multiplier
            
            info = self._get_info()
            info['termination_reason'] = 'victory'
            
            if self.render_mode == "human":
                self._render_frame()
            
            return self._get_obs(), reward, terminated, truncated, info
        
        # Max steps
        max_steps = config['environment']['max_steps_factor'] * self.grid_size
        if self.steps >= max_steps:
            truncated = True
        
        self.total_reward += reward
        self.score = self.safe_cells_revealed
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info

    def _count_adjacent_mines(self, row, col):
        """Count mines in 8 adjacent cells"""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    if self.mine_grid[r, c]:
                        count += 1
        return count

    def _flood_fill(self, row, col):
        """Recursively reveal cells with 0 adjacent mines"""
        queue = deque([(row, col)])
        visited = set()
        
        while queue:
            r, c = queue.popleft()
            
            if (r, c) in visited:
                continue
            
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                continue
            
            if self.revealed[r, c]:
                continue
            
            visited.add((r, c))
            self.revealed[r, c] = True
            self.safe_cells_revealed += 1
            
            # If this cell has 0 adjacent mines, add neighbors
            if self.number_grid[r, c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        queue.append((r + dr, c + dc))

    def _get_obs(self):
        """
        Returns observation with FIXED viewport_size
        
        Image encoding:
        - -1.0: masked area (outside actual grid) - model learns to avoid these
        - 0.0: unrevealed
        - 0.1-0.9: revealed with number (0.1=0 adjacent, 0.2=1 adjacent, ..., 0.9=8 adjacent)
        - 1.0: flagged (not used in current version)
        
        Smaller grids are placed at top-left, rest is masked with -1.0
        Model receives penalty for clicking masked areas, so it learns to avoid them.
        """
        # Fill with mask value (-1.0) for areas outside grid
        self.obs_array.fill(-1.0)
        
        # Fill actual grid area
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.revealed[r, c]:
                    # Revealed cell: encode number (0-8 -> 0.1-0.9)
                    num = self.number_grid[r, c]
                    self.obs_array[r, c] = 0.1 + (num / 10.0)
                elif self.flagged[r, c]:
                    self.obs_array[r, c] = 1.0
                else:
                    self.obs_array[r, c] = 0.0
        
        obs_image = np.expand_dims(self.obs_array, axis=-1)
        
        # Scalars
        remaining_cells = self.safe_cells_total - self.safe_cells_revealed
        remaining_cells_norm = (remaining_cells / self.safe_cells_total) * 2.0 - 1.0
        
        revealed_ratio_norm = (self.safe_cells_revealed / self.safe_cells_total) * 2.0 - 1.0
        
        mine_density_norm = (self.mine_density - 0.1) / 0.2  # Normalize [0.1, 0.3] -> [-0.5, 0.5]
        
        steps_per_cell = self.steps / max(self.safe_cells_revealed, 1)
        steps_per_cell_norm = np.clip((steps_per_cell - 2.0) / 5.0, -1.0, 1.0)
        
        # Grid size normalized
        min_grid = config['environment']['min_grid_size']
        max_grid = config['environment']['max_grid_size']
        grid_size_norm = (self.grid_size - min_grid) / max(max_grid - min_grid, 1) * 2.0 - 1.0
        
        observation = {
            'image': obs_image.astype(np.float32),
            'remaining_cells': np.array([remaining_cells_norm], dtype=np.float32),
            'revealed_ratio': np.array([revealed_ratio_norm], dtype=np.float32),
            'mine_density_norm': np.array([mine_density_norm], dtype=np.float32),
            'steps_per_cell': np.array([steps_per_cell_norm], dtype=np.float32),
            'grid_size_norm': np.array([grid_size_norm], dtype=np.float32),
        }
        
        return observation

    def _get_info(self):
        """Returns info about current state"""
        steps_per_cell = self.steps / max(self.safe_cells_revealed, 1)
        progress = (self.safe_cells_revealed / self.safe_cells_total) * 100.0
        
        return {
            'score': self.score,
            'steps': self.steps,
            'safe_cells_revealed': self.safe_cells_revealed,
            'safe_cells_total': self.safe_cells_total,
            'grid_size': self.grid_size,
            'num_mines': self.num_mines,
            'steps_per_cell': steps_per_cell,
            'total_reward': self.total_reward,
            'progress': progress,
            'difficulty_multiplier': self.difficulty_multiplier,
            'milestones_achieved': len(self.milestones_achieved),
            'invalid_action': False,  # Default value
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render the game using pygame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            cell_size = config['environment']['cell_size']
            window_size = self.grid_size * cell_size
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Minesweeper AI")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        cell_size = config['environment']['cell_size']
        canvas = pygame.Surface((self.grid_size * cell_size, self.grid_size * cell_size))
        canvas.fill((192, 192, 192))  # Gray background
        
        # Draw cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = c * cell_size
                y = r * cell_size
                
                if self.revealed[r, c]:
                    if self.mine_grid[r, c]:
                        # Mine (red)
                        pygame.draw.rect(canvas, (255, 0, 0), (x, y, cell_size, cell_size))
                    else:
                        # Safe cell (white)
                        pygame.draw.rect(canvas, (255, 255, 255), (x, y, cell_size, cell_size))
                        
                        # Draw number
                        num = self.number_grid[r, c]
                        if num > 0:
                            font = pygame.font.Font(None, cell_size // 2)
                            colors = [
                                (0, 0, 255),    # 1: blue
                                (0, 128, 0),    # 2: green
                                (255, 0, 0),    # 3: red
                                (0, 0, 128),    # 4: dark blue
                                (128, 0, 0),    # 5: dark red
                                (0, 128, 128),  # 6: cyan
                                (0, 0, 0),      # 7: black
                                (128, 128, 128) # 8: gray
                            ]
                            color = colors[num - 1] if num <= 8 else (0, 0, 0)
                            text = font.render(str(num), True, color)
                            text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
                            canvas.blit(text, text_rect)
                else:
                    # Unrevealed (dark gray)
                    pygame.draw.rect(canvas, (128, 128, 128), (x, y, cell_size, cell_size))
                
                # Grid lines
                pygame.draw.rect(canvas, (64, 64, 64), (x, y, cell_size, cell_size), 1)
        
        self.screen = canvas
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def make_env(render_mode=None, grid_size=None):
    """Factory function for creating environments"""
    def _init():
        return MinesweeperEnv(render_mode=render_mode, grid_size=grid_size)
    return _init