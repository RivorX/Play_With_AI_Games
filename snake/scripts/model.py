# model.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=None, viewport_size=None):
        super().__init__()
        
        # Grid size
        if grid_size is None:
            self.grid_size = np.random.randint(
                config['environment']['min_grid_size'],
                config['environment']['max_grid_size'] + 1
            )
        else:
            self.grid_size = grid_size
        
        # Viewport
        if viewport_size is None:
            self.viewport_size = config['environment']['viewport_size']
        else:
            self.viewport_size = viewport_size
        
        # ✅ NOWE: Oblicz difficulty multiplier dla tej planszy
        self.reward_config = config['environment'].get('reward_scaling', {})
        self.reward_scaling_enabled = self.reward_config.get('enable', True)
        
        if self.reward_scaling_enabled:
            min_grid = config['environment']['min_grid_size']
            max_grid = config['environment']['max_grid_size']
            min_mult = self.reward_config.get('min_difficulty_multiplier', 1.0)
            max_mult = self.reward_config.get('max_difficulty_multiplier', 2.0)
            
            # Liniowa interpolacja: min_grid → 1.0x, max_grid → 2.0x
            if max_grid > min_grid:
                progress = (self.grid_size - min_grid) / (max_grid - min_grid)
                self.difficulty_multiplier = min_mult + (max_mult - min_mult) * progress
            else:
                self.difficulty_multiplier = min_mult
        else:
            self.difficulty_multiplier = 1.0
        
        # Reward values
        self.base_food_reward = self.reward_config.get('base_food_reward', 10.0)
        self.base_death_penalty = self.reward_config.get('base_death_penalty', -10.0)
        self.milestones = self.reward_config.get('milestones', {})
        self.efficiency_config = self.reward_config.get('efficiency_bonus', {})
        
        # Tracking dla milestone
        self.milestones_achieved = set()
        
        # Action space: 0=lewo, 1=prosto, 2=prawo
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=config['environment']['observation_space']['low'],
                high=config['environment']['observation_space']['high'],
                shape=(self.viewport_size, self.viewport_size, 1),
                dtype=np.float32
            ),
            'direction': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'dx_head': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'dy_head': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'front_coll': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'left_coll': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'right_coll': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Snake state
        self.snake = None
        self.food = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.total_reward = 0.0
        
        # Max steps without food (skalowane)
        base_max_steps = config['environment']['max_steps_without_food']
        self.max_steps_without_food = base_max_steps * self.grid_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Losuj nowy grid_size jeśli nie był podany w __init__
        if options and 'grid_size' in options:
            self.grid_size = options['grid_size']
        
        # Przelicz difficulty multiplier dla nowej planszy
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
        
        # Initialize snake (centrum planszy) - TYLKO GŁOWA
        center = self.grid_size // 2
        self.snake = [(center, center)]  # ✅ Tylko głowa na starcie
        self.direction = 0  # UP
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.total_reward = 0.0
        
        # Przelicz max_steps_without_food
        base_max_steps = config['environment']['max_steps_without_food']
        self.max_steps_without_food = base_max_steps * self.grid_size
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """
        ✅ Progressive Reward Shaping
        - Nagrody skalują się z trudnością planszy (grid_size)
        - Milestone bonusy za % zajęcia planszy
        - Kara za śmierć proporcjonalna do postępu
        """
        self.steps += 1
        self.steps_since_food += 1
        
        # Zmiana kierunku (akcje: 0=lewo, 1=prosto, 2=prawo)
        if action == 0:  # Lewo
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # Prawo
            self.direction = (self.direction + 1) % 4
        # action == 1: prosto (bez zmiany)
        
        # Nowa pozycja głowy
        head_x, head_y = self.snake[0]
        dx, dy = config['environment']['directions'][self.direction]
        new_head = (head_x + dx, head_y + dy)
        
        # Inicjalizacja reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # ==================== KOLIZJA ====================
        # Sprawdź kolizję ze ścianą
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            terminated = True
        # Sprawdź kolizję z ciałem (bez ogona, bo ogon się przesuwa)
        elif new_head in self.snake[:-1]:
            terminated = True
        
        if terminated:
            # ✅ Kara za śmierć - skalowana z trudnością
            death_penalty = self.base_death_penalty * self.difficulty_multiplier
            reward = death_penalty
            
            info = self._get_info()
            info['termination_reason'] = 'collision'
            
            if self.render_mode == "human":
                self._render_frame()
            
            return self._get_obs(), reward, terminated, truncated, info
        
        # ==================== RUCH WĘŻA ====================
        self.snake.insert(0, new_head)
        ate_food = False
        
        # Sprawdź czy zjadł jedzenie
        if new_head == self.food:
            ate_food = True
            self.score += 1
            self.food = self._place_food()
            
            # ✅ NAGRODA ZA JEDZENIE - skalowana z trudnością planszy
            food_reward = self.base_food_reward * self.difficulty_multiplier
            reward = food_reward
            
            # ✅ MILESTONE BONUSY (progresywne, eksponencjalne)
            current_occupancy = len(self.snake) / (self.grid_size ** 2)
            
            for threshold_float, bonus in self.milestones.items():
                threshold = float(threshold_float)  # YAML może zwrócić string
                
                # Sprawdź czy osiągnął próg I jeszcze go nie dostał
                if current_occupancy >= threshold and threshold not in self.milestones_achieved:
                    reward += bonus * self.difficulty_multiplier  # Skaluj milestone z trudnością
                    self.milestones_achieved.add(threshold)
                    
                    # Debug print
                    print(f"🎯 MILESTONE! {int(threshold*100)}% planszy (grid={self.grid_size}x{self.grid_size}) → Bonus: +{bonus * self.difficulty_multiplier:.1f}")
                    
                    # Specjalne info dla 100% planszy
                    if threshold >= 0.99:
                        print(f"🏆🏆🏆 FULL BOARD! Grid={self.grid_size}x{self.grid_size} 🏆🏆🏆")
            
            # ✅ EFFICIENCY BONUS (małe steps_per_apple)
            if self.efficiency_config.get('enable', False):
                steps_per_apple = self.steps_since_food
                efficiency_threshold = self.efficiency_config.get('threshold', 10.0)
                efficiency_reward = self.efficiency_config.get('reward', 2.0)
                
                if steps_per_apple < efficiency_threshold:
                    reward += efficiency_reward * self.difficulty_multiplier
            
            # Reset kroków bez jedzenia
            self.steps_since_food = 0
        else:
            # Nie zjadł - usuń ogon
            self.snake.pop()
        
        # ==================== TIMEOUT ====================
        # Zbyt długo bez jedzenia
        if self.steps_since_food > self.max_steps_without_food:
            terminated = True
            reward += self.base_death_penalty * 0.5  # Mniejsza kara niż za kolizję
            info = self._get_info()
            info['termination_reason'] = 'timeout'
            
            if self.render_mode == "human":
                self._render_frame()
            
            return self._get_obs(), reward, terminated, truncated, info
        
        # ==================== MAX STEPS ====================
        max_steps = config['environment']['max_steps_factor'] * self.grid_size
        if self.steps >= max_steps:
            truncated = True
        
        # ==================== TRACKING ====================
        self.total_reward += reward
        
        # ==================== OBSERVATION & INFO ====================
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Zwraca obserwację (viewport + skalary)"""
        # Viewport - mapa ze środkiem na głowie
        head_x, head_y = self.snake[0]
        half_vp = self.viewport_size // 2
        
        # Inicjalizacja viewport (wszystkie pola jako tło)
        viewport = np.zeros((self.viewport_size, self.viewport_size), dtype=np.float32)
        
        # Oblicz zakres viewport w grid
        start_x = head_x - half_vp
        start_y = head_y - half_vp
        end_x = start_x + self.viewport_size
        end_y = start_y + self.viewport_size
        
        # Rysuj ściany (poza granicami planszy)
        for i in range(self.viewport_size):
            for j in range(self.viewport_size):
                grid_x = start_x + i
                grid_y = start_y + j
                
                # Ściana (poza planszą)
                if grid_x < 0 or grid_x >= self.grid_size or grid_y < 0 or grid_y >= self.grid_size:
                    viewport[i, j] = -1.0
        
        # Rysuj ciało węża (bez głowy)
        for segment in self.snake[1:]:
            seg_x, seg_y = segment
            vp_x = seg_x - start_x
            vp_y = seg_y - start_y
            
            if 0 <= vp_x < self.viewport_size and 0 <= vp_y < self.viewport_size:
                viewport[vp_x, vp_y] = 0.5
        
        # Rysuj głowę
        vp_head_x = head_x - start_x
        vp_head_y = head_y - start_y
        if 0 <= vp_head_x < self.viewport_size and 0 <= vp_head_y < self.viewport_size:
            viewport[vp_head_x, vp_head_y] = 1.0
        
        # Rysuj jedzenie
        food_x, food_y = self.food
        vp_food_x = food_x - start_x
        vp_food_y = food_y - start_y
        if 0 <= vp_food_x < self.viewport_size and 0 <= vp_food_y < self.viewport_size:
            viewport[vp_food_x, vp_food_y] = 0.75
        
        # Kanał (H, W, 1)
        channel_mapa = viewport
        obs_image = np.expand_dims(channel_mapa, axis=-1)
        
        # Skalary
        # Direction (sin, cos)
        angle = self.direction * np.pi / 2
        direction_sin = np.sin(angle)
        direction_cos = np.cos(angle)
        
        # Wektor do jedzenia (znormalizowany)
        dx_raw = self.food[0] - self.snake[0][0]
        dy_raw = self.food[1] - self.snake[0][1]
        max_dist = self.grid_size
        dx_norm = np.clip(dx_raw / max_dist, -1.0, 1.0)
        dy_norm = np.clip(dy_raw / max_dist, -1.0, 1.0)
        
        # Kolizje (front, left, right)
        front_coll = self._check_collision_in_direction(0)
        left_coll = self._check_collision_in_direction(-1)
        right_coll = self._check_collision_in_direction(1)
        
        observation = {
            'image': obs_image.astype(np.float32),
            'direction': np.array([direction_sin, direction_cos], dtype=np.float32),
            'dx_head': np.array([dx_norm], dtype=np.float32),
            'dy_head': np.array([dy_norm], dtype=np.float32),
            'front_coll': np.array([front_coll], dtype=np.float32),
            'left_coll': np.array([left_coll], dtype=np.float32),
            'right_coll': np.array([right_coll], dtype=np.float32)
        }
        
        return observation

    def _check_collision_in_direction(self, turn):
        """
        Sprawdza czy kolizja w danym kierunku
        turn: -1 (lewo), 0 (prosto), 1 (prawo)
        """
        new_dir = (self.direction + turn) % 4
        head_x, head_y = self.snake[0]
        dx, dy = config['environment']['directions'][new_dir]
        new_pos = (head_x + dx, head_y + dy)
        
        # Kolizja ze ścianą
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return 1.0
        
        # Kolizja z ciałem
        if new_pos in self.snake[:-1]:
            return 1.0
        
        return 0.0

    def _place_food(self):
        """Umieszcza jedzenie w losowym wolnym miejscu"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))
        
        if len(empty_cells) == 0:
            # Wąż wypełnił całą planszę!
            return self.snake[0]  # Fallback (nie powinno się zdarzyć)
        
        return empty_cells[np.random.randint(len(empty_cells))]

    def _get_info(self):
        """Zwraca info o aktualnym stanie"""
        steps_per_apple = self.steps / max(self.score, 1)
        map_occupancy = (len(self.snake) / (self.grid_size ** 2)) * 100.0
        
        return {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake),
            'grid_size': self.grid_size,
            'steps_per_apple': steps_per_apple,
            'total_reward': self.total_reward,
            'map_occupancy': map_occupancy,
            'difficulty_multiplier': self.difficulty_multiplier,
            'milestones_achieved': len(self.milestones_achieved)
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            snake_size = config['environment']['snake_size']
            window_size = self.grid_size * snake_size
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Snake RL")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        snake_size = config['environment']['snake_size']
        canvas = pygame.Surface((self.grid_size * snake_size, self.grid_size * snake_size))
        canvas.fill((0, 0, 0))  # Tło czarne
        
        # Rysuj jedzenie (czerwone)
        food_rect = pygame.Rect(
            self.food[1] * snake_size,
            self.food[0] * snake_size,
            snake_size,
            snake_size
        )
        pygame.draw.rect(canvas, (255, 0, 0), food_rect)
        
        # Rysuj węża (zielony)
        for segment in self.snake:
            seg_rect = pygame.Rect(
                segment[1] * snake_size,
                segment[0] * snake_size,
                snake_size,
                snake_size
            )
            pygame.draw.rect(canvas, (0, 255, 0), seg_rect)
        
        # Rysuj głowę (jasnozielony)
        head_rect = pygame.Rect(
            self.snake[0][1] * snake_size,
            self.snake[0][0] * snake_size,
            snake_size,
            snake_size
        )
        pygame.draw.rect(canvas, (150, 255, 150), head_rect)
        
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
    """Factory function dla tworzenia środowisk"""
    def _init():
        return SnakeEnv(render_mode=render_mode, grid_size=grid_size)
    return _init