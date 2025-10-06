import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections
import pygame
import yaml
import os
import math

# Wczytaj konfigurację
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Hiperparametry środowiska
SNAKE_SIZE = config['environment']['snake_size']
DIRECTIONS = np.array(config['environment']['directions'])
VIEWPORT_SIZE = config['environment']['viewport_size']  # Rozmiar viewport z konfiguracji

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, grid_size=16):
        super(SnakeEnv, self).__init__()
        self.render_mode = render_mode
        self.grid_size = grid_size  # Rozmiar siatki z parametru (może być losowy)
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        
        # Observation space dla RecurrentPPO (bez framestacking)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=-1.0,  # Ściany mają wartość -1
                high=config['environment']['observation_space']['high'],
                shape=(VIEWPORT_SIZE, VIEWPORT_SIZE, 1),
                dtype=config['environment']['observation_space']['dtype']
            ),
            'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # sin/cos
            'dx_head': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'dy_head': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'front_coll': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'left_coll': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'right_coll': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.snake = None
        self.snake_set = None  # Set dla O(1) lookup kolizji
        self.food = None
        self.direction = None
        self.screen = None
        self.clock = None
        self.steps = 0
        self.steps_without_food = 0
        self.total_reward = 0
        self.state_counter = {}
        self.done = False
        self.min_dist = float('inf')
        
        # === TRACKING NAGRÓD DLA DEBUGOWANIA ===
        self.reward_components = {
            'food': 0.0,
            'speed_bonus': 0.0,
            'distance_shaping': 0.0,
            'progress_bonus': 0.0,
            'time_penalty': 0.0,
            'loop_penalty': 0.0,
            'exploration_penalty': 0.0,
            'death_penalty': 0.0,
            'timeout_penalty': 0.0,
            'win_bonus': 0.0
        }
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * SNAKE_SIZE, self.grid_size * SNAKE_SIZE))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # grid_size pochodzi z __init__ - może być losowy
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])
        self.snake_set = {tuple(self.snake[0])}  # Set dla szybkiego lookup
        self.food = self._place_food()
        self.direction = np.random.randint(0, 4)
        self.steps = 0
        self.steps_without_food = 0
        self.total_reward = 0
        self.state_counter = {}
        self.done = False
        self.min_dist = float('inf')
        
        # Reset trackingu nagród
        for key in self.reward_components:
            self.reward_components[key] = 0.0
        
        return self._get_obs(), {
            "score": 0, 
            "total_reward": 0, 
            "grid_size": self.grid_size,
            "reward_components": self.reward_components.copy()
        }

    def _place_food(self):
        available_positions = [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)
                              if (i, j) not in self.snake_set]
        if not available_positions:
            return None
        return np.array(available_positions[np.random.randint(0, len(available_positions))])

    def _is_collision(self, head):
        # Kolizja ze ścianą
        if head[0] < 0 or head[0] >= self.grid_size or head[1] < 0 or head[1] >= self.grid_size:
            return True
        # Kolizja z ciałem węża (pomijając głowę - index 0)
        head_tuple = tuple(head)
        current_head = tuple(self.snake[0])
        return head_tuple in self.snake_set and head_tuple != current_head

    def _get_potential_collision(self, direction):
        head = np.array(self.snake[0])
        delta = DIRECTIONS[direction]
        new_head = head + delta
        return 1.0 if self._is_collision(new_head) else 0.0

    def _get_render_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for i, segment in enumerate(self.snake):
            state[segment[0], segment[1], 0] = 1 if i == 0 else 2
        if self.food is not None:
            state[self.food[0], self.food[1], 0] = 3
        return state

    def _get_viewport_observation(self):
        """
        MEGA ZOPTYMALIZOWANA wersja viewport 16x16.
        Unika kosztownej konwersji deque→list→array przy każdym wywołaniu.
        """
        half_view = VIEWPORT_SIZE // 2
        head = np.array(self.snake[0])
        
        # Zakres viewport w grid coordinates
        y_start, y_end = head[0] - half_view, head[0] + half_view
        x_start, x_end = head[1] - half_view, head[1] + half_view
        
        # Przecięcie z granicami planszy
        grid_y_start = max(0, y_start)
        grid_y_end = min(self.grid_size, y_end)
        grid_x_start = max(0, x_start)
        grid_x_end = min(self.grid_size, x_end)
        
        # Odpowiadające pozycje w viewport
        vp_y_start = grid_y_start - y_start
        vp_y_end = vp_y_start + (grid_y_end - grid_y_start)
        vp_x_start = grid_x_start - x_start
        vp_x_end = vp_x_start + (grid_x_end - grid_x_start)
        
        # Inicjalizacja viewport (ściany = -1.0)
        viewport = np.full((VIEWPORT_SIZE, VIEWPORT_SIZE), -1.0, dtype=np.float32)
        
        # Wypełnij obszar planszy
        if grid_y_end > grid_y_start and grid_x_end > grid_x_start:
            viewport[vp_y_start:vp_y_end, vp_x_start:vp_x_end] = 0.0
            
            # MEGA OPTYMALIZACJA: Iteruj bezpośrednio po deque zamiast konwertować
            # To unika kosztownej alokacji list(self.snake) przy każdym kroku
            if len(self.snake) > 1:
                # Pre-allocate dla body segments (max możliwa długość)
                body_coords = []
                for i, segment in enumerate(self.snake):
                    if i == 0:  # Pomijamy głowę
                        continue
                    gy, gx = segment
                    # Sprawdź czy w viewport
                    if grid_y_start <= gy < grid_y_end and grid_x_start <= gx < grid_x_end:
                        vp_y = gy - y_start
                        vp_x = gx - x_start
                        body_coords.append([vp_y, vp_x])
                
                # Wektoryzowane ustawienie ciała (jeśli są segmenty)
                if body_coords:
                    body_coords = np.array(body_coords, dtype=np.int32)
                    viewport[body_coords[:, 0], body_coords[:, 1]] = 0.5
            
            # Głowa (zawsze w centrum jeśli widoczna)
            head_y, head_x = head
            if grid_y_start <= head_y < grid_y_end and grid_x_start <= head_x < grid_x_end:
                viewport[head_y - y_start, head_x - x_start] = 1.0
            
            # Jedzenie
            if self.food is not None:
                fy, fx = self.food
                if grid_y_start <= fy < grid_y_end and grid_x_start <= fx < grid_x_end:
                    viewport[fy - y_start, fx - x_start] = 0.75
        
        return viewport[:, :, np.newaxis]  # Dodaj wymiar kanału

    def _get_obs(self):
        viewport = self._get_viewport_observation()
        head = np.array(self.snake[0])
        
        half_view = VIEWPORT_SIZE // 2
        
        # FIX 1: Poprawione osie X/Y - food[1] to X, food[0] to Y
        food_viewport_x = self.food[1] - head[1] + half_view
        food_viewport_y = self.food[0] - head[0] + half_view
        
        # FIX 2: Prawidłowe przypisanie - X do dx, Y do dy
        dx_viewport = (food_viewport_x - half_view) / half_view  # -1..1
        dy_viewport = (food_viewport_y - half_view) / half_view  # -1..1
        
        # FIX 3: Lepsze kodowanie kierunku (sin/cos)
        # direction: 0=góra, 1=prawo, 2=dół, 3=lewo
        # Rotacja: 0° = góra = (0, -1), 90° = prawo = (1, 0), itd.
        angle = self.direction * math.pi / 2
        direction_x = math.sin(angle)  # sin daje X (lewo-prawo)
        direction_y = -math.cos(angle)  # -cos daje Y (góra-dół)
        
        # Oblicz potencjalne kolizje
        front_coll = self._get_potential_collision(self.direction)
        left_coll = self._get_potential_collision((self.direction - 1) % 4)
        right_coll = self._get_potential_collision((self.direction + 1) % 4)

        return {
            'image': viewport,
            'direction': np.array([direction_x, direction_y], dtype=np.float32),
            'dx_head': np.array([dx_viewport], dtype=np.float32),
            'dy_head': np.array([dy_viewport], dtype=np.float32),
            'front_coll': np.array([front_coll], dtype=np.float32),
            'left_coll': np.array([left_coll], dtype=np.float32),
            'right_coll': np.array([right_coll], dtype=np.float32)
        }

    def step(self, action):
        self.steps += 1
        self.steps_without_food += 1
        
        # Reset komponentów nagród dla tego kroku
        step_rewards = {
            'food': 0.0,
            'distance_shaping': 0.0,
            'death_penalty': 0.0,
            'timeout_penalty': 0.0
        }
        
        # Zapisz poprzednią pozycję i dystans
        prev_head = np.array(self.snake[0])
        prev_dist = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])
        
        # Aktualizacja kierunku przed ruchem
        turn_penalty = 0.0
        if action == 1:  # skręć w lewo
            self.direction = (self.direction - 1) % 4
            turn_penalty = -0.05  # kara za skręt
        elif action == 2:  # skręć w prawo
            self.direction = (self.direction + 1) % 4
            turn_penalty = -0.05  # kara za skręt
        
        # Ruch węża
        head = prev_head + DIRECTIONS[self.direction]
        
        # === UPROSZCZONY SYSTEM NAGRÓD ===
        reward = -0.01  # Mała kara za krok
        reward += turn_penalty  # kara za skręt
        
        # 1. KOLIZJA - ŚMIERĆ
        if self._is_collision(head):
            self.done = True
            death_penalty = -10.0
            step_rewards['death_penalty'] = death_penalty
            reward = death_penalty
            
            obs = self._get_obs()
            
            # Aktualizuj totalne komponenty
            for key in step_rewards:
                self.reward_components[key] += step_rewards[key]
            
            info = {
                "score": len(self.snake) - 1, 
                "total_reward": self.total_reward + reward, 
                "grid_size": self.grid_size,
                "reward_components": self.reward_components.copy(),
                "step_rewards": step_rewards
            }
            self.total_reward += reward
            return obs, reward, self.done, False, info
        
        # 2. Dodaj nową głowę
        self.snake.appendleft(head.tolist())
        self.snake_set.add(tuple(head))
        
        # 3. ZJEDZENIE JEDZENIA
        if np.array_equal(head, self.food):
            # Nagroda za jedzenie
            food_reward = 20.0
            step_rewards['food'] = food_reward
            reward += food_reward
            
            # Reset
            self.food = self._place_food()
            self.steps_without_food = 0
            
            # Sprawdź wygraną (cała plansza wypełniona)
            if len(self.snake) == self.grid_size * self.grid_size:
                self.done = True
                win_bonus = 50.0
                step_rewards['win_bonus'] = win_bonus
                reward += win_bonus
        else:
            # Usuń ogon (nie zjedzono jedzenia)
            tail = self.snake.pop()
            self.snake_set.discard(tuple(tail))
            
            # 4. DISTANCE SHAPING (subtelny)
            new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            dist_change = prev_dist - new_dist
            distance_reward = dist_change * 0.05
            step_rewards['distance_shaping'] = distance_reward
            reward += distance_reward
        
        # 5. TIMEOUT - adaptacyjny
        max_steps_without_food = max(100, self.grid_size * 10)
        if self.steps_without_food >= max_steps_without_food:
            self.done = True
            timeout_penalty = -3.0
            step_rewards['timeout_penalty'] = timeout_penalty
            reward += timeout_penalty
        
        # 6. MAKSYMALNA DŁUGOŚĆ EPIZODU
        max_steps = 200 * self.grid_size
        if self.steps > max_steps:
            self.done = True
            if step_rewards['timeout_penalty'] == 0.0:
                timeout_penalty = -3.0
                step_rewards['timeout_penalty'] = timeout_penalty
                reward += timeout_penalty
        
        # Aktualizuj totalne komponenty
        for key in step_rewards:
            self.reward_components[key] += step_rewards[key]
        
        self.total_reward += reward
        obs = self._get_obs()
        info = {
            "score": len(self.snake) - 1, 
            "total_reward": self.total_reward, 
            "grid_size": self.grid_size,
            "steps_without_food": self.steps_without_food,
            "reward_components": self.reward_components.copy(),
            "step_rewards": step_rewards
        }
        
        return obs, reward, self.done, False, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((0, 0, 0))
        state = self._get_render_state()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                value = state[x, y, 0]
                if value == 1:
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     (x * SNAKE_SIZE, y * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
                elif value == 2:
                    pygame.draw.rect(self.screen, (0, 200, 0),
                                     (x * SNAKE_SIZE, y * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
                elif value == 3:
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     (x * SNAKE_SIZE, y * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


def make_env(render_mode=None, grid_size=None):
    """
    Tworzy środowisko Snake - BEZ SequenceWrapper i BEZ DictFrameStack
    RecurrentPPO obsługuje sekwencje i pamięć sam
    
    Args:
        render_mode: Tryb renderowania ('human' lub None)
        grid_size: Rozmiar siatki - jeśli None, losowany z zakresu config (min_grid_size, max_grid_size)
    """
    def _init():
        # Losuj grid_size z zakresu config jeśli nie podano
        actual_grid_size = grid_size
        if actual_grid_size is None:
            min_size = config['environment']['min_grid_size']
            max_size = config['environment']['max_grid_size']
            actual_grid_size = np.random.randint(min_size, max_size + 1)
        env = SnakeEnv(render_mode=render_mode, grid_size=actual_grid_size)
        return env
    return _init