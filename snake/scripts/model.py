import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections
import pygame
import yaml
import os

# Wczytaj konfigurację
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Hiperparametry środowiska
SNAKE_SIZE = config['environment']['snake_size']
DIRECTIONS = np.array(config['environment']['directions'])
VIEWPORT_SIZE = config['environment']['viewport_size']

# PRE-COMPUTED STAŁE (obliczone raz na starcie programu)
# Kierunki jako wektory [x, y] dla sin/cos
DIRECTION_VECTORS = np.array([
    [0.0, -1.0],   # 0: góra    (sin(0°), -cos(0°))
    [1.0, 0.0],    # 1: prawo   (sin(90°), -cos(90°))
    [0.0, 1.0],    # 2: dół     (sin(180°), -cos(180°))
    [-1.0, 0.0]    # 3: lewo    (sin(270°), -cos(270°))
], dtype=np.float32)


class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, grid_size=16):
        super(SnakeEnv, self).__init__()
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        
        # Observation space dla RecurrentPPO
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=-1.0,
                high=config['environment']['observation_space']['high'],
                shape=(VIEWPORT_SIZE, VIEWPORT_SIZE, 1),
                dtype=config['environment']['observation_space']['dtype']
            ),
            'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'dx_head': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'dy_head': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'front_coll': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'left_coll': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'right_coll': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        self.snake = None
        self.snake_set = None
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
        
        # Tracking nagród
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
        
        # Cache dla half_view (używany często)
        self.half_view = VIEWPORT_SIZE // 2
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * SNAKE_SIZE, self.grid_size * SNAKE_SIZE))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])
        self.snake_set = {tuple(self.snake[0])}
        self.food = self._place_food()
        self.direction = np.random.randint(0, 4)
        self.steps = 0
        self.steps_without_food = 0
        self.total_reward = 0
        self.state_counter = {}
        self.done = False
        self.min_dist = float('inf')
        
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
        return np.array(available_positions[np.random.randint(0, len(available_positions))], dtype=np.int32)

    def _is_collision(self, head):
        # Kolizja ze ścianą
        if head[0] < 0 or head[0] >= self.grid_size or head[1] < 0 or head[1] >= self.grid_size:
            return True
        # Kolizja z ciałem węża
        head_tuple = tuple(head)
        current_head = tuple(self.snake[0])
        return head_tuple in self.snake_set and head_tuple != current_head

    def _get_potential_collision(self, direction):
        """Sprawdza kolizję w danym kierunku - ZOPTYMALIZOWANE"""
        head = self.snake[0]  # Nie konwertuj na array jeśli nie trzeba
        delta = DIRECTIONS[direction]
        new_head = [head[0] + delta[0], head[1] + delta[1]]
        
        # Sprawdź ściany (szybsza wersja)
        if new_head[0] < 0 or new_head[0] >= self.grid_size or \
           new_head[1] < 0 or new_head[1] >= self.grid_size:
            return 1.0
        
        # Sprawdź ciało
        return 1.0 if tuple(new_head) in self.snake_set else 0.0

    def _get_render_state(self):
        """Tylko dla renderowania - nie optymalizujemy"""
        state = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for i, segment in enumerate(self.snake):
            state[segment[0], segment[1], 0] = 1 if i == 0 else 2
        if self.food is not None:
            state[self.food[0], self.food[1], 0] = 3
        return state

    def _get_viewport_observation(self):
        """
        ULTRA ZOPTYMALIZOWANA wersja viewport 16x16
        Używa wektoryzacji NumPy zamiast pętli Python
        """
        head = self.snake[0]  # Lista [y, x]
        
        # Zakres viewport w grid coordinates
        y_start = head[0] - self.half_view
        y_end = head[0] + self.half_view
        x_start = head[1] - self.half_view
        x_end = head[1] + self.half_view
        
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
            
            # MEGA OPTYMALIZACJA: Konwersja deque → numpy array JEDNORAZOWO
            if len(self.snake) > 1:
                # Konwertuj całego węża na numpy array (O(n) raz zamiast O(n) razy)
                snake_array = np.array(self.snake, dtype=np.int32)
                
                # Filtruj segmenty w viewport (wektoryzacja)
                body_array = snake_array[1:]  # Bez głowy
                in_y_range = (body_array[:, 0] >= grid_y_start) & (body_array[:, 0] < grid_y_end)
                in_x_range = (body_array[:, 1] >= grid_x_start) & (body_array[:, 1] < grid_x_end)
                in_viewport = in_y_range & in_x_range
                
                if np.any(in_viewport):
                    body_coords = body_array[in_viewport]
                    # Przelicz na współrzędne viewport
                    vp_coords_y = body_coords[:, 0] - y_start
                    vp_coords_x = body_coords[:, 1] - x_start
                    # Ustaw wartości (fancy indexing - szybkie)
                    viewport[vp_coords_y, vp_coords_x] = 0.5
            
            # Głowa (zawsze w centrum jeśli widoczna)
            head_y, head_x = head[0], head[1]
            if grid_y_start <= head_y < grid_y_end and grid_x_start <= head_x < grid_x_end:
                viewport[head_y - y_start, head_x - x_start] = 1.0
            
            # Jedzenie
            if self.food is not None:
                fy, fx = self.food[0], self.food[1]
                if grid_y_start <= fy < grid_y_end and grid_x_start <= fx < grid_x_end:
                    viewport[fy - y_start, fx - x_start] = 0.75
        
        return viewport[:, :, np.newaxis]  # Dodaj wymiar kanału

    def _get_obs(self):
        """
        NAJSZYBSZA WERSJA - używa pre-computed stałych
        """
        viewport = self._get_viewport_observation()
        head = self.snake[0]  # Lista [y, x]
        
        # Wektoryzowane obliczenia pozycji jedzenia względem głowy
        food_viewport_x = self.food[1] - head[1] + self.half_view
        food_viewport_y = self.food[0] - head[0] + self.half_view
        
        # Normalizacja do [-1, 1]
        dx_viewport = (food_viewport_x - self.half_view) / self.half_view
        dy_viewport = (food_viewport_y - self.half_view) / self.half_view
        
        # ZERO OBLICZEŃ - używamy pre-computed lookup table!
        direction_vec = DIRECTION_VECTORS[self.direction]
        
        # Kolizje - już zoptymalizowane
        front_coll = self._get_potential_collision(self.direction)
        left_coll = self._get_potential_collision((self.direction - 1) % 4)
        right_coll = self._get_potential_collision((self.direction + 1) % 4)

        return {
            'image': viewport,
            'direction': direction_vec,  # Gotowy array z lookup table
            'dx_head': np.array([dx_viewport], dtype=np.float32),
            'dy_head': np.array([dy_viewport], dtype=np.float32),
            'front_coll': np.array([front_coll], dtype=np.float32),
            'left_coll': np.array([left_coll], dtype=np.float32),
            'right_coll': np.array([right_coll], dtype=np.float32)
        }

    def step(self, action):
        self.steps += 1
        self.steps_without_food += 1

        terminated = False
        truncated = False

        # Reset komponentów nagród
        step_rewards = {
            'food': 0.0,
            'distance_shaping': 0.0,
            'death_penalty': 0.0,
            'timeout_penalty': 0.0
        }

        # Zapisz poprzednią pozycję (jako lista, nie array!)
        prev_head = self.snake[0]
        # Manhattan distance bez abs() - używamy sumy różnic
        prev_dist = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])

        # Aktualizacja kierunku
        turn_penalty = 0.0
        if action == 1:  # skręć w lewo
            self.direction = (self.direction - 1) % 4
            turn_penalty = -0.05
        elif action == 2:  # skręć w prawo
            self.direction = (self.direction + 1) % 4
            turn_penalty = -0.05

        # Ruch węża - ZOPTYMALIZOWANE (bez konwersji na array)
        delta = DIRECTIONS[self.direction]
        head = [prev_head[0] + delta[0], prev_head[1] + delta[1]]

        # Uproszczony system nagród
        reward = -0.01 + turn_penalty

        # 1. KOLIZJA - ŚMIERĆ
        if self._is_collision(head):
            self.done = True
            terminated = True
            death_penalty = -10.0
            step_rewards['death_penalty'] = death_penalty
            reward = death_penalty

            obs = self._get_obs()

            for key in step_rewards:
                self.reward_components[key] += step_rewards[key]

            score = len(self.snake) - 1
            steps_per_apple = self.steps / score if score > 0 else self.steps

            info = {
                "score": score, 
                "snake_length": len(self.snake),
                "steps_per_apple": steps_per_apple,
                "total_reward": self.total_reward + reward, 
                "grid_size": self.grid_size,
                "reward_components": self.reward_components.copy(),
                "step_rewards": step_rewards
            }
            self.total_reward += reward
            return obs, reward, terminated, truncated, info

        # 2. Dodaj nową głowę
        self.snake.appendleft(head)
        self.snake_set.add(tuple(head))

        # 3. ZJEDZENIE JEDZENIA
        if head[0] == self.food[0] and head[1] == self.food[1]:
            food_reward = 20.0
            step_rewards['food'] = food_reward
            reward += food_reward

            self.food = self._place_food()
            self.steps_without_food = 0

            # Sprawdź wygraną
            if len(self.snake) == self.grid_size * self.grid_size:
                self.done = True
                terminated = True
                win_bonus = 50.0
                step_rewards['win_bonus'] = win_bonus
                reward += win_bonus
        else:
            # Usuń ogon
            tail = self.snake.pop()
            self.snake_set.discard(tuple(tail))

            # Distance shaping
            new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            dist_change = prev_dist - new_dist
            distance_reward = dist_change * 0.05
            step_rewards['distance_shaping'] = distance_reward
            reward += distance_reward

        # 5. TIMEOUT - adaptacyjny
        max_steps_without_food = max(100, self.grid_size * 10)
        if self.steps_without_food >= max_steps_without_food:
            self.done = True
            truncated = True
            timeout_penalty = -3.0
            step_rewards['timeout_penalty'] = timeout_penalty
            reward += timeout_penalty

        # 6. MAKSYMALNA DŁUGOŚĆ EPIZODU
        max_steps = 200 * self.grid_size
        if self.steps > max_steps:
            self.done = True
            truncated = True
            if step_rewards['timeout_penalty'] == 0.0:
                timeout_penalty = -3.0
                step_rewards['timeout_penalty'] = timeout_penalty
                reward += timeout_penalty

        # Aktualizuj totalne komponenty
        for key in step_rewards:
            self.reward_components[key] += step_rewards[key]

        self.total_reward += reward
        obs = self._get_obs()

        score = len(self.snake) - 1
        steps_per_apple = self.steps / score if score > 0 else self.steps

        info = {
            "score": score, 
            "snake_length": len(self.snake),
            "steps_per_apple": steps_per_apple,
            "total_reward": self.total_reward, 
            "grid_size": self.grid_size,
            "steps_without_food": self.steps_without_food,
            "reward_components": self.reward_components.copy(),
            "step_rewards": step_rewards
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((0, 0, 0))
        state = self._get_render_state()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                value = state[y, x, 0]
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
    """
    def _init():
        actual_grid_size = grid_size
        if actual_grid_size is None:
            min_size = config['environment']['min_grid_size']
            max_size = config['environment']['max_grid_size']
            actual_grid_size = np.random.randint(min_size, max_size + 1)
        env = SnakeEnv(render_mode=render_mode, grid_size=actual_grid_size)
        return env
    return _init