import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections
import pygame
import yaml
import os
import torch
from scipy.ndimage import zoom

# Wczytaj konfigurację
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Hiperparametry środowiska
SNAKE_SIZE = config['environment']['snake_size']
DIRECTIONS = np.array(config['environment']['directions'])
FIXED_OBS_SIZE = 16  # Stały rozmiar obserwacji (16x16)

def set_grid_size(new_grid_size):
    global GRID_SIZE
    GRID_SIZE = new_grid_size

class DictFrameStack(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.image_stack = collections.deque(maxlen=stack_size)
        image_space = self.env.observation_space['image']
        # Poprawiona przestrzeń obserwacji dla stackowanego obrazu
        stacked_image_space = spaces.Box(
            low=image_space.low.min(),
            high=image_space.high.max(),
            shape=(image_space.shape[2] * stack_size, image_space.shape[0], image_space.shape[1]),  # [C, H, W]
            dtype=image_space.dtype
        )
        self.observation_space = spaces.Dict({
            'image': stacked_image_space,
            'direction': self.env.observation_space['direction'],
            'grid_size': self.env.observation_space['grid_size'],
            'dx_head': self.env.observation_space['dx_head'],
            'dy_head': self.env.observation_space['dy_head']
        })

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Inicjalizacja stosu ramek z zerami
        self.image_stack.clear()
        for _ in range(self.stack_size - 1):
            self.image_stack.append(np.zeros_like(obs['image']))
        self.image_stack.append(obs['image'])
        # Konwersja do formatu [C, H, W]
        stacked_image = np.concatenate(list(self.image_stack), axis=-1)  # [H, W, C*stack_size]
        stacked_image = np.transpose(stacked_image, (2, 0, 1))  # [C*stack_size, H, W]
        new_obs = {
            'image': stacked_image,
            'direction': obs['direction'],
            'grid_size': obs['grid_size'],
            'dx_head': obs['dx_head'],
            'dy_head': obs['dy_head']
        }
        return new_obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.image_stack.append(obs['image'])
        # Konwersja do formatu [C, H, W]
        stacked_image = np.concatenate(list(self.image_stack), axis=-1)  # [H, W, C*stack_size]
        stacked_image = np.transpose(stacked_image, (2, 0, 1))  # [C*stack_size, H, W]
        new_obs = {
            'image': stacked_image,
            'direction': obs['direction'],
            'grid_size': obs['grid_size'],
            'dx_head': obs['dx_head'],
            'dy_head': obs['dy_head']
        }
        return new_obs, reward, done, truncated, info

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, grid_size=None):
        super(SnakeEnv, self).__init__()
        self.render_mode = render_mode
        self.default_grid_size = grid_size
        self.grid_size = grid_size if grid_size is not None else np.random.randint(
            config['environment']['min_grid_size'], 
            config['environment']['max_grid_size'] + 1
        )
        # Przestrzeń obserwacji jako Dict: obraz (1 kanał: mapa) + skalary
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(FIXED_OBS_SIZE, FIXED_OBS_SIZE, 1),  # Tylko mapa [H, W, 1]
                dtype=np.float32
            ),
            'direction': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'grid_size': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'dx_head': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'dy_head': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        self.steps_without_food = 0
        self.state_counter = {}
        self.reset()
        if render_mode == "human":
            pygame.init()
            self.screen = None
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.default_grid_size is None:
            self.grid_size = np.random.randint(
                config['environment']['min_grid_size'], 
                config['environment']['max_grid_size'] + 1
            )
        head_x = np.random.randint(1, self.grid_size - 1)
        head_y = np.random.randint(1, self.grid_size - 1)
        self.snake = collections.deque([[head_x, head_y]])
        self.direction = 1
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        self.total_reward = 0
        self.steps_without_food = 0
        self.state_counter = {}
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.grid_size * SNAKE_SIZE, self.grid_size * SNAKE_SIZE))
            pygame.display.set_caption("Snake")
        obs = self._get_obs()
        return obs, {}

    def _place_food(self):
        while True:
            food = np.random.randint(0, self.grid_size, size=2)
            if list(food) not in self.snake:
                return food

    def _get_obs(self):
        active_state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for segment in list(self.snake)[1:]:
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                active_state[segment[0], segment[1]] = 0.5
        if 0 <= self.food[0] < self.grid_size and 0 <= self.food[1] < self.grid_size:
            active_state[self.food[0], self.food[1]] = 0.75
        head = self.snake[0]
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            active_state[head[0], head[1]] = 1.0
        else:
            print(f"Warning: Invalid head position: {head}")
        zoom_factor = FIXED_OBS_SIZE / self.grid_size
        active_state = zoom(active_state, zoom_factor, order=0)

        # Obliczanie dx_head i dy_head
        denom = max(1, self.grid_size - 1)
        dx_head = (self.food[0] - head[0]) / denom
        dy_head = (self.food[1] - head[1]) / denom

        image = active_state[..., np.newaxis]  # [H, W, 1]

        obs = {
            'image': image,
            'direction': np.array([(self.direction + 1) / 4.0], dtype=np.float32),
            'grid_size': np.array([self.grid_size / 16.0], dtype=np.float32),
            'dx_head': np.array([dx_head], dtype=np.float32),
            'dy_head': np.array([dy_head], dtype=np.float32)
        }
        return obs

    def _get_render_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for segment in self.snake:
            state[segment[0], segment[1], 0] = 1
        state[self.snake[0][0], self.snake[0][1], 0] = 2
        state[self.food[0], self.food[1], 0] = 3
        state[self.snake[0][0], self.snake[0][1], 1] = self.direction
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                distance = abs(x - self.food[0]) + abs(y - self.food[1])
                state[x, y, 2] = min(distance / self.grid_size, 1.0)
        return state

    def _is_collision(self, point):
        if point[0] < 0 or point[0] >= self.grid_size or point[1] < 0 or point[1] >= self.grid_size:
            return 1
        if list(point) in self.snake:
            return 1
        return 0

    def step(self, action):
        head = np.array(self.snake[0])
        denom = max(1, self.grid_size - 1)
        prev_dist = (abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])) / (2 * denom)
        next_direction = (self.direction + (action - 1)) % 4
        next_head = head + DIRECTIONS[next_direction]
        reward = 0
        if list(next_head) in list(self.snake)[1:]:
            reward -= 0.5
        self.direction = next_direction
        head = next_head
        self.steps += 1
        self.done = False

        reward -= 0.02
        reward += 0.02

        neighbors = [
            [head[0] + 1, head[1]],
            [head[0] - 1, head[1]],
            [head[0], head[1] + 1],
            [head[0], head[1] - 1]
        ]
        close_body = 0
        for n in neighbors:
            if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size:
                if list(n) in self.snake:
                    close_body += 1
        reward -= 0.02 * close_body

        trap_count = 0
        for n in neighbors:
            if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size:
                if list(n) in self.snake:
                    trap_count += 1
            else:
                trap_count += 1
        if trap_count >= 3:
            reward -= 1.0

        new_dist = (abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])) / (2 * denom)
        reward += 0.2 * (prev_dist - new_dist)

        state_hash = (tuple(head.tolist()), self.direction, len(self.snake))
        self.state_counter[state_hash] = self.state_counter.get(state_hash, 0) + 1
        reward -= 0.5 * self.state_counter[state_hash]

        max_steps_without_food = config['environment'].get('max_steps_without_food', 80) * self.grid_size
        if self.steps_without_food >= max_steps_without_food:
            self.done = True
            reward -= 10

        max_steps = config['environment']['max_steps_factor'] * len(self.snake) * self.grid_size
        if self._is_collision(head) or self.steps > max_steps:
            self.done = True
            death_penalty = -20 - max(0, 10 - self.steps * 0.1)
            reward = death_penalty
            self.state_counter = {}
        else:
            prev_length = len(self.snake)
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                self.food = self._place_food()
                reward = 20
                reward += 1
                bonus = max(0, 20 - self.steps_without_food)
                reward += bonus
                self.steps_without_food = 0
                self.state_counter = {}
            else:
                self.snake.pop()
                self.steps_without_food += 1

        self.total_reward += reward
        if len(self.snake) == self.grid_size * self.grid_size:
            self.done = True
            reward += 10 * self.grid_size
        obs = self._get_obs()
        info = {"score": len(self.snake) - 1, "total_reward": self.total_reward, "grid_size": self.grid_size}
        
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
    def _init():
        env = SnakeEnv(render_mode=render_mode, grid_size=grid_size)
        env = DictFrameStack(env, stack_size=4)
        return env
    return _init