import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation as FrameStack
import collections
import pygame
import yaml
import os
import torch

# Wczytaj konfigurację
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Hiperparametry środowiska
GRID_SIZE = config['environment']['grid_size']
SNAKE_SIZE = config['environment']['snake_size']
DIRECTIONS = np.array(config['environment']['directions'])

# Domyślny max grid_size dla paddingu
MAX_GRID_SIZE = config['environment']['grid_size']

def set_grid_size(new_grid_size):
    global GRID_SIZE
    GRID_SIZE = new_grid_size

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0.0,
            high=9.0,
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE, 4),  # Bazowa obserwacja: mapa, dx, dy, kierunek
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        self.steps_without_food = 0
        self.reset()
        if render_mode == "human":
            pygame.init()
            self.screen = None
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])  # Wąż: 1 kratka
        self.direction = 1
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        self.total_reward = 0
        self.steps_without_food = 0
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.grid_size * SNAKE_SIZE, self.grid_size * SNAKE_SIZE))
            pygame.display.set_caption("Snake")
        return self._get_obs(), {}

    def _place_food(self):
        while True:
            food = np.random.randint(0, self.grid_size, size=2)
            if list(food) not in self.snake:
                return food

    def _get_obs(self):
        # Kanał 0: 0-puste, 1-ciało, 2-jabłko, 9-głowa
        active_state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for segment in list(self.snake)[1:]:
            active_state[segment[0], segment[1]] = 1  # ciało
        active_state[self.food[0], self.food[1]] = 2  # jabłko
        head = self.snake[0]
        active_state[head[0], head[1]] = 9  # głowa

        # Kanał 1: kierunek do jabłka (współrzędne znormalizowane do [-1, 1])
        direction_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        dx = (self.food[0] - head[0]) / max(1, self.grid_size - 1)
        dy = (self.food[1] - head[1]) / max(1, self.grid_size - 1)
        direction_channel[head[0], head[1]] = dx  # x

        # Kanał 2: y
        direction_channel_y = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        direction_channel_y[head[0], head[1]] = dy

        # Kanał 3: kierunek węża (0-3 normalizowany do 0-1)
        dir_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        dir_channel[head[0], head[1]] = self.direction / 3.0  # Normalizacja

        # Padding do MAX_GRID_SIZE
        padded_state = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
        padded_dir = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
        padded_dir_y = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
        padded_dir_channel = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
        start_x = (MAX_GRID_SIZE - self.grid_size) // 2
        start_y = (MAX_GRID_SIZE - self.grid_size) // 2
        padded_state[start_x:start_x + self.grid_size, start_y:start_y + self.grid_size] = active_state
        padded_dir[start_x:start_x + self.grid_size, start_y:start_y + self.grid_size] = direction_channel
        padded_dir_y[start_x:start_x + self.grid_size, start_y:start_y + self.grid_size] = direction_channel_y
        padded_dir_channel[start_x:start_x + self.grid_size, start_y:start_y + self.grid_size] = dir_channel

        # Stwórz obserwację z 4 kanałami: [mapa, dx, dy, direction]
        obs = np.stack([padded_state, padded_dir, padded_dir_y, padded_dir_channel], axis=-1)
        return obs

    def _get_render_state(self):
        # Mapa dla renderowania (bez paddingu, tylko grid_size x grid_size)
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for segment in self.snake:
            state[segment[0], segment[1], 0] = 1
        state[self.snake[0][0], self.snake[0][1], 0] = 2
        state[self.food[0], self.food[1], 0] = 3
        state[self.snake[0][0], self.snake[0][1], 1] = self.direction
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                distance = abs(x - self.food[0]) + abs(y - self.food[1])
                state[x, y, 2] = min(distance / self.grid_size, 3.0)
        return state

    def _is_collision(self, point):
        if point[0] < 0 or point[0] >= self.grid_size or point[1] < 0 or point[1] >= self.grid_size:
            return 1
        if list(point) in self.snake:
            return 1
        return 0

    def step(self, action):
        head = self.snake[0]
        prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        self.direction = (self.direction + (action - 1)) % 4
        head = self.snake[0] + DIRECTIONS[self.direction]
        self.steps += 1
        reward = 0
        self.done = False

        # Kara za każdy krok
        reward -= 0.05

        # Nagroda/kara za zmianę odległości do jabłka
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        if new_dist < prev_dist:
            reward += 0.1
        elif new_dist > prev_dist:
            reward -= 0.1

        # Kara za zbyt długie niejedzenie jabłka
        if self.steps_without_food >= config['environment'].get('max_steps_without_food', 100):
            self.done = True
            reward -= 10

        # Sprawdź kolizję lub przekroczenie limitu kroków
        if self._is_collision(head) or self.steps > config['environment']['max_steps_factor'] * len(self.snake):
            self.done = True
            reward = -20
        else:
            prev_length = len(self.snake)
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                self.food = self._place_food()
                reward = 50
                self.steps_without_food = 0
                if len(self.snake) > prev_length:
                    reward += 5
            else:
                self.snake.pop()
                self.steps_without_food += 1

        self.total_reward += reward
        if len(self.snake) == self.grid_size * self.grid_size:
            self.done = True
            reward += 200
        obs = self._get_obs()
        info = {"score": len(self.snake) - 1, "total_reward": self.total_reward}
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

def make_env(render_mode=None):
    def _init():
        env = SnakeEnv(render_mode=render_mode)
        env = FrameStack(env, stack_size=4, padding_type="reset")  # Stack 4 ostatnich obserwacji
        return env
    return _init