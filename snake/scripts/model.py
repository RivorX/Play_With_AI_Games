import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
            high=3.0,
            shape=(MAX_GRID_SIZE, MAX_GRID_SIZE, 3),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        self.steps_without_food = 0
        self.reset()
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * SNAKE_SIZE, self.grid_size * SNAKE_SIZE))
            pygame.display.set_caption("Snake")
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
        return self._get_obs(), {}

    def _place_food(self):
        while True:
            food = np.random.randint(0, self.grid_size, size=2)
            if list(food) not in self.snake:
                return food

    def _get_obs(self):
        # Mapa dla modelu (z paddingiem do MAX_GRID_SIZE)
        active_state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Kanał 0: Mapa gry (0: puste, 1: ciało, 2: głowa, 3: jedzenie)
        for segment in self.snake:
            active_state[segment[0], segment[1], 0] = 1
        active_state[self.snake[0][0], self.snake[0][1], 0] = 2
        active_state[self.food[0], self.food[1], 0] = 3
        
        # Kanał 1: Kierunek ruchu (0-3 w komórce głowy)
        active_state[self.snake[0][0], self.snake[0][1], 1] = self.direction
        
        # Kanał 2: Odległość Manhattan do jedzenia (znormalizowana)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                distance = abs(x - self.food[0]) + abs(y - self.food[1])
                active_state[x, y, 2] = min(distance / self.grid_size, 3.0)
        
        # Padding do MAX_GRID_SIZE, puste pola (0) w kanale 0
        padded_state = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE, 3), dtype=np.float32)
        start_x = (MAX_GRID_SIZE - self.grid_size) // 2
        start_y = (MAX_GRID_SIZE - self.grid_size) // 2
        padded_state[start_x:start_x + self.grid_size, start_y:start_y + self.grid_size, :] = active_state
        
        return padded_state

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
        self.direction = (self.direction + (action - 1)) % 4
        head = self.snake[0] + DIRECTIONS[self.direction]
        self.steps += 1
        reward = 0
        self.done = False

        old_distance = abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - head[1])
        new_distance = abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])
        reward += 0.1 * (old_distance - new_distance)
        
        empty_cells = np.sum(self._get_render_state()[:, :, 0] == 0)
        total_cells = self.grid_size * self.grid_size
        reward -= 0.01 * (1 - empty_cells / total_cells)

        if self._is_collision(head) or self.steps > config['environment']['max_steps_factor'] * len(self.snake):
            self.done = True
            reward = -10
        else:
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                self.food = self._place_food()
                reward = 20
                self.steps_without_food = 0
            else:
                self.snake.pop()
                self.steps_without_food += 1
                reward += 0.05

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
        state = self._get_render_state()  # Użyj mapy bez paddingu
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
        return SnakeEnv(render_mode=render_mode)
    return _init