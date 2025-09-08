import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections
import pygame
import yaml
import os

# Wczytaj konfigurację względem lokalizacji pliku model.py
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Hiperparametry środowiska
GRID_SIZE = config['environment']['grid_size']
SNAKE_SIZE = config['environment']['snake_size']
DIRECTIONS = np.array(config['environment']['directions'])

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        self.observation_space = spaces.Box(
            low=config['environment']['observation_space']['low'],
            high=config['environment']['observation_space']['high'],
            shape=tuple(config['environment']['observation_space']['shape']),
            dtype=np.dtype(config['environment']['observation_space']['dtype'])
        )
        self.steps_without_food = 0
        self.reset()
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((GRID_SIZE * SNAKE_SIZE, GRID_SIZE * SNAKE_SIZE))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])
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
        head = self.snake[0]
        point_l = head + DIRECTIONS[(self.direction - 1) % 4]
        point_r = head + DIRECTIONS[(self.direction + 1) % 4]
        point_f = head + DIRECTIONS[self.direction]
        dir_l = (self.direction == 3)
        dir_r = (self.direction == 1)
        dir_u = (self.direction == 0)
        dir_d = (self.direction == 2)
        state = [
            self._is_collision(point_f),
            self._is_collision(point_l),
            self._is_collision(point_r),
            dir_l, dir_r, dir_u, dir_d,
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
        ]
        return np.array(state, dtype=np.float32)

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
        if self._is_collision(head) or self.steps > config['environment']['max_steps_factor'] * len(self.snake):
            # Kolizja lub przekroczono maksymalną liczbę kroków
            self.done = True
            reward = -50
        else:
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                # Zjedzono jedzenie
                self.food = self._place_food()
                reward = 10
                self.steps_without_food = 0
            else:
                # Przesunięcie węża
                self.snake.pop()
                self.steps_without_food += 1
            if len(self.snake) == self.grid_size * self.grid_size:
                self.done = True
                reward += 100
        self.total_reward += reward
        # Dodaj karę za brak zebrania jabłka przez dłuższy czas
        if self.steps_without_food > config['environment']['max_steps_without_food']:
            reward -= 0.01 / len(self.snake)
        obs = self._get_obs()
        info = {"score": len(self.snake) - 1, "total_reward": self.total_reward}
        return obs, reward, self.done, False, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (segment[0] * SNAKE_SIZE, segment[1] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (self.food[0] * SNAKE_SIZE, self.food[1] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

def make_env(render_mode=None):
    def _init():
        return SnakeEnv(render_mode=render_mode)
    return _init