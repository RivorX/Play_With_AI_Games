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
        self.image_stack.clear()
        for _ in range(self.stack_size - 1):
            self.image_stack.append(np.zeros_like(obs['image']))
        self.image_stack.append(obs['image'])
        stacked_image = np.concatenate(list(self.image_stack), axis=-1)
        stacked_image = np.transpose(stacked_image, (2, 0, 1))
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
        stacked_image = np.concatenate(list(self.image_stack), axis=-1)
        stacked_image = np.transpose(stacked_image, (2, 0, 1))
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
        self.default_grid_size = grid_size if grid_size is not None else config['environment']['max_grid_size']
        self.grid_size = self.default_grid_size
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=config['environment']['observation_space']['low'],
                high=config['environment']['observation_space']['high'],
                shape=(FIXED_OBS_SIZE, FIXED_OBS_SIZE, 1),
                dtype=config['environment']['observation_space']['dtype']
            ),
            'direction': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'grid_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'dx_head': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'dy_head': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })
        self.snake = None
        self.food = None
        self.direction = None
        self.screen = None
        self.clock = None
        self.steps = 0
        self.steps_without_food = 0
        self.total_reward = 0
        self.state_counter = {}
        self.done = False
        self.min_dist = float('inf')  # Śledzenie minimalnego dystansu do jedzenia
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * SNAKE_SIZE, self.grid_size * SNAKE_SIZE))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.grid_size = self.default_grid_size if self.default_grid_size is not None else np.random.randint(
            config['environment']['min_grid_size'], config['environment']['max_grid_size'] + 1)
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])
        self.food = self._place_food()
        self.direction = np.random.randint(0, 4)
        self.steps = 0
        self.steps_without_food = 0
        self.total_reward = 0
        self.state_counter = {}
        self.done = False
        self.min_dist = float('inf')  # Resetuj minimalny dystans
        return self._get_obs(), {"score": 0, "total_reward": 0, "grid_size": self.grid_size}

    def _place_food(self):
        available_positions = [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)
                              if [i, j] not in self.snake]
        if not available_positions:
            return None
        return np.array(available_positions[np.random.randint(0, len(available_positions))])

    def _is_collision(self, head):
        return (head[0] < 0 or head[0] >= self.grid_size or
                head[1] < 0 or head[1] >= self.grid_size or
                head.tolist() in list(self.snake)[1:])

    def _get_render_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for i, segment in enumerate(self.snake):
            state[segment[0], segment[1], 0] = 1 if i == 0 else 2
        if self.food is not None:
            state[self.food[0], self.food[1], 0] = 3
        return state

    def _get_obs(self):
        active_state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for segment in self.snake:
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                active_state[segment[0], segment[1]] = 0.5
        head = self.snake[0]
        active_state[head[0], head[1]] = 1.0
        if self.food is not None:
            active_state[self.food[0], self.food[1]] = 0.75
        if self.grid_size != FIXED_OBS_SIZE:
            scale_factor = FIXED_OBS_SIZE / self.grid_size
            active_state = zoom(active_state, scale_factor, order=0)
        active_state = active_state[:, :, np.newaxis]
        denom = max(1, self.grid_size - 1)
        return {
            'image': active_state,
            'direction': np.array([(self.direction + 1) / 4.0], dtype=np.float32),
            'grid_size': np.array([self.grid_size / 16.0], dtype=np.float32),
            'dx_head': np.array([(self.food[0] - head[0]) / denom], dtype=np.float32),
            'dy_head': np.array([(self.food[1] - head[1]) / denom], dtype=np.float32)
        }

    def step(self, action):
        self.steps += 1
        prev_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        
        # Bazowa nagroda (brak kary za krok)
        reward = 0.0
        close_body = 0
        head = np.array(self.snake[0])
        neighbors = head + np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        for n in neighbors:
            if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size:
                if list(n) in self.snake:
                    close_body += 1
        reward -= 0.01 * close_body  # Zmniejszona kara za bliskość ciała

        trap_count = 0
        for n in neighbors:
            if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size:
                if list(n) in self.snake:
                    trap_count += 1
            else:
                trap_count += 1
        if trap_count >= 3:
            reward -= 0.5  # Zmniejszona kara za pułapkę

        # Aktualizacja kierunku przed ruchem
        if action == 0:  # kontynuuj
            pass
        elif action == 1:  # skręć w lewo
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # skręć w prawo
            self.direction = (self.direction + 1) % 4

        # Ruch węża
        head = np.array(self.snake[0])
        if self.direction == 0:  # lewo
            head[1] -= 1
        elif self.direction == 1:  # dół
            head[0] += 1
        elif self.direction == 2:  # prawo
            head[1] += 1
        elif self.direction == 3:  # góra
            head[0] -= 1

        # Oblicz nowy dystans Manhattan
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        
        # Nagroda za zbliżenie się do jedzenia
        if new_dist < prev_dist:
            reward += 0.1  # Nagroda za zmniejszenie dystansu
            if new_dist < self.min_dist:
                self.min_dist = new_dist
                reward += 1.0  # Dodatkowa nagroda za pobicie rekordu zbliżenia

        # Kara za powtarzanie stanów
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
                self.min_dist = float('inf')  # Resetuj minimalny dystans po zjedzeniu
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