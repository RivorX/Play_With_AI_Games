import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation as FrameStack
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

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, grid_size=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size if grid_size is not None else config['environment']['grid_size']
        self.render_mode = render_mode
        # Stała przestrzeń obserwacji: 4 ramki x 16x16 x 6 kanałów + 4 ostatnie kierunki
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FIXED_OBS_SIZE, FIXED_OBS_SIZE, 6 + 4),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(config['environment']['action_space']['n'])
        self.steps_without_food = 0
        self.state_counter = {}
        self.last_directions = collections.deque([0, 0, 0, 0], maxlen=4)  # historia 4 ostatnich kierunków
        self.reset()
        if render_mode == "human":
            pygame.init()
            self.screen = None
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])
        self.direction = 1  # Domyślny kierunek: dół
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        self.total_reward = 0
        self.steps_without_food = 0
        self.state_counter = {}
        self.last_directions = collections.deque([0, 0, 0, 0], maxlen=4)
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
        # Kanał 0: mapa (0-puste, 0.5-ciało, 0.75-jabłko, 1-głowa)
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
        active_state = zoom(active_state, zoom_factor, order=0)  # Zachowaj order=0 dla mapy

        # Kanał 1: znormalizowany dx
        direction_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        dx = 0.0
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            dx = (self.food[0] - head[0]) / max(1, self.grid_size - 1)
            direction_channel[head[0], head[1]] = dx
        direction_channel = zoom(direction_channel, zoom_factor, order=1)  # order=1 dla dx

        # Kanał 2: znormalizowany dy
        direction_channel_y = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        dy = 0.0
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            dy = (self.food[1] - head[1]) / max(1, self.grid_size - 1)
            direction_channel_y[head[0], head[1]] = dy
        direction_channel_y = zoom(direction_channel_y, zoom_factor, order=1)  # order=1 dla dy

        # Kanał 3: kierunek
        dir_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            dir_value = (self.direction + 1) / 4.0  # Mapuje 0->0.25, 1->0.5, 2->0.75, 3->1.0
            dir_channel[head[0], head[1]] = dir_value
        else:
            print(f"Warning: Invalid head position for direction: {head}")
            dir_value = 0.0
        if self.grid_size == FIXED_OBS_SIZE:
            dir_channel = dir_channel
        else:
            dir_channel = zoom(dir_channel, zoom_factor, order=0)

        # Kanał 4: grid_size
        size_channel = np.full((self.grid_size, self.grid_size), self.grid_size / 16.0, dtype=np.float32)
        size_channel = zoom(size_channel, zoom_factor, order=1)  # order=1 dla grid_size

        # Kanał 5: odległość Manhattan
        distance_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                distance = abs(x - self.food[0]) + abs(y - self.food[1])
                distance_channel[x, y] = min(distance / self.grid_size, 1.0)
        distance_channel = zoom(distance_channel, zoom_factor, order=1)  # order=1 dla odległości

        # Nowe kanały: historia ostatnich 4 kierunków (każdy jako osobny kanał, wartość w pozycji głowy)
        direction_history_channels = []
        for d in self.last_directions:
            ch = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
                ch[head[0], head[1]] = (d + 1) / 4.0  # tak jak w dir_channel
            direction_history_channels.append(zoom(ch, zoom_factor, order=0))

        obs = np.stack([
            active_state, direction_channel, direction_channel_y, dir_channel, size_channel, distance_channel,
            *direction_history_channels
        ], axis=-1)
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
        head = self.snake[0]
        denom = max(1, self.grid_size - 1)
        prev_dist = (abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])) / (2 * denom)
        # Aktualizacja kierunku
        self.direction = (self.direction + (action - 1)) % 4
        self.last_directions.append(self.direction)
        head = self.snake[0] + DIRECTIONS[self.direction]
        self.steps += 1
        reward = 0
        self.done = False




        # Kara za każdy krok
        reward -= 0.02

        # Nagroda za przeżycie kolejnego kroku
        reward += 0.01

        # Kara za bycie blisko własnego ciała (sąsiadujące pola wokół głowy)
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
        reward -= 0.05 * close_body  # kara za każdy sąsiadujący segment ciała

        # Kara za bycie w pułapce (głowa otoczona przez ciało lub ścianę z 3 stron)
        trap_count = 0
        for n in neighbors:
            if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size:
                if list(n) in self.snake:
                    trap_count += 1
            else:
                trap_count += 1  # ściana też liczymy jako przeszkodę
        if trap_count >= 3:
            reward -= 1.0

        # Nagroda/kara za zmianę odległości do jabłka (łagodniejsza)
        new_dist = (abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])) / (2 * denom)
        if new_dist < prev_dist:
            reward += 0.1
        elif new_dist > prev_dist:
            reward -= 0.1

        # Kara za powtarzanie stanów (pętlenie się) - mocniejsza
        state_hash = (tuple(head.tolist()), self.direction, len(self.snake))
        self.state_counter[state_hash] = self.state_counter.get(state_hash, 0) + 1
        if self.state_counter[state_hash] > 3:
            reward -= 2.0

        # Kara za zbyt długie niejedzenie jabłka (łagodniejsza)
        max_steps_without_food = config['environment'].get('max_steps_without_food', 80) * self.grid_size
        if self.steps_without_food >= max_steps_without_food:
            self.done = True
            reward -= 10

        # Sprawdź kolizję lub przekroczenie limitu kroków
        max_steps = config['environment']['max_steps_factor'] * len(self.snake) * self.grid_size
        if self._is_collision(head) or self.steps > max_steps:
            self.done = True
            # Kara za szybką śmierć (im krótszy epizod, tym większa kara)
            death_penalty = -20 - max(0, 10 - self.steps * 0.1)
            reward = death_penalty
            self.state_counter = {}
        else:
            prev_length = len(self.snake)
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                self.food = self._place_food()
                # Nagroda bazowa za jabłko (mniejsza)
                reward = 20
                # Dodatkowa nagroda za wydłużenie węża
                reward += 1
                # Bonus za szybkie zdobycie jabłka (max +20)
                bonus = max(0, 20 - self.steps_without_food)
                reward += bonus
                self.steps_without_food = 0
                # Usuwamy dodatkową nagrodę za wydłużenie węża
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
        env = FrameStack(env, stack_size=4, padding_type="reset")
        return env
    return _init