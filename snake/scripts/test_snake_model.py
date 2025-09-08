import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import collections
import pygame

# Hiperparametry środowiska
GRID_SIZE = 10
SNAKE_SIZE = 20  # Dla wizualizacji w pikselach
DIRECTIONS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # Góra, prawo, dół, lewo

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        # Przestrzeń akcji: 0-lewo, 1-prosto, 2-prawo
        self.action_space = spaces.Discrete(3)
        # Przestrzeń stanów: 11 cech binarnych
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.reset()
        # Opcjonalna wizualizacja
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((GRID_SIZE * SNAKE_SIZE, GRID_SIZE * SNAKE_SIZE))
            pygame.display.set_caption("Snake PPO Test")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = collections.deque([[self.grid_size // 2, self.grid_size // 2]])
        self.direction = 1  # Start: prawo
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        self.total_reward = 0
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
            self._is_collision(point_f),  # Zagrożenie prosto
            self._is_collision(point_l),  # Zagrożenie lewo
            self._is_collision(point_r),  # Zagrożenie prawo
            dir_l, dir_r, dir_u, dir_d,  # Kierunek
            self.food[0] < head[0],  # Jabłko lewo
            self.food[0] > head[0],  # Jabłko prawo
            self.food[1] < head[1],  # Jabłko góra
            self.food[1] > head[1],  # Jabłko dół
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

        if self._is_collision(head) or self.steps > 100 * len(self.snake):
            self.done = True
            reward = -10
        else:
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                self.food = self._place_food()
                reward = 10
            else:
                self.snake.pop()

        self.total_reward += reward
        obs = self._get_obs()
        info = {"score": len(self.snake) - 1, "total_reward": self.total_reward}

        return obs, reward, self.done, False, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((0, 0, 0))  # Czarne tło
        # Rysuj węża
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (segment[0] * SNAKE_SIZE, segment[1] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
        # Rysuj jabłko
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (self.food[0] * SNAKE_SIZE, self.food[1] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
        pygame.display.flip()
        self.clock.tick(10)  # FPS

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

def test_model(model_path, render_mode="human", episodes=5):
    """
    Testuje zapisany model PPO w środowisku Snake.
    
    Args:
        model_path (str): Ścieżka do pliku modelu (.zip).
        render_mode (str): "human" dla wizualizacji, None bez wizualizacji.
        episodes (int): Liczba epizodów do przetestowania.
    """
    print(f"Ładowanie modelu z: {model_path}")
    env = SnakeEnv(render_mode=render_mode)
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Nie znaleziono modelu w {model_path}. Upewnij się, że plik istnieje.")
        return

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\nEpizod {episode + 1}")
        while not done:
            if render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            action, _ = model.predict(obs, deterministic=True)  # Deterministyczne akcje dla testu
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if render_mode == "human":
                env.render()
                pygame.time.wait(50)  # Płynniejsza wizualizacja
            print(f"Krok: {steps}, Wynik: {info['score']}, Nagroda: {info['total_reward']}")
        print(f"Epizod {episode + 1} zakończony. Łączna nagroda: {total_reward}, Długość węża: {info['score']}")
    env.close()

if __name__ == "__main__":
    # Ścieżka do modelu
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'snake_ppo_model.zip')
    # Alternatywnie, użyj najlepszego modelu:
    # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best_snake_ppo_model.zip')
    
    test_model(model_path, render_mode="human", episodes=5)