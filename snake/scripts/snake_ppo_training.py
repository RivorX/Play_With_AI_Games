from stable_baselines3.common.callbacks import BaseCallback
# Callback do zapisu postępu treningu do CSV
class TrainProgressCallback(BaseCallback):
    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.header_written = False
        self.last_logged = 0

    def _on_step(self) -> bool:
        # Zapisuj tylko na końcu rollout (po epizodzie) i co 1000 timesteps
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            if self.num_timesteps - self.last_logged >= 1000:
                ep_rew_mean = self.model.ep_info_buffer and np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) or None
                ep_len_mean = self.model.ep_info_buffer and np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]) or None
                try:
                    write_header = not os.path.exists(self.csv_path)
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if write_header and os.path.exists(self.csv_path):
                            # Sprawdź, czy plik nie jest pusty, i dodaj nagłówek tylko raz
                            with open(self.csv_path, 'r') as f:
                                if not f.read(1):  # Pusty plik
                                    writer.writerow(['timesteps', 'mean_reward', 'mean_ep_length'])
                        writer.writerow([self.num_timesteps, ep_rew_mean, ep_len_mean])
                    self.last_logged = self.num_timesteps
                except Exception as e:
                    print(f"Błąd zapisu train_progress.csv: {e}")
        return True
import os
import time
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import collections
import pygame
import shutil
import csv
import matplotlib
matplotlib.use('Agg')  # Użyj backendu Agg, aby uniknąć problemów z wątkami i GUI
import matplotlib.pyplot as plt
import subprocess

# Hiperparametry środowiska
GRID_SIZE = 15
SNAKE_SIZE = 20
DIRECTIONS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.steps_without_food = 0
        self.reset()
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((GRID_SIZE * SNAKE_SIZE, GRID_SIZE * SNAKE_SIZE))
            pygame.display.set_caption("Snake PPO Optimized")
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
        if self._is_collision(head) or self.steps > 100 * len(self.snake):
            self.done = True
            reward = -10
        else:
            self.snake.appendleft(head.tolist())
            if np.array_equal(head, self.food):
                self.food = self._place_food()
                reward = 10
                self.steps_without_food = 0
            else:
                self.snake.pop()
                self.steps_without_food += 1
            if len(self.snake) == self.grid_size * self.grid_size:
                self.done = True
                reward += 100
        self.total_reward += reward
        # Dodaj karę za brak zebrania jabłka przez dłuższy czas
        if self.steps_without_food > 50:
            reward -= 0.1 / len(self.snake)
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

# Ścieżka do zapisu modelu i logów
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')
logs_dir = os.path.join(base_dir, '..', 'logs')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'snake_ppo_model.zip')
test_model_path = os.path.join(models_dir, 'test_model.zip')  # Tymczasowy plik dla testów
# EvalCallback will write best model as models_dir/best_model.zip; we'll copy it to model_path when found
test_log_path = os.path.join(logs_dir, 'test_log.txt')
test_csv_path = os.path.join(logs_dir, 'test_results.csv')
plot_path = os.path.join(logs_dir, 'training_progress.png')
test_progress_path = os.path.join(logs_dir, 'test_progress.png')

# Funkcja testowania w osobnym wątku z zapisem logów do pliku
def test_thread(model_path, test_model_path, log_path):
    current_model_timestamp = 0
    while True:
        env = SnakeEnv(render_mode="human")
        model = None
        try:
            new_timestamp = os.path.getmtime(model_path)
            if new_timestamp > current_model_timestamp:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Ładuję nowy model...\n")
                shutil.copy(model_path, test_model_path)  # Kopiuj do tymczasowego pliku
                model = PPO.load(test_model_path)
                current_model_timestamp = new_timestamp
            elif model is None:
                shutil.copy(model_path, test_model_path)
                model = PPO.load(test_model_path)
        except FileNotFoundError:
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Nie znaleziono modelu w {model_path}. Czekam 5 sekund...\n")
            time.sleep(5)
            env.close()
            continue

        obs, _ = env.reset()
        done = False
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Rozpoczynam test z wizualizacją...\n")
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Wynik: {info['score']}, Nagroda: {info['total_reward']}\n")
            # Append numeric results to CSV for plotting
            write_header = not os.path.exists(test_csv_path)
            try:
                with open(test_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(['timestamp', 'score', 'total_reward'])
                    writer.writerow([time.time(), info['score'], info['total_reward']])
            except Exception:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Błąd zapisu do CSV testów.\n")

            # Update the test progress plot
            try:
                timestamps = []
                scores = []
                rewards = []
                with open(test_csv_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) >= 3:
                            timestamps.append(float(row[0]))
                            scores.append(float(row[1]))
                            rewards.append(float(row[2]))

                if scores:
                    episodes = list(range(1, len(scores) + 1))
                    plt.figure(figsize=(8, 4))
                    plt.plot(episodes, scores, label='score', alpha=0.6)
                    plt.plot(episodes, rewards, label='total_reward', alpha=0.6)
                    window = max(1, min(50, len(scores)//10))
                    if window > 1:
                        ma = np.convolve(scores, np.ones(window)/window, mode='valid')
                        plt.plot(range(window, len(scores)+1), ma, label=f'score_ma({window})', color='red')
                    plt.xlabel('episode')
                    plt.ylabel('value')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(test_progress_path)
                    plt.close()
            except Exception:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Błąd podczas tworzenia wykresu testu.\n")
            pygame.time.wait(50)
        env.close()
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Epizod zakończony. Sprawdzam nowy model...\n")
        time.sleep(1)

def train(use_progress_bar=True):
    n_envs = 16
    n_steps = 8192
    expected_steps_per_iter = n_envs * n_steps
    env = make_vec_env(make_env(render_mode=None), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # Sprawdź, czy model istnieje i zapytaj o kontynuację
    model = None
    train_csv_path = os.path.join(logs_dir, 'train_progress.csv')
    test_csv_path = os.path.join(logs_dir, 'test_results.csv')
    if os.path.exists(model_path):
        choice = input(f"Model {model_path} istnieje. Kontynuować trening z tego modelu? [y/n]: ").strip().lower()
        if choice == 'y':
            print("Ładuję istniejący model...")
            model = PPO.load(model_path, env=env)
            total_timesteps = model.num_timesteps  # Synchronizacja z modelem
            print(f"Załadowano model z timesteps: {total_timesteps}")
        elif choice == 'n':
            print("Tworzę nowy model...")
            # Usuń pliki postępu treningu i testu, ale zrób backup
            backup_csv = train_csv_path + '.backup'
            if os.path.exists(train_csv_path):
                shutil.copy(train_csv_path, backup_csv)
            for p in [train_csv_path, test_csv_path]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    print(f"Nie udało się usunąć {p}: {e}")
            print(f"Stworzono kopię zapasową w {backup_csv}")
        else:
            print("Nieprawidłowy wybór. Kończę program.")
            env.close()
            return
    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0002,
            n_steps=n_steps,
            batch_size=4096,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.03,
            vf_coef=0.5,
            verbose=1,
            device='cuda',
            policy_kwargs={"net_arch": [256, 256, 128]}
        )

    # Callback do zatrzymania na plateau
    stop_on_plateau = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_env = make_vec_env(make_env(render_mode=None), n_envs=1, vec_env_cls=SubprocVecEnv)
    # Save best model into models_dir as 'best_model.zip' (EvalCallback default name),
    # then copy it to the canonical final filename snake_ppo_model.zip after each learning iteration.
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                                 log_path=models_dir, eval_freq=10000,
                                 deterministic=True, render=False,
                                 callback_after_eval=stop_on_plateau)

    # Pętla treningu z zapisem modelu co iterację
    total_timesteps = 0
    max_timesteps = 1_000_000
    steps_per_iteration = 16384
    test_started = False
    # Plik do zapisu postępu treningu
    # train_csv_path już ustawiony wyżej
    # Dodaj callback do zapisu postępu treningu
    train_progress_callback = TrainProgressCallback(train_csv_path)
    print(f"Oczekiwane kroki na iterację: {expected_steps_per_iter} (n_envs={n_envs} * n_steps={n_steps})")
    while total_timesteps < max_timesteps:
        # Ucz się i zbierz statystyki
        history = model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False, callback=[eval_callback, train_progress_callback], progress_bar=use_progress_bar)
        total_timesteps += steps_per_iteration
        model.save(model_path)
        backup_model = model_path + '.backup'
        shutil.copy(model_path, backup_model)  # Tworzenie backupu modelu
        print(f"Stworzono kopię zapasową modelu w {backup_model}")
        # Rzeczywista liczba kroków zebranych przez model
        print(f"Całkowite timesteps: {total_timesteps}, Model zapisany. (Rzeczywiste kroki w tej iteracji: {expected_steps_per_iter} lub więcej, zależnie od rolloutów)")
        # Jeśli EvalCallback zapisał lepszy model jako 'best_model.zip' w models_dir, skopiuj go
        best_model_candidate = os.path.join(models_dir, 'best_model.zip')
        if os.path.exists(best_model_candidate):
            try:
                shutil.copy(best_model_candidate, model_path)
                print(f"Zaktualizowano najlepszy model: {model_path}")
            except Exception as e:
                print(f"Nie udało się skopiować best_model.zip do {model_path}: {e}")
        # Automatycznie generuj wykres postępu treningu po każdej iteracji
        try:
            train_plot_script = os.path.join(os.path.dirname(__file__), 'plot_train_progress.py')
            subprocess.run(['python', train_plot_script], check=True)
        except Exception as e:
            print(f"Błąd podczas generowania wykresu treningu: {e}")
        if not test_started:
            threading.Thread(target=test_thread, args=(model_path, test_model_path, test_log_path), daemon=True).start()
            test_started = True

    env.close()
    eval_env.close()

if __name__ == "__main__":
    model = None  # Ensure model is defined in the global scope
    try:
        train(use_progress_bar=True)
    except KeyboardInterrupt:
        print("Przerwano trening. Zapisuję model...")
        if model is not None:
            model.save(model_path)
        else:
            print("Brak modelu do zapisania.")
        exit(0)
    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", test_log_path)