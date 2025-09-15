import os
import time
import threading
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import subprocess
import sys
import yaml
from model import make_env, set_grid_size
from cnn import CustomCNN
import logging
import pickle

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Funkcja do resetowania plików logów kanałów
def reset_channel_logs():
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    for channel_name in ['mapa', 'dx', 'dy', 'kierunek', 'grid_size', 'odleglosc']:
        log_path = os.path.join(log_dir, f'training_{channel_name}.log')
        with open(log_path, 'w', encoding='utf-8'):
            pass  # nadpisz plik pustą zawartością

# Funkcja do inicjalizacji loggerów kanałów (zawsze po resecie plików)
def init_channel_loggers():
    loggers = {}
    log_dir = os.path.join(base_dir, 'logs')
    for i, channel_name in enumerate(['mapa', 'dx', 'dy', 'kierunek', 'grid_size', 'odleglosc']):
        log_path = os.path.join(log_dir, f'training_{channel_name}.log')
        logger = logging.getLogger(f'channel_{i}')
        handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        loggers[channel_name] = logger
    return loggers

enable_channel_logs = config.get('training', {}).get('enable_channel_logs', False)
channel_loggers = {}

# Funkcja do logowania obserwacji do osobnych plików dla każdego kanału
def log_observation(obs, channel_loggers, grid_size, step):
    if not enable_channel_logs:
        return
    latest_frame = obs[-1]
    kanały = {
        'mapa': latest_frame[:, :, 0],
        'dx': latest_frame[:, :, 1],
        'dy': latest_frame[:, :, 2],
        'kierunek': latest_frame[:, :, 3],
        'grid_size': latest_frame[:, :, 4],
        'odleglosc': latest_frame[:, :, 5] if latest_frame.shape[-1] > 5 else None
    }
    mapa = kanały['mapa']
    head_pos = np.where(mapa == 1.0)
    food_pos = np.where(mapa == 0.75)
    if len(head_pos[0]) > 0:
        head_x, head_y = head_pos[0][0], head_pos[1][0]
    else:
        head_x, head_y = -1, -1
    if len(food_pos[0]) > 0:
        food_x, food_y = food_pos[0][0], food_pos[1][0]
    else:
        food_x, food_y = -1, -1
    dx = kanały['dx'][head_x, head_y] if head_x >= 0 else None
    dy = kanały['dy'][head_x, head_y] if head_x >= 0 else None
    kierunek = kanały['kierunek'][head_x, head_y] if head_x >= 0 else None
    if head_x >= 0 and food_x >= 0:
        distance = abs(head_x - food_x) + abs(head_y - food_y)
    else:
        distance = float('inf')
    for channel_name, arr in kanały.items():
        logger = channel_loggers.get(channel_name)
        if logger is None or arr is None:
            continue
        logger.info(f"--- Obserwacja dla grid_size={grid_size}, krok={step} ---")
        logger.info(f"Kanał {channel_name}:\n{np.array_str(arr, precision=2, suppress_small=True, max_line_width=120)}")
        if channel_name == 'mapa':
            logger.info(f"Pozycja głowy: ({head_x}, {head_y}) | Pozycja jedzenia: ({food_x}, {food_y})")
        if channel_name == 'dx':
            logger.info(f"dx w pozycji głowy: {dx}")
        if channel_name == 'dy':
            logger.info(f"dy w pozycji głowy: {dy}")
        if channel_name == 'kierunek':
            logger.info(f"Kierunek węża (0-lewo,1-dół,2-prawo,3-góra): {kierunek}")
        if channel_name == 'odleglosc':
            logger.info(f"Odległość Manhattan: {distance}")
        logger.info("-" * 60)

# Harmonogram liniowy learning rate
def linear_schedule(initial_value, min_value=0.0):
    def func(progress_remaining):
        return min_value + (initial_value - min_value) * progress_remaining
    return func

# Callback do zapisu postępu treningu
class TrainProgressCallback(BaseCallback):
    def __init__(self, csv_path, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.initial_timesteps = initial_timesteps
        self.header_written = False
        self.last_logged = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            if (self.num_timesteps + self.initial_timesteps) - self.last_logged >= 1000:
                ep_rew_mean = self.model.ep_info_buffer and np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) or None
                ep_len_mean = self.model.ep_info_buffer and np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]) or None
                grid_size = self.training_env.get_attr('grid_size')[0]
                try:
                    write_header = not os.path.exists(self.csv_path)
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if write_header:
                            writer.writerow(['timesteps', 'mean_reward', 'mean_ep_length', 'grid_size'])
                        writer.writerow([self.num_timesteps + self.initial_timesteps, ep_rew_mean, ep_len_mean, grid_size])
                    self.last_logged = self.num_timesteps + self.initial_timesteps
                except Exception as e:
                    print(f"Błąd zapisu train_progress.csv: {e}")
        return True

# Custom EvalCallback do robienia wykresu co plot_interval walidacji
class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, callback_on_new_best=None, callback_after_eval=None, best_model_save_path=None,
                 log_path=None, eval_freq=10000, deterministic=True, render=False, verbose=1,
                 warn=True, n_eval_episodes=5, plot_interval=3, plot_script_path=None):
        super().__init__(eval_env, callback_on_new_best=callback_on_new_best, callback_after_eval=callback_after_eval,
                         best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq,
                         deterministic=deterministic, render=render, verbose=verbose, warn=warn,
                         n_eval_episodes=n_eval_episodes)
        self.eval_count = 0
        self.plot_interval = plot_interval
        self.plot_script_path = plot_script_path

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Wykonaj ewaluację
            super()._on_step()
            self.eval_count += 1
            if self.eval_count % self.plot_interval == 0:
                # Uruchom skrypt do generowania wykresu
                try:
                    subprocess.run([sys.executable, self.plot_script_path], check=True)
                    print(f"Wygenerowano wykres po {self.eval_count} walidacji.")
                except Exception as e:
                    print(f"Błąd podczas generowania wykresu: {e}")
        return True

# Funkcja testowania w osobnym wątku
def test_thread(model_path, test_model_path, log_path, test_csv_path, test_progress_path, grid_size, stop_event=None):
    current_model_timestamp = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            print("[TestThread] Otrzymano sygnał zakończenia. Kończę wątek testujący.")
            break
        set_grid_size(grid_size)
        env = make_env(render_mode="human", grid_size=grid_size)()
        model = None
        try:
            new_timestamp = os.path.getmtime(model_path)
            if new_timestamp > current_model_timestamp:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Ładuję nowy model dla grid_size={grid_size}...\n")
                shutil.copy(model_path, test_model_path)
                model = PPO.load(test_model_path)
                model.ent_coef = config['model']['ent_coef']
                model._setup_lr_schedule()
                current_model_timestamp = new_timestamp
            elif model is None:
                shutil.copy(model_path, test_model_path)
                model = PPO.load(test_model_path)
                model.ent_coef = config['model']['ent_coef']
                model._setup_lr_schedule()
        except FileNotFoundError:
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Nie znaleziono modelu w {model_path}. Czekam 5 sekund...\n")
            time.sleep(5)
            env.close()
            continue
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Błąd ładowania modelu: {e}\n")
            time.sleep(5)
            env.close()
            continue

        scores = []
        rewards = []
        lengths = []
        for ep in range(5):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            scores.append(info['score'])
            rewards.append(total_reward)
            lengths.append(steps)
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Epizod {ep+1}: Wynik={info['score']}, Nagroda={total_reward}, Kroki={steps}\n")

        mean_score = np.mean(scores)
        mean_reward = np.mean(rewards)
        mean_length = np.mean(lengths)
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Średni wynik: {mean_score}, Średnia nagroda: {mean_reward}, Średnia długość: {mean_length}\n")

        try:
            write_header = not os.path.exists(test_csv_path)
            with open(test_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(['timestamp', 'mean_score', 'mean_reward', 'mean_ep_length'])
                writer.writerow([time.time(), mean_score, mean_reward, mean_length])
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Błąd zapisu CSV: {e}\n")

        # Odczyt danych do wykresu
        timestamps = []
        mean_scores = []
        mean_rewards = []
        mean_lengths = []
        try:
            with open(test_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamps.append(float(row['timestamp']))
                    mean_scores.append(float(row['mean_score']))
                    mean_rewards.append(float(row['mean_reward']))
                    mean_lengths.append(float(row['mean_ep_length']))
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Błąd odczytu CSV: {e}\n")

        # Generowanie wykresu
        if timestamps:  # Sprawdź, czy istnieją dane do wykresu
            plt.figure(figsize=(8, 4))
            plt.plot(timestamps, mean_scores, label='mean_score')
            plt.plot(timestamps, mean_rewards, label='mean_reward')
            plt.plot(timestamps, mean_lengths, label='mean_ep_length')
            plt.xlabel('Timestamp')
            plt.ylabel('Value')
            plt.title('Test Progress')
            plt.legend()
            plt.tight_layout()
            plt.savefig(test_progress_path)
            plt.close()
            with open(log_path, 'a') as f:
                f.write(f"[{time.ctime()}] Zaktualizowano wykres testów: {test_progress_path}\n")

        env.close()
        time.sleep(5)

# Funkcja do zapisu stanu treningu
def save_training_state(model, env, eval_env, total_timesteps):
    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    vec_norm_path = model_path_absolute.replace('.zip', '_vecnorm.pkl')
    vec_norm_eval_path = model_path_absolute.replace('.zip', '_vecnorm_eval.pkl')

    for attempt in range(5):
        try:
            model.save(model_path_absolute)
            env.save(vec_norm_path)
            eval_env.save(vec_norm_eval_path)
            with open(model_path_absolute.replace('.zip', '_state.pkl'), 'wb') as f:
                pickle.dump({'total_timesteps': total_timesteps}, f)
            print(f"Stan treningu zapisany po {total_timesteps} krokach.")
            break
        except PermissionError as e:
            print(f"Próba {attempt + 1}/5: Nie udało się zapisać stanu: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"Błąd zapisu stanu: {e}")
            break
    else:
        print("Nie udało się zapisać stanu po 5 próbach.")

    # Sprawdź, czy istnieje best_model.zip i zaktualizuj snake_ppo_model.zip, jeśli jest nowszy
    best_model_file_path = os.path.join(best_model_save_path, 'best_model.zip')
    if os.path.exists(best_model_file_path):
        for attempt in range(5):
            try:
                with open(best_model_file_path, 'rb') as f:
                    pass
                shutil.copy(best_model_file_path, model_path_absolute)
                print(f"Zaktualizowano najlepszy model: {model_path_absolute}")
                break
            except PermissionError as e:
                print(f"Próba {attempt + 1}/5: Nie udało się skopiować best_model.zip: {e}")
                time.sleep(3)
            except Exception as e:
                print(f"Nie udało się skopiować best_model.zip: {e}")
                break
        else:
            print(f"Nie udało się skopiować best_model.zip po 5 próbach.")
    else:
        print(f"Plik {best_model_file_path} nie istnieje. Pomijam kopiowanie.")

test_stop_event = None
test_thread_handle = None

def train(use_progress_bar=False, use_config_hyperparams=True):
    global test_stop_event, test_thread_handle, best_model_save_path

    n_envs = config['training']['n_envs']
    n_steps = config['model']['n_steps']
    min_lr = config['model'].get('min_learning_rate', 0.0)
    eval_freq = config['training']['eval_freq']
    plot_interval = config['training']['plot_interval']

    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    vec_norm_path = model_path_absolute.replace('.zip', '_vecnorm.pkl')
    vec_norm_eval_path = model_path_absolute.replace('.zip', '_vecnorm_eval.pkl')
    state_path = model_path_absolute.replace('.zip', '_state.pkl')

    load_model = os.path.exists(model_path_absolute)
    total_timesteps = 0
    if load_model:
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                total_timesteps = state['total_timesteps']
            print(f"Wznowienie treningu od {total_timesteps} kroków.")
        except Exception as e:
            print(f"Błąd ładowania stanu: {e}. Zaczynam od zera.")

    reset_channel_logs()
    if enable_channel_logs:
        channel_loggers.update(init_channel_loggers())

    # Utwórz środowiska z losowymi grid_size (przekazując grid_size=None)
    env = make_vec_env(make_env(render_mode=None, grid_size=None), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    if load_model and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Eval env na fixed grid_size=16
    eval_env = make_vec_env(make_env(render_mode=None, grid_size=16), n_envs=1, vec_env_cls=SubprocVecEnv)
    if load_model and os.path.exists(vec_norm_eval_path):
        env = VecNormalize.load(vec_norm_path, env)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = config['model']['policy_kwargs']
    policy_kwargs['features_extractor_class'] = CustomCNN

    if load_model:
        model = PPO.load(model_path_absolute, env=env)
        model.ent_coef = config['model']['ent_coef']
        model.learning_rate = linear_schedule(config['model']['learning_rate'], min_lr)
        model._setup_lr_schedule()
    else:
        model = PPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], min_lr),
            n_steps=config['model']['n_steps'],
            batch_size=config['model']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config['model']['device']
        )

    if use_config_hyperparams and load_model:
        policy_kwargs = config['model']['policy_kwargs']
        policy_kwargs['features_extractor_class'] = CustomCNN
        model_new = PPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], min_lr),
            n_steps=config['model']['n_steps'],
            batch_size=config['model']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config['model']['device']
        )
        model_tmp = PPO.load(model_path_absolute)
        model_new.policy.load_state_dict(model_tmp.policy.state_dict())
        model = model_new
        del model_tmp

    global best_model_save_path
    best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
    eval_log_path = os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir']))

    stop_on_plateau = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config['training']['max_no_improvement_evals'],
        min_evals=config['training']['min_evals'],
        verbose=1
    )

    plot_script_path = os.path.join(os.path.dirname(__file__), 'plot_train_progress.py')

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_save_path,
        log_path=eval_log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_on_plateau,
        plot_interval=plot_interval,
        plot_script_path=plot_script_path
    )

    train_progress_callback = TrainProgressCallback(
        os.path.join(base_dir, config['paths']['train_csv_path']),
        initial_timesteps=total_timesteps
    )

    if config['training'].get('enable_testing', False):
        test_stop_event = threading.Event()
        test_model_path = os.path.join(base_dir, config['paths']['test_model_path'])
        test_log_path = os.path.join(base_dir, config['paths']['test_log_path'])
        test_csv_path = os.path.join(base_dir, config['paths']['test_csv_path'])
        test_progress_path = os.path.join(base_dir, config['paths']['test_progress_path'])
        grid_size = 16  # Fixed dla testu
        test_thread_handle = threading.Thread(target=test_thread, args=(
            model_path_absolute, test_model_path, test_log_path, test_csv_path, test_progress_path, grid_size, test_stop_event
        ))
        test_thread_handle.start()
        print("Uruchomiono wątek testujący w tle.")

    # Debug logowanie dla losowego grid_size
    if enable_channel_logs:
        for logger in channel_loggers.values():
            logger.info(f"\n--- Debug: Rozpoczęcie treningu ---")
        debug_env = make_env(render_mode=None, grid_size=None)()  # Losowy grid_size
        obs, _ = debug_env.reset()
        grid_size = debug_env.env.grid_size  # Pobierz losowy grid_size
        log_observation(obs, channel_loggers, grid_size, step=0)
        for debug_step in range(5):
            action = debug_env.action_space.sample()
            obs, reward, done, _, info = debug_env.step(action)
            log_observation(obs, channel_loggers, grid_size, step=debug_step + 1)
            if enable_channel_logs:
                for logger in channel_loggers.values():
                    logger.info(f"Akcja: {action}, Nagroda: {reward}, Done: {done}, Info: {info}")
            if done:
                break
        debug_env.close()
        if enable_channel_logs:
            for logger in channel_loggers.values():
                logger.info(f"--- Koniec debug dla grid_size={grid_size} ---")

    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        reset_num_timesteps=not load_model,
        callback=[eval_callback, train_progress_callback],
        progress_bar=use_progress_bar
    )

    save_training_state(model, env, eval_env, config['training']['total_timesteps'])

    env.close()
    eval_env.close()

    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", os.path.join(base_dir, config['paths']['test_log_path']))

if __name__ == "__main__":
    model = None
    try:
        train(use_progress_bar=True)
    except KeyboardInterrupt:
        print("Przerwano trening.")
        if test_stop_event is not None:
            test_stop_event.set()
            print("Wysłano sygnał zakończenia do wątku testującego.")
            if test_thread_handle is not None:
                test_thread_handle.join(timeout=10)
                print("Wątek testujący zakończony.")
        sys.exit(0)
    except Exception as e:
        print(f"Błąd podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        if test_stop_event is not None:
            test_stop_event.set()
            if test_thread_handle is not None:
                test_thread_handle.join(timeout=10)
        sys.exit(1)
    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", os.path.join(base_dir, config['paths']['test_log_path']))