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
    for i, channel_name in enumerate([
        'mapa', 'dx', 'dy', 'kierunek', 'grid_size', 'odleglosc']):
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

        obs, _ = env.reset()
        done = False
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Rozpoczynam test z wizualizacją dla grid_size={grid_size}...\n")
        while not done:
            if stop_event is not None and stop_event.is_set():
                print("[TestThread] Otrzymano sygnał zakończenia w trakcie epizodu. Kończę wątek testujący.")
                env.close()
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Wynik: {info['score']}, Nagroda: {info['total_reward']}, Grid_size: {info['grid_size']}\n")
                write_header = not os.path.exists(test_csv_path)
                try:
                    with open(test_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if write_header:
                            writer.writerow(['timestamp', 'score', 'total_reward', 'grid_size'])
                        writer.writerow([time.time(), info['score'], info['total_reward'], info['grid_size']])
                except Exception:
                    with open(log_path, 'a') as f:
                        f.write(f"[{time.ctime()}] Błąd zapisu do CSV testów.\n")
            except Exception as e:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Błąd w kroku testu: {e}\n")
                done = True
        env.close()
        time.sleep(5)

# Funkcja wyświetlająca harmonogram curriculum
def print_curriculum_schedule(curriculum_grid_sizes, curriculum_multipliers, n_envs, n_steps):
    print(f"\nCurriculum grid_sizes: {curriculum_grid_sizes}")
    print(f"Curriculum multipliers: {curriculum_multipliers}")
    print(f"n_envs: {n_envs}, n_steps: {n_steps}")
    print("Harmonogram curriculum:")
    total_steps = 0
    for i, (grid_size, multiplier) in enumerate(zip(curriculum_grid_sizes, curriculum_multipliers), 1):
        steps = n_envs * n_steps * multiplier
        start = total_steps
        end = total_steps + steps - 1
        print(f"  {i}. grid_size={grid_size}: od {start} do {end} (kroków: {steps})")
        total_steps += steps

# Funkcja do zapisu stanu treningu
def save_training_state(model, env, eval_env, total_timesteps, current_level, model_path, vec_norm_path, vec_norm_eval_path):
    model.save(model_path)
    env.save(vec_norm_path)
    eval_env.save(vec_norm_eval_path)
    with open(os.path.join(base_dir, config['paths']['models_dir'], 'training_state.pkl'), 'wb') as f:
        pickle.dump({'total_timesteps': total_timesteps, 'current_level': current_level}, f)
    print(f"Zapisano stan treningu: model={model_path}, vec_norm={vec_norm_path}, vec_norm_eval={vec_norm_eval_path}, total_timesteps={total_timesteps}, current_level={current_level}")

# Funkcja do wczytania stanu treningu
def load_training_state(model_path, vec_norm_path, vec_norm_eval_path):
    total_timesteps = 0
    current_level = 0
    env = None
    eval_env = None
    model = None
    load_model = False
    use_config_hyperparams = True

    if os.path.exists(model_path):
        while True:
            response = input(f"Czy chcesz wczytać istniejący model z {model_path}? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'n']:
                break
            print("Proszę wpisać 'y', 'n' lub wcisnąć Enter (domyślnie 'Y').")
        load_model = response != 'n'

    if load_model:
        while True:
            response = input("Czy użyć hiperparametrów z pliku konfiguracyjnego? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'n']:
                break
            print("Proszę wpisać 'y', 'n' lub wcisnąć Enter (domyślnie 'Y').")
        use_config_hyperparams = response != 'n'

    try:
        if load_model and os.path.exists(model_path):
            model = PPO.load(model_path)
            print(f"Wczytano model z {model_path}")
            if use_config_hyperparams:
                print("Używam hiperparametrów z config.yaml")
                model.learning_rate = linear_schedule(config['model']['learning_rate'], config['model'].get('min_learning_rate', 0.00005))
                model.n_steps = config['model']['n_steps']
                model.batch_size = config['model']['batch_size']
                model.n_epochs = config['model']['n_epochs']
                model.gamma = config['model']['gamma']
                model.gae_lambda = config['model']['gae_lambda']
                model.ent_coef = config['model']['ent_coef']
                model.vf_coef = config['model']['vf_coef']
                # clip_range zostanie ustawione przy inicjalizacji modelu w funkcji train
            else:
                print("Używam hiperparametrów z wczytanego modelu")
        if os.path.exists(vec_norm_path):
            print(f"Wczytano VecNormalize dla env z {vec_norm_path}")
        if os.path.exists(vec_norm_eval_path):
            print(f"Wczytano VecNormalize dla eval_env z {vec_norm_eval_path}")
        state_path = os.path.join(base_dir, config['paths']['models_dir'], 'training_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                total_timesteps = state['total_timesteps']
                current_level = state['current_level']
                print(f"Wczytano stan treningu: total_timesteps={total_timesteps}, current_level={current_level}")
    except Exception as e:
        print(f"Błąd podczas wczytywania stanu treningu: {e}")
    return model, env, eval_env, total_timesteps, current_level, use_config_hyperparams

# Główna funkcja treningu
test_stop_event = None
test_thread_handle = None

def train(use_progress_bar=False):
    global test_stop_event, test_thread_handle
    test_stop_event = threading.Event()
    test_started = False

    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    test_model_path = os.path.normpath(os.path.join(base_dir, config['paths']['test_model_path']))
    test_log_path = os.path.normpath(os.path.join(base_dir, config['paths']['test_log_path']))
    test_csv_path = os.path.normpath(os.path.join(base_dir, config['paths']['test_csv_path']))
    test_progress_path = os.path.normpath(os.path.join(base_dir, config['paths']['test_progress_path']))
    vec_norm_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir'], 'vec_normalize.pkl'))
    vec_norm_eval_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir'], 'vec_normalize_eval.pkl'))

    curriculum_grid_sizes = config['training']['curriculum_grid_sizes']
    curriculum_multipliers = config['training']['curriculum_multipliers']
    n_envs = config['training']['n_envs']
    n_steps = config['model']['n_steps']
    
    # Wyświetl harmonogram curriculum
    print_curriculum_schedule(curriculum_grid_sizes, curriculum_multipliers, n_envs, n_steps)

    # Wczytaj stan treningu, jeśli istnieje
    model, env, eval_env, total_timesteps, current_level, use_config_hyperparams = load_training_state(model_path_absolute, vec_norm_path, vec_norm_eval_path)
    min_lr = config['model'].get('min_learning_rate', 0.00005)

    # Jeśli trening od zera (model == None), zresetuj train_progress.csv i liczniki
    if model is None:
        train_csv_path = os.path.join(base_dir, config['paths']['train_csv_path'])
        with open(train_csv_path, 'w', encoding='utf-8'):
            pass  # nadpisz pustą zawartością
        total_timesteps = 0
        current_level = 0

    # Zawsze resetuj logi kanałów na początku treningu (niezależnie czy kontynuacja czy nie)
    if enable_channel_logs:
        reset_channel_logs()
        global channel_loggers
        channel_loggers = init_channel_loggers()

    # Oblicz kroki dla każdego poziomu curriculum
    curriculum_steps = [n_envs * n_steps * multiplier for multiplier in curriculum_multipliers]
    cumulative_steps = np.cumsum([0] + curriculum_steps)

    # Zacznij od wczytanego poziomu lub od początku
    for level, grid_size in enumerate(curriculum_grid_sizes[current_level:], current_level):
        set_grid_size(grid_size)
        steps_per_iteration = curriculum_steps[level]
        steps_already_done = max(0, total_timesteps - cumulative_steps[level])
        steps_remaining = max(0, curriculum_steps[level] - steps_already_done)

        if steps_remaining <= 0:
            print(f"Poziom {level+1} (grid_size={grid_size}) już ukończony, pomijam.")
            continue

        print(f"\nRozpoczynam poziom {level+1} (grid_size={grid_size}), kroki: {steps_already_done}/{curriculum_steps[level]} (pozostało: {steps_remaining})")

        # Logowanie debug do osobnych plików kanałów
        if enable_channel_logs:
            for logger in channel_loggers.values():
                logger.info(f"\n--- Debug: Pierwsze wywołanie env dla grid_size={grid_size} ---")
        debug_env = make_env(render_mode=None, grid_size=grid_size)()
        obs, _ = debug_env.reset()
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

        # Tworzenie środowiska
        if env is None:
            env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        else:
            env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            env = VecNormalize.load(vec_norm_path, env) if os.path.exists(vec_norm_path) else VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        if eval_env is None:
            eval_env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=1, vec_env_cls=SubprocVecEnv)
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        else:
            eval_env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=1, vec_env_cls=SubprocVecEnv)
            eval_env = VecNormalize.load(vec_norm_eval_path, eval_env) if os.path.exists(vec_norm_eval_path) else VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

        # Inicjalizacja modelu, jeśli nie wczytano
        if model is None:
            policy_kwargs = config['model']['policy_kwargs']
            policy_kwargs['features_extractor_class'] = CustomCNN
            model = PPO(
                config['model']['policy'],
                env,
                learning_rate=linear_schedule(config['model']['learning_rate'], min_lr),
                n_steps=n_steps,
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
        else:
            model.set_env(env)
            if use_config_hyperparams:
                # Ponownie inicjalizujemy model z parametrami z configu
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
                # Wczytujemy zapisane wagi modelu
                model_tmp = PPO.load(model_path_absolute)
                model_new.policy.load_state_dict(model_tmp.policy.state_dict())
                model = model_new
                del model_tmp

        best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
        eval_log_path = os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir']))

        stop_on_plateau = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=config['training']['max_no_improvement_evals'],
            min_evals=config['training']['min_evals'],
            verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=eval_log_path,
            eval_freq=config['training']['eval_freq'],
            deterministic=True,
            render=False,
            callback_after_eval=stop_on_plateau
        )

        train_progress_callback = TrainProgressCallback(
            os.path.join(base_dir, config['paths']['train_csv_path']),
            initial_timesteps=total_timesteps
        )

        while steps_remaining > 0 and total_timesteps < config['training']['total_timesteps']:
            steps_to_train = min(steps_per_iteration, steps_remaining)
            model.learn(
                total_timesteps=steps_to_train,
                reset_num_timesteps=False,
                callback=[eval_callback, train_progress_callback],
                progress_bar=use_progress_bar
            )
            total_timesteps += steps_to_train
            steps_remaining -= steps_to_train
            save_training_state(model, env, eval_env, total_timesteps, level, model_path_absolute, vec_norm_path, vec_norm_eval_path)

            if config['training']['enable_testing'] and not test_started:
                test_thread_handle = threading.Thread(
                    target=test_thread,
                    args=(
                        model_path_absolute,
                        test_model_path,
                        test_log_path,
                        test_csv_path,
                        test_progress_path,
                        grid_size,
                        test_stop_event
                    ),
                    daemon=True
                )
                test_thread_handle.start()
                test_started = True

            time.sleep(5)

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

            train_csv_path = os.path.join(base_dir, config['paths']['train_csv_path'])
            if os.path.exists(train_csv_path):
                try:
                    train_plot_script = os.path.join(os.path.dirname(__file__), 'plot_train_progress.py')
                    subprocess.run([sys.executable, train_plot_script], check=True)
                    print(f"Wygenerowano wykres treningu: {os.path.join(base_dir, config['paths']['plot_path'])}")
                except Exception as e:
                    print(f"Błąd podczas generowania wykresu treningu: {e}")
            else:
                print(f"Plik {train_csv_path} nie istnieje. Pomijam generowanie wykresu.")

        env.close()
        eval_env.close()

    # Kontynuacja treningu z ostatnim grid_size
    if total_timesteps < config['training']['total_timesteps']:
        grid_size = curriculum_grid_sizes[-1]
        set_grid_size(grid_size)
        steps_per_iteration = n_envs * n_steps * curriculum_multipliers[-1]
        steps_remaining = config['training']['total_timesteps'] - total_timesteps

        print(f"\nKontynuuję trening z ostatnim grid_size={grid_size} aż do {config['training']['total_timesteps']} kroków")

        if enable_channel_logs:
            for logger in channel_loggers.values():
                logger.info(f"\n--- Debug: Kontynuacja treningu dla grid_size={grid_size} ---")
        debug_env = make_env(render_mode=None, grid_size=grid_size)()
        obs, _ = debug_env.reset()
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
                logger.info(f"--- Koniec debug dla kontynuacji grid_size={grid_size} ---")

        env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        env = VecNormalize.load(vec_norm_path, env) if os.path.exists(vec_norm_path) else VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=1, vec_env_cls=SubprocVecEnv)
        eval_env = VecNormalize.load(vec_norm_eval_path, eval_env) if os.path.exists(vec_norm_eval_path) else VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        model.set_env(env)
        if use_config_hyperparams:
            # Ponownie inicjalizujemy model z parametrami z configu
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
            # Wczytujemy zapisane wagi modelu
            model_tmp = PPO.load(model_path_absolute)
            model_new.policy.load_state_dict(model_tmp.policy.state_dict())
            model = model_new
            del model_tmp

        best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
        eval_log_path = os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir']))

        stop_on_plateau = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=config['training']['max_no_improvement_evals'],
            min_evals=config['training']['min_evals'],
            verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=eval_log_path,
            eval_freq=config['training']['eval_freq'],
            deterministic=True,
            render=False,
            callback_after_eval=stop_on_plateau
        )

        train_progress_callback = TrainProgressCallback(
            os.path.join(base_dir, config['paths']['train_csv_path']),
            initial_timesteps=total_timesteps
        )

        while total_timesteps < config['training']['total_timesteps']:
            steps_to_train = min(steps_per_iteration, steps_remaining)
            model.learn(
                total_timesteps=steps_to_train,
                reset_num_timesteps=False,
                callback=[eval_callback, train_progress_callback],
                progress_bar=use_progress_bar
            )
            total_timesteps += steps_to_train
            steps_remaining -= steps_to_train
            save_training_state(model, env, eval_env, total_timesteps, len(curriculum_grid_sizes), model_path_absolute, vec_norm_path, vec_norm_eval_path)

            time.sleep(5)

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

            train_csv_path = os.path.join(base_dir, config['paths']['train_csv_path'])
            if os.path.exists(train_csv_path):
                try:
                    train_plot_script = os.path.join(os.path.dirname(__file__), 'plot_train_progress.py')
                    subprocess.run([sys.executable, train_plot_script], check=True)
                    print(f"Wygenerowano wykres treningu: {os.path.join(base_dir, config['paths']['plot_path'])}")
                except Exception as e:
                    print(f"Błąd podczas generowania wykresu treningu: {e}")
            else:
                print(f"Plik {train_csv_path} nie istnieje. Pomijam generowanie wykresu.")

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