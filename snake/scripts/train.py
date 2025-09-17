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
from cnn import CustomFeaturesExtractor
import logging
import pickle

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Definicja zmiennych globalnych
test_stop_event = None
test_thread_handle = None

def reset_channel_logs():
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    for channel_name in ['mapa', 'direction', 'grid_size', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']:
        log_path = os.path.join(log_dir, f'training_{channel_name}.log')
        with open(log_path, 'w', encoding='utf-8'):
            pass

def init_channel_loggers():
    loggers = {}
    log_dir = os.path.join(base_dir, 'logs')
    for channel_name in ['mapa', 'direction', 'grid_size', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']:
        log_path = os.path.join(log_dir, f'training_{channel_name}.log')
        logger = logging.getLogger(f'channel_{channel_name}')
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
channel_loggers = init_channel_loggers() if enable_channel_logs else {}

def log_observation(obs, channel_loggers, grid_size, step):
    if not enable_channel_logs:
        return
    image = obs['image']
    latest_channel = image[-1:, :, :]  # [1, H, W]
    mapa = latest_channel[0, :, :]  # [H, W]
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
    if head_x >= 0 and food_x >= 0:
        distance = abs(head_x - food_x) + abs(head_y - food_y)
    else:
        distance = float('inf')
    
    logger = channel_loggers.get('mapa')
    if logger:
        logger.info(f"--- Obserwacja dla grid_size={grid_size}, krok={step} ---")
        logger.info(f"Kanał mapa:\n{np.array_str(mapa, precision=2, suppress_small=True, max_line_width=120)}")
        logger.info(f"Pozycja głowy: ({head_x}, {head_y}) | Pozycja jedzenia: ({food_x}, {food_y})")
        logger.info(f"Dystans Manhattan: {distance}")
        logger.info("-" * 60)

    for scalar_name in ['direction', 'grid_size', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']:
        logger = channel_loggers.get(scalar_name)
        if logger:
            logger.info(f"--- Obserwacja dla grid_size={grid_size}, krok={step} ---")
            logger.info(f"{scalar_name.capitalize()}: {obs[scalar_name][0]}")
            logger.info("-" * 60)

def linear_schedule(initial_value, min_value=0.00005):
    def func(progress_remaining):
        return min_value + (initial_value - min_value) * progress_remaining
    return func

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
                grid_size = np.mean(self.training_env.get_attr('grid_size'))  # Średni grid_size
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

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, callback_on_new_best=None, callback_after_eval=None, best_model_save_path=None,
                 log_path=None, eval_freq=10000, deterministic=True, render=False, verbose=1,
                 warn=True, n_eval_episodes=5, plot_interval=3, plot_script_path=None, initial_timesteps=0):
        super().__init__(eval_env, callback_on_new_best=callback_on_new_best, callback_after_eval=callback_after_eval,
                         best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq,
                         deterministic=deterministic, render=render, verbose=verbose, warn=warn,
                         n_eval_episodes=n_eval_episodes)
        self.eval_count = 0
        self.plot_interval = plot_interval
        self.plot_script_path = plot_script_path
        self.initial_timesteps = initial_timesteps

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_count += 1
            # Zapis najnowszego modelu po każdej walidacji
            total_timesteps = self.model.num_timesteps + self.initial_timesteps
            save_training_state(self.model, self.model.env, self.eval_env, total_timesteps, self.best_model_save_path)
            print(f"Zaktualizowano najnowszy model po {total_timesteps} krokach z mean_reward={self.last_mean_reward}")

            # Zapis najlepszego modelu, jeśli poprawiono mean_reward
            if self.best_mean_reward < self.last_mean_reward:
                self.model.save(os.path.join(self.best_model_save_path, f'best_model_{total_timesteps}.zip'))
                print(f"New best model saved at {total_timesteps} timesteps with mean reward {self.last_mean_reward}")
            if self.eval_count % self.plot_interval == 0:
                try:
                    subprocess.run([sys.executable, self.plot_script_path], check=True)
                    print(f"Wygenerowano wykres po {self.eval_count} walidacji.")
                except Exception as e:
                    print(f"Błąd podczas generowania wykresu: {e}")
        return True

def test_thread(model_path, test_model_path, log_path, test_csv_path, test_progress_path, grid_size, stop_event=None):
    current_model_timestamp = 0
    timestamps = []
    mean_scores = []
    mean_rewards = []
    mean_lengths = []
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
        timestamps.append(time.time())
        mean_scores.append(mean_score)
        mean_rewards.append(mean_reward)
        mean_lengths.append(mean_length)
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

        if timestamps:
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

def save_training_state(model, env, eval_env, total_timesteps, save_path):
    model_path_absolute = os.path.normpath(os.path.join(save_path, 'snake_ppo_model.zip'))
    vec_norm_path = model_path_absolute.replace('.zip', '_vecnorm.pkl')
    vec_norm_eval_path = model_path_absolute.replace('.zip', '_vecnorm_eval.pkl')
    state_path = model_path_absolute.replace('.zip', '_state.pkl')

    for attempt in range(5):
        try:
            model.save(model_path_absolute)
            env.save(vec_norm_path)
            eval_env.save(vec_norm_eval_path)
            with open(state_path, 'wb') as f:
                pickle.dump({'total_timesteps': total_timesteps}, f)
            print(f"Zaktualizowano najnowszy model: {model_path_absolute} po {total_timesteps} krokach.")
            break
        except PermissionError as e:
            print(f"Próba {attempt + 1}/5: Nie udało się zapisać stanu: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"Błąd zapisu stanu: {e}")
            break
    else:
        print("Nie udało się zapisać stanu po 5 próbach.")

def train(use_progress_bar=False, use_config_hyperparams=True):
    global test_stop_event, test_thread_handle, best_model_save_path

    n_envs = config['training']['n_envs']
    n_steps = config['model']['n_steps']
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
        # Interaktywne pytania dla użytkownika gdy znaleziono model
        try:
            resp = input(f"Znaleziono istniejący model pod {model_path_absolute}. Czy kontynuować trening? [Y/n]: ").strip()
        except Exception:
            resp = ''
        if resp.lower() in ('n', 'no'):
            print("Anulowano trening przez użytkownika.")
            return

        try:
            resp2 = input("Użyć hyperparametrów z configu zamiast z modelu? [Y/n]: ").strip()
        except Exception:
            resp2 = ''
        # Domyślnie True (użyj configu)
        use_config_hyperparams = False if resp2.lower() in ('n', 'no') else True

    reset_channel_logs()
    if enable_channel_logs:
        channel_loggers.update(init_channel_loggers())

    env = make_vec_env(make_env(render_mode=None, grid_size=None), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    if load_model and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = make_vec_env(make_env(render_mode=None, grid_size=16), n_envs=1, vec_env_cls=SubprocVecEnv)
    if load_model and os.path.exists(vec_norm_eval_path):
        eval_env = VecNormalize.load(vec_norm_eval_path, eval_env)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = config['model']['policy_kwargs']
    policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor

    if load_model:
        model = PPO.load(model_path_absolute, env=env)
        model.ent_coef = config['model']['ent_coef']
        model.learning_rate = linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate'])
        model._setup_lr_schedule()
    else:
        model = PPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate']),
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
        policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor
        model_new = PPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate']),
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

    stop_on_plateau = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config['training']['max_no_improvement_evals'],
        min_evals=config['training']['min_evals'],
        verbose=1
    )

    plot_script_path = os.path.join(os.path.dirname(__file__), 'plot_train_progress.py')

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_save_path,
        log_path=os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir'])),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_on_plateau,
        plot_interval=plot_interval,
        plot_script_path=plot_script_path,
        initial_timesteps=total_timesteps
    )

    train_progress_callback = TrainProgressCallback(
        os.path.join(base_dir, config['paths']['train_csv_path']),
        initial_timesteps=total_timesteps
    )

    if config['training'].get('enable_testing', False):
        global test_stop_event, test_thread_handle
        test_stop_event = threading.Event()
        test_model_path = os.path.join(base_dir, config['paths']['test_model_path'])
        test_log_path = os.path.join(base_dir, config['paths']['test_log_path'])
        test_csv_path = os.path.join(base_dir, config['paths']['test_csv_path'])
        test_progress_path = os.path.join(base_dir, config['paths']['test_progress_path'])
        grid_size = 16
        test_thread_handle = threading.Thread(target=test_thread, args=(
            model_path_absolute, test_model_path, test_log_path, test_csv_path, test_progress_path, grid_size, test_stop_event
        ))
        test_thread_handle.start()
        print("Uruchomiono wątek testujący w tle.")

    if enable_channel_logs:
        for logger in channel_loggers.values():
            logger.info(f"\n--- Debug: Rozpoczęcie treningu ---")
        debug_env = make_env(render_mode=None, grid_size=None)()
        obs, _ = debug_env.reset()
        grid_size = debug_env.env.grid_size
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

    try:
        # Oblicz pozostałe kroki do treningu
        configured_total = config['training'].get('total_timesteps', 0)
        remaining_timesteps = configured_total - total_timesteps
        # Jeżeli osiągnięto już >=80% limitu, zapytaj czy dodać dodatkowe kroki
        try:
            if configured_total > 0 and total_timesteps / configured_total >= 0.8:
                print(f"Użyto {total_timesteps}/{configured_total} kroków ({total_timesteps/configured_total:.1%}). To >=80% limitu.")
                extra = input("Ile dodatkowych kroków dodać? (0 = brak, domyślnie 0): ").strip()
                try:
                    extra_int = int(extra) if extra != '' else 0
                except ValueError:
                    print("Niepoprawna wartość, używam 0 dodatkowych kroków.")
                    extra_int = 0
                remaining_timesteps += extra_int
        except Exception:
            # W środowiskach nieinteraktywnych input może rzucić wyjątkiem; ignoruj wtedy
            pass
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                reset_num_timesteps=not load_model,
                callback=[eval_callback, train_progress_callback],
                progress_bar=use_progress_bar,
                tb_log_name=f"ppo_snake_{total_timesteps}"
            )
        else:
            print(f"Trening zakończony: osiągnięto {total_timesteps} kroków.")
    except Exception as e:
        print(f"Błąd podczas treningu: {e}")
        raise
    finally:
        env.close()
        eval_env.close()

    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", os.path.join(base_dir, config['paths']['test_log_path']))

if __name__ == "__main__":
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