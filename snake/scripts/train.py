import os
import time
import threading
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
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

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Callback do zapisu postępu treningu
class TrainProgressCallback(BaseCallback):
    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.header_written = False
        self.last_logged = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            if self.num_timesteps - self.last_logged >= 1000:
                ep_rew_mean = self.model.ep_info_buffer and np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) or None
                ep_len_mean = self.model.ep_info_buffer and np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]) or None
                grid_size = self.training_env.get_attr('grid_size')[0]
                try:
                    write_header = not os.path.exists(self.csv_path)
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if write_header:
                            writer.writerow(['timesteps', 'mean_reward', 'mean_ep_length', 'grid_size'])
                        writer.writerow([self.num_timesteps, ep_rew_mean, ep_len_mean, grid_size])
                    self.last_logged = self.num_timesteps
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
                model.clip_range = lambda x: config['model']['clip_range']
                model._setup_lr_schedule()
                current_model_timestamp = new_timestamp
            elif model is None:
                shutil.copy(model_path, test_model_path)
                model = PPO.load(test_model_path)
                model.ent_coef = config['model']['ent_coef']
                model.clip_range = lambda x: config['model']['clip_range']
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
                try:
                    timestamps, scores, rewards, grid_sizes_data = [], [], [], []
                    with open(test_csv_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader, None)
                        for row in reader:
                            if len(row) >= 4:
                                timestamps.append(float(row[0]))
                                scores.append(float(row[1]))
                                rewards.append(float(row[2]))
                                grid_sizes_data.append(float(row[3]))
                    if scores:
                        episodes = list(range(1, len(scores) + 1))
                        plt.figure(figsize=(8, 4))
                        for gs in set(grid_sizes_data):
                            indices = [i for i, g in enumerate(grid_sizes_data) if g == gs]
                            plt.plot([episodes[i] for i in indices], [scores[i] for i in indices], label=f'score (grid_size={int(gs)})', alpha=0.6)
                            plt.plot([episodes[i] for i in indices], [rewards[i] for i in indices], label=f'reward (grid_size={int(gs)})', alpha=0.6, linestyle='--')
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
            except Exception as e:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Błąd podczas predykcji: {e}\n")
                done = True
        env.close()
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Epizod zakończony. Sprawdzam nowy model...\n")
        time.sleep(1)

def train(use_progress_bar=True):
    global test_stop_event, test_thread_handle
    # Utwórz katalogi
    models_dir = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
    logs_dir = os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir']))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Inicjalizacja curriculum
    curriculum_grid_sizes = config['training']['curriculum_grid_sizes']
    curriculum_multipliers = config['training']['curriculum_multipliers']
    curriculum_steps = [config['training']['n_envs'] * config['model']['n_steps'] * m for m in curriculum_multipliers]
    print(f"Plan curriculum: {list(zip(curriculum_grid_sizes, curriculum_steps))}")

    # Oblicz steps_per_iteration
    steps_per_iteration = config['training']['n_envs'] * config['model']['n_steps']
    
    # Sprawdź, czy model istnieje
    model = None
    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    print(f"Sprawdzam istnienie modelu w: {model_path_absolute}")
    total_timesteps = 0
    test_started = False
    test_stop_event = threading.Event()

    # Przygotuj policy_kwargs z CustomCNN
    policy_kwargs = {
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': config['model']['policy_kwargs']['features_extractor_kwargs'],
        'net_arch': config['model']['policy_kwargs']['net_arch']
    }

    for level, (grid_size, max_steps) in enumerate(zip(curriculum_grid_sizes, curriculum_steps)):
        print(f"\nRozpoczynam poziom curriculum {level + 1}: grid_size={grid_size}, max_steps={max_steps}")
        set_grid_size(grid_size)
        env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=config['training']['n_envs'], vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(make_env(render_mode=None, grid_size=grid_size), n_envs=1, vec_env_cls=SubprocVecEnv)
        
        # Inicjalizacja modelu dla nowego poziomu curriculum
        train_csv_path = os.path.join(base_dir, config['paths']['train_csv_path'])
        train_progress_callback = TrainProgressCallback(train_csv_path)
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

        if model is None:
            try:
                model = PPO.load(model_path_absolute, env=env)
                print(f"Załadowano istniejący model z: {model_path_absolute}")
            except FileNotFoundError:
                print(f"Model nie istnieje, tworzę nowy model.")
                model = PPO(
                    config['model']['policy'],
                    env,
                    learning_rate=config['model']['learning_rate'],
                    n_steps=config['model']['n_steps'],
                    batch_size=config['model']['batch_size'],
                    n_epochs=config['model']['n_epochs'],
                    gamma=config['model']['gamma'],
                    gae_lambda=config['model']['gae_lambda'],
                    clip_range=config['model']['clip_range'],
                    ent_coef=config['model']['ent_coef'],
                    vf_coef=config['model']['vf_coef'],
                    policy_kwargs=policy_kwargs,
                    device=config['model']['device'],
                    verbose=1
                )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=max_steps,
            reset_num_timesteps=False,
            callback=[eval_callback, train_progress_callback],
            progress_bar=use_progress_bar
        )
        total_timesteps += max_steps
        model.save(model_path_absolute)
        print(f"Model zapisany w: {model_path_absolute}")

        if config['training']['enable_testing'] and not test_started:
            test_thread_handle = threading.Thread(
                target=test_thread,
                args=(
                    model_path_absolute,
                    os.path.join(base_dir, config['paths']['test_model_path']),
                    os.path.join(base_dir, config['paths']['test_log_path']),
                    os.path.join(base_dir, config['paths']['test_csv_path']),
                    os.path.join(base_dir, config['paths']['test_progress_path']),
                    grid_size,
                    test_stop_event
                ),
                daemon=True
            )
            test_thread_handle.start()
            test_started = True

        time.sleep(5)

        backup_model = model_path_absolute + '.backup'
        try:
            shutil.copy(model_path_absolute, backup_model)
            print(f"Stworzono kopię zapasową modelu w {backup_model}")
        except Exception as e:
            print(f"Nie udało się stworzyć kopii zapasowej modelu: {e}")

        best_model_file_path = os.path.join(best_model_save_path, 'best_model.zip')
        if os.path.exists(best_model_file_path):
            for attempt in range(5):
                try:
                    with open(best_model_file_path, 'rb') as f:
                        pass
                    shutil.copy(best_model_file_path, model_path_absolute)
                    best_model_backup_path = best_model_file_path + '.backup'
                    shutil.copy(best_model_file_path, best_model_backup_path)
                    print(f"Zaktualizowano najlepszy model: {model_path_absolute}")
                    print(f"Stworzono kopię zapasową najlepszego modelu w {best_model_backup_path}")
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

    # Kontynuacja treningu
    if total_timesteps < config['training']['total_timesteps']:
        print(f"\nKontynuuję trening z ostatnim grid_size={curriculum_grid_sizes[-1]} aż do {config['training']['total_timesteps']} kroków")
        set_grid_size(curriculum_grid_sizes[-1])
        env = make_vec_env(make_env(render_mode=None, grid_size=curriculum_grid_sizes[-1]), n_envs=config['training']['n_envs'], vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(make_env(render_mode=None, grid_size=curriculum_grid_sizes[-1]), n_envs=1, vec_env_cls=SubprocVecEnv)
        model.set_env(env)
        
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

        while total_timesteps < config['training']['total_timesteps']:
            model.learn(
                total_timesteps=steps_per_iteration,
                reset_num_timesteps=False,
                callback=[eval_callback, train_progress_callback],
                progress_bar=use_progress_bar
            )
            total_timesteps += steps_per_iteration
            model.save(model_path_absolute)
            print(f"Model zapisany w: {model_path_absolute}")

            time.sleep(5)

            backup_model = model_path_absolute + '.backup'
            try:
                shutil.copy(model_path_absolute, backup_model)
                print(f"Stworzono kopię zapasową modelu w {backup_model}")
            except Exception as e:
                print(f"Nie udało się stworzyć kopii zapasowej modelu: {e}")

            best_model_file_path = os.path.join(best_model_save_path, 'best_model.zip')
            if os.path.exists(best_model_file_path):
                for attempt in range(5):
                    try:
                        with open(best_model_file_path, 'rb') as f:
                            pass
                        shutil.copy(best_model_file_path, model_path_absolute)
                        best_model_backup_path = best_model_file_path + '.backup'
                        shutil.copy(best_model_file_path, best_model_backup_path)
                        print(f"Zaktualizowano najlepszy model: {model_path_absolute}")
                        print(f"Stworzono kopię zapasową najlepszego modelu w {best_model_backup_path}")
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