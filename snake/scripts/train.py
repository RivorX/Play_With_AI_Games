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
                try:
                    write_header = not os.path.exists(self.csv_path)
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if write_header:
                            writer.writerow(['timesteps', 'mean_reward', 'mean_ep_length'])
                        writer.writerow([self.num_timesteps, ep_rew_mean, ep_len_mean])
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
        env = make_env(render_mode="human")()
        model = None
        try:
            new_timestamp = os.path.getmtime(model_path)
            if new_timestamp > current_model_timestamp:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Ładuję nowy model dla grid_size={grid_size}...\n")
                shutil.copy(model_path, test_model_path)
                model = PPO.load(test_model_path)
                model.ent_coef = config['model']['ent_coef']  # Ustaw nowy ent_coef
                # Ustaw clip_range jako funkcję zwracającą stałą wartość
                model.clip_range = lambda x: config['model']['clip_range']
                model._setup_lr_schedule()  # Reset harmonogramów
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
                    f.write(f"[{time.ctime()}] Wynik: {info['score']}, Nagroda: {info['total_reward']}\n")
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
                try:
                    timestamps, scores, rewards = [], [], []
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
            except Exception as e:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Błąd podczas predykcji: {e}\n")
                done = True
        env.close()
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Epizod zakończony. Sprawdzam nowy model...\n")
        time.sleep(1)

def train(use_progress_bar=True):
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
    if os.path.exists(model_path_absolute):
        choice = input(f"Model {model_path_absolute} istnieje. Kontynuować trening z tego modelu? [y/n]: ").strip().lower()
        if choice == 'y':
            print("Ładuję istniejący model...")
            model = PPO.load(model_path_absolute)
            total_timesteps = model.num_timesteps
            print(f"Załadowano model z timesteps: {total_timesteps}")
            update_params = input("Czy chcesz załadować nowe hiperparametry z config.yaml? [y/n]: ").strip().lower()
            if update_params == 'y':
                model.ent_coef = config['model']['ent_coef']
                model.clip_range = lambda x: config['model']['clip_range']  # Ustaw jako funkcja
                model.learning_rate = config['model']['learning_rate']
                model.gamma = config['model']['gamma']
                model.gae_lambda = config['model']['gae_lambda']
                model.vf_coef = config['model']['vf_coef']
                model._setup_lr_schedule()  # Reset harmonogramów
                print(f"Zaktualizowano hiperparametry: ent_coef={model.ent_coef}, clip_range={config['model']['clip_range']}, itd.")
        elif choice == 'n':
            print("Tworzę nowy model...")
            for p in [config['paths']['model_path'], config['paths']['best_model_path'], config['paths']['test_model_path'], config['paths']['best_model_backup_path']]:
                full_path = os.path.join(base_dir, p)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        print(f"Usunięto stary plik modelu: {full_path}")
                    except Exception as e:
                        print(f"Nie udało się usunąć {full_path}: {e}")
            for p in [config['paths']['train_csv_path'], config['paths']['test_csv_path'], config['paths']['plot_path'], config['paths']['test_progress_path'], config['paths']['test_log_path']]:
                full_path = os.path.join(base_dir, p)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        print(f"Usunięto stary plik logu/wykresu: {full_path}")
                    except Exception as e:
                        print(f"Nie udało się usunąć {full_path}: {e}")
            backup_csv = os.path.join(base_dir, config['paths']['train_csv_path']) + '.backup'
            if os.path.exists(os.path.join(base_dir, config['paths']['train_csv_path'])):
                shutil.copy(os.path.join(base_dir, config['paths']['train_csv_path']), backup_csv)
                print(f"Stworzono kopię zapasową CSV w {backup_csv}")
        else:
            print("Nieprawidłowy wybór. Kończę program.")
            return
    else:
        print(f"Model {model_path_absolute} nie istnieje. Tworzę nowy model...")

    # Określ początkowy poziom curriculum na podstawie total_timesteps
    total_timesteps = 0 if model is None else model.num_timesteps
    cumulative_steps = 0
    start_level = 0
    for level, max_steps in enumerate(curriculum_steps):
        cumulative_steps += max_steps
        if total_timesteps >= cumulative_steps:
            start_level = level + 1
        else:
            break

    test_started = False
    test_stop_event = threading.Event()
    train_progress_callback = TrainProgressCallback(os.path.join(base_dir, config['paths']['train_csv_path']))
    
    # Pętla curriculum
    for level, (grid_size, max_steps) in enumerate(zip(curriculum_grid_sizes, curriculum_steps)):
        if level < start_level:
            print(f"Pomijam poziom curriculum {level + 1}: grid_size={grid_size}, max_steps={max_steps} (już wykonano)")
            continue
        print(f"\nRozpoczynam poziom curriculum {level + 1}: grid_size={grid_size}, max_steps={max_steps}")
        
        set_grid_size(grid_size)
        env = make_vec_env(make_env(render_mode=None), n_envs=config['training']['n_envs'], vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(make_env(render_mode=None), n_envs=1, vec_env_cls=SubprocVecEnv)
        
        print(f"Środowisko utworzone z observation_space: {env.observation_space}")
        
        if model is None:
            model = PPO(
                policy=config['model']['policy'],
                env=env,
                learning_rate=config['model']['learning_rate'],
                n_steps=config['model']['n_steps'],
                batch_size=config['model']['batch_size'],
                n_epochs=config['model']['n_epochs'],
                gamma=config['model']['gamma'],
                gae_lambda=config['model']['gae_lambda'],
                clip_range=config['model']['clip_range'],
                ent_coef=config['model']['ent_coef'],
                vf_coef=config['model']['vf_coef'],
                verbose=1,
                device=config['model']['device'],
                policy_kwargs={
                    "features_extractor_class": CustomCNN,
                    "features_extractor_kwargs": config['model']['policy_kwargs']['features_extractor_kwargs'],
                    "net_arch": config['model']['policy_kwargs']['net_arch']
                }
            )
        else:
            model.set_env(env)
        
        stop_on_plateau = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=config['training']['max_no_improvement_evals'],
            min_evals=config['training']['min_evals'],
            verbose=1
        )
        best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
        print(f"Ścieżka zapisu najlepszego modelu (katalog): {best_model_save_path}")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=best_model_save_path,
            eval_freq=config['training']['eval_freq'],
            deterministic=True,
            render=False,
            callback_after_eval=stop_on_plateau
        )

        level_timesteps = 0
        while level_timesteps < max_steps and total_timesteps < config['training']['total_timesteps']:
            model.learn(
                total_timesteps=steps_per_iteration,
                reset_num_timesteps=False,
                callback=[eval_callback, train_progress_callback],
                progress_bar=use_progress_bar
            )
            level_timesteps += steps_per_iteration
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

            best_model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['best_model_path']))
            if os.path.exists(best_model_path_absolute):
                for attempt in range(5):
                    try:
                        with open(best_model_path_absolute, 'rb') as f:
                            pass
                        shutil.copy(best_model_path_absolute, model_path_absolute)
                        best_model_backup_path = best_model_path_absolute + '.backup'
                        shutil.copy(best_model_path_absolute, best_model_backup_path)
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
                print(f"Plik {best_model_path_absolute} nie istnieje. Pomijam kopiowanie.")

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

            if config['training']['enable_testing'] and not test_started:
                threading.Thread(
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
                ).start()
                test_started = True

        env.close()
        eval_env.close()

    # Kontynuacja treningu
    if total_timesteps < config['training']['total_timesteps']:
        print(f"\nKontynuuję trening z ostatnim grid_size={curriculum_grid_sizes[-1]} aż do {config['training']['total_timesteps']} kroków")
        set_grid_size(curriculum_grid_sizes[-1])
        env = make_vec_env(make_env(render_mode=None), n_envs=config['training']['n_envs'], vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(make_env(render_mode=None), n_envs=1, vec_env_cls=SubprocVecEnv)
        model.set_env(env)
        
        stop_on_plateau = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=config['training']['max_no_improvement_evals'],
            min_evals=config['training']['min_evals'],
            verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=best_model_save_path,
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

            best_model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['best_model_path']))
            if os.path.exists(best_model_path_absolute):
                for attempt in range(5):
                    try:
                        with open(best_model_path_absolute, 'rb') as f:
                            pass
                        shutil.copy(best_model_path_absolute, model_path_absolute)
                        best_model_backup_path = best_model_path_absolute + '.backup'
                        shutil.copy(best_model_path_absolute, best_model_backup_path)
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
                print(f"Plik {best_model_path_absolute} nie istnieje. Pomijam kopiowanie.")

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

            if config['training']['enable_testing'] and not test_started:
                threading.Thread(
                    target=test_thread,
                    args=(
                        model_path_absolute,
                        os.path.join(base_dir, config['paths']['test_model_path']),
                        os.path.join(base_dir, config['paths']['test_log_path']),
                        os.path.join(base_dir, config['paths']['test_csv_path']),
                        os.path.join(base_dir, config['paths']['test_progress_path']),
                        curriculum_grid_sizes[-1],
                        test_stop_event
                    ),
                    daemon=True
                ).start()
                test_started = True

        env.close()
        eval_env.close()

    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", os.path.join(base_dir, config['paths']['test_log_path']))

if __name__ == "__main__":
    model = None
    try:
        train(use_progress_bar=True)
    except KeyboardInterrupt:
        print("Przerwano trening.")
        import gc
        test_stop_event = None
        for obj in gc.get_objects():
            if isinstance(obj, threading.Event) and obj.is_set() is False:
                test_stop_event = obj
                break
        if test_stop_event is not None:
            test_stop_event.set()
            print("Wysłano sygnał zakończenia do wątku testującego.")
        sys.exit(0)
    except Exception as e:
        print(f"Błąd podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", os.path.join(base_dir, config['paths']['test_log_path']))