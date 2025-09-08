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
from model import make_env

# Wczytaj konfigurację względem lokalizacji pliku train.py
base_dir = os.path.dirname(os.path.dirname(__file__))  # Katalog nadrzędny (projekt/)
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Callback do zapisu postępu treningu do CSV
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

# Funkcja testowania w osobnym wątku z zapisem logów i wyników do CSV
def test_thread(model_path, test_model_path, log_path, test_csv_path, test_progress_path):
    current_model_timestamp = 0
    while True:
        env = make_env(render_mode="human")()
        model = None
        try:
            new_timestamp = os.path.getmtime(model_path)
            if new_timestamp > current_model_timestamp:
                with open(log_path, 'a') as f:
                    f.write(f"[{time.ctime()}] Ładuję nowy model...\n")
                shutil.copy(model_path, test_model_path)
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
        env.close()
        with open(log_path, 'a') as f:
            f.write(f"[{time.ctime()}] Epizod zakończony. Sprawdzam nowy model...\n")
        time.sleep(1)

def train(use_progress_bar=True):
    # Utwórz katalogi względem katalogu nadrzędnego
    models_dir = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
    logs_dir = os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir']))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Utwórz środowisko
    env = make_vec_env(make_env(render_mode=None), n_envs=config['training']['n_envs'], vec_env_cls=SubprocVecEnv)
    
    # Oblicz steps_per_iteration dynamicznie
    steps_per_iteration = config['training']['n_envs'] * config['model']['n_steps']
    
    # Sprawdź, czy model istnieje i zapytaj o kontynuację
    model = None
    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    print(f"Sprawdzam istnienie modelu w: {model_path_absolute}")
    if os.path.exists(model_path_absolute):
        choice = input(f"Model {model_path_absolute} istnieje. Kontynuować trening z tego modelu? [y/n]: ").strip().lower()
        if choice == 'y':
            print("Ładuję istniejący model...")
            model = PPO.load(model_path_absolute, env=env)
            total_timesteps = model.num_timesteps
            print(f"Załadowano model z timesteps: {total_timesteps}")
        elif choice == 'n':
            print("Tworzę nowy model...")
            backup_csv = os.path.join(base_dir, config['paths']['train_csv_path']) + '.backup'
            if os.path.exists(os.path.join(base_dir, config['paths']['train_csv_path'])):
                shutil.copy(os.path.join(base_dir, config['paths']['train_csv_path']), backup_csv)
                print(f"Stworzono kopię zapasową CSV w {backup_csv}")
            for p in [config['paths']['train_csv_path'], config['paths']['test_csv_path']]:
                try:
                    full_path = os.path.join(base_dir, p)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                        print(f"Usunięto plik: {full_path}")
                except Exception as e:
                    print(f"Nie udało się usunąć {full_path}: {e}")
        else:
            print("Nieprawidłowy wybór. Kończę program.")
            env.close()
            return
    else:
        print(f"Model {model_path_absolute} nie istnieje. Tworzę nowy model...")
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
            policy_kwargs={"net_arch": config['model']['policy_kwargs']['net_arch']}
        )

    # Callback do zatrzymania na plateau
    stop_on_plateau = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config['training']['max_no_improvement_evals'],
        min_evals=config['training']['min_evals'],
        verbose=1
    )
    eval_env = make_vec_env(make_env(render_mode=None), n_envs=1, vec_env_cls=SubprocVecEnv)
    # Ustaw best_model_save_path na katalog models, a nie na pełną ścieżkę do pliku
    best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
    print(f"Ścieżka zapisu najlepszego modelu (katalog): {best_model_save_path}")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,  # Katalog, a nie plik
        log_path=best_model_save_path,
        eval_freq=config['training']['eval_freq'],
        deterministic=True,
        render=False,
        callback_after_eval=stop_on_plateau
    )

    # Pętla treningu
    total_timesteps = 0
    test_started = False
    train_progress_callback = TrainProgressCallback(os.path.join(base_dir, config['paths']['train_csv_path']))
    print(f"Oczekiwane kroki na iterację: {steps_per_iteration} (n_envs={config['training']['n_envs']} * n_steps={config['model']['n_steps']})")
    while total_timesteps < config['training']['total_timesteps']:
        model.learn(
            total_timesteps=steps_per_iteration,
            reset_num_timesteps=False,
            callback=[eval_callback, train_progress_callback],
            progress_bar=use_progress_bar
        )
        total_timesteps += steps_per_iteration
        # Zapisz model
        model.save(model_path_absolute)
        print(f"Model zapisany w: {model_path_absolute}")

        # Poczekaj na zakończenie wszystkich operacji zapisu
        time.sleep(5)  # Zwiększone opóźnienie do 5 sekund

        # Tworzenie kopii zapasowej modelu
        backup_model = model_path_absolute + '.backup'
        try:
            shutil.copy(model_path_absolute, backup_model)
            print(f"Stworzono kopię zapasową modelu w {backup_model}")
        except Exception as e:
            print(f"Nie udało się stworzyć kopii zapasowej modelu: {e}")

        # Próba skopiowania najlepszego modelu
        best_model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['best_model_path']))
        if os.path.exists(best_model_path_absolute):
            for attempt in range(5):  # Zwiększamy liczbę prób do 5
                try:
                    with open(best_model_path_absolute, 'rb') as f:
                        pass  # Próba otwarcia pliku w trybie odczytu
                    shutil.copy(best_model_path_absolute, model_path_absolute)
                    # Tworzenie kopii zapasowej najlepszego modelu
                    best_model_backup = best_model_path_absolute + '.backup'
                    shutil.copy(best_model_path_absolute, best_model_backup)
                    print(f"Zaktualizowano najlepszy model: {model_path_absolute}")
                    print(f"Stworzono kopię zapasową najlepszego modelu w {best_model_backup}")
                    break
                except PermissionError as e:
                    print(f"Próba {attempt + 1}/5: Nie udało się skopiować best_model.zip do {model_path_absolute}: {e}")
                    time.sleep(3)  # Zwiększone opóźnienie do 3 sekund
                except Exception as e:
                    print(f"Nie udało się skopiować best_model.zip do {model_path_absolute}: {e}")
                    break
            else:
                print(f"Nie udało się skopiować best_model.zip po 5 próbach.")
        else:
            print(f"Plik {best_model_path_absolute} nie istnieje. Pomijam kopiowanie najlepszego modelu.")
        
        # Sprawdź, czy train_progress.csv istnieje przed wywołaniem skryptu
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
                    os.path.join(base_dir, config['paths']['test_progress_path'])
                ),
                daemon=True
            ).start()
            test_started = True

    env.close()
    eval_env.close()

if __name__ == "__main__":
    model = None
    try:
        train(use_progress_bar=True)
    except KeyboardInterrupt:
        print("Przerwano trening. Zapisuję model...")
        if model is not None:
            model.save(os.path.join(base_dir, config['paths']['model_path']))
        else:
            print("Brak modelu do zapisania.")
        exit(0)
    print("Trening zakończony! Testowanie działa w tle. Logi testowania w:", os.path.join(base_dir, config['paths']['test_log_path']))