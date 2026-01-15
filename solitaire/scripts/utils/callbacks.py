import os
import csv
import sys
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable_policy


class TrainProgressCallback(BaseCallback):
    """
    Callback dla logowania postępów treningu Solitaire
    """
    def __init__(self, log_path, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.csv_path = log_path
        self.initial_timesteps = initial_timesteps
        self.last_logged = 0
        
        # Pre-allocated lists
        self.episode_scores = []
        self.episode_wins = []
        self.episode_foundations = []
        self.episode_moves = []
        
        # Sprawdź czy CSV istnieje (wznowienie treningu)
        self._csv_exists = os.path.exists(log_path)

    def _on_step(self) -> bool:
        """Zbiera dane z zakończonych epizodów"""
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        # ✅ Zbieraj dane z zakończonych epizodów
        for info, done in zip(infos, dones):
            if done:
                # Użyj score z info (który teraz jest aktualizowany w model.py)
                score = info.get('score', 0)
                won = info.get('won', False)
                foundations_filled = sum(info.get('foundations', [0, 0, 0, 0]))
                moves = info.get('moves', 0)
                
                self.episode_scores.append(score)
                self.episode_wins.append(1.0 if won else 0.0)
                self.episode_foundations.append(foundations_filled)
                self.episode_moves.append(moves)
        
        # Logowanie co 1000 kroków
        if any(dones) and (self.num_timesteps + self.initial_timesteps) - self.last_logged >= 1000:
            # ✅ Sprawdź czy mamy wystarczająco danych do logowania
            if not self.episode_scores:
                return True  # Pomiń logowanie jeśli nie ma danych
            
            ep_buffer = self.model.ep_info_buffer
            
            ep_rew_mean = np.mean([ep['r'] for ep in ep_buffer]) if ep_buffer else None
            ep_len_mean = np.mean([ep['l'] for ep in ep_buffer]) if ep_buffer else None
            
            mean_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
            max_score = np.max(self.episode_scores) if self.episode_scores else 0.0
            win_rate = np.mean(self.episode_wins) if self.episode_wins else 0.0
            mean_foundations = np.mean(self.episode_foundations) if self.episode_foundations else 0.0
            mean_moves = np.mean(self.episode_moves) if self.episode_moves else 0.0
            
            # Pobierz losses
            policy_loss = getattr(self.model, '_last_policy_loss', None)
            value_loss = getattr(self.model, '_last_value_loss', None)
            entropy_loss = getattr(self.model, '_last_entropy_loss', None)
            
            try:
                write_header = not self._csv_exists
                with open(self.csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow([
                            'timesteps', 
                            'mean_reward', 
                            'mean_ep_length', 
                            'mean_score', 
                            'max_score', 
                            'win_rate',
                            'mean_foundations',
                            'mean_moves',
                            'policy_loss', 
                            'value_loss', 
                            'entropy_loss'
                        ])
                        self._csv_exists = True
                    
                    writer.writerow([
                        self.num_timesteps + self.initial_timesteps, 
                        ep_rew_mean, 
                        ep_len_mean, 
                        mean_score,
                        max_score,
                        win_rate,
                        mean_foundations,
                        mean_moves,
                        policy_loss,
                        value_loss,
                        entropy_loss
                    ])
                
                # Reset agregacji
                self.last_logged = self.num_timesteps + self.initial_timesteps
                self.episode_scores = []
                self.episode_wins = []
                self.episode_foundations = []
                self.episode_moves = []
            except Exception as e:
                print(f"Warning: Failed to log training progress: {e}")
        
        return True


class CustomEvalCallback(BaseCallback):
    """
    Custom EvalCallback z obsługą MaskablePPO i automatycznym plotowaniem
    Zastępuje standardowy EvalCallback, który nie obsługuje maskowania akcji.
    """
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq, 
                 plot_script_path=None, plot_interval=1, callback_on_new_best=None,
                 deterministic=True, render=False, verbose=1, warn=True, n_eval_episodes=5):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.plot_script_path = plot_script_path
        self.plot_interval = plot_interval
        self.callback_on_new_best = callback_on_new_best
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.n_eval_episodes = n_eval_episodes
        
        self.eval_count = 0
        self.best_mean_reward = -np.inf
        
        # Utwórz katalogi
        if best_model_save_path is not None:
            os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
            
    def _init_callback(self) -> None:
        # Does not work (warnings) with SubprocVecEnv
        # if not isinstance(self.eval_env, VecEnv):
        #     self.eval_env = DummyVecEnv([lambda: self.eval_env])
        
        # Inicjalizacja sub-callbacka (np. StopTrainingOnNoModelImprovement)
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)
            # Ustaw parent dla StopTrainingOnNoModelImprovement, 
            # ponieważ wymaga on dostępu do last_mean_reward z EvalCallback
            self.callback_on_new_best.parent = self

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            # --- EWALUACJA (Maskable) ---
            if self.verbose > 0:
                print(f"Evaluating policy at {self.num_timesteps} timesteps...")
            
            # Użyj evaluate_policy z sb3_contrib
            episode_rewards, episode_lengths = evaluate_maskable_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
            )
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            
            # Logowanie
            if self.verbose > 0:
                print(f"Eval result: reward={mean_reward:.2f} +/- {std_reward:.2f} | length={mean_ep_length:.2f}")
            
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", float(mean_ep_length))
            
            # Zapisz najlepszy model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"🔥 New best mean reward! {self.best_mean_reward:.2f} -> {mean_reward:.2f}")
                self.best_mean_reward = mean_reward
                
                if self.best_model_save_path is not None:
                    save_path = os.path.join(self.best_model_save_path, "best_model.zip")
                    self.model.save(save_path)
                    print(f"📦 Saved best model to {save_path}")
                
                # Trigger callback
                if self.callback_on_new_best is not None:
                    self.callback_on_new_best.on_step()

            # Plot co plot_interval ewaluacji
            if self.plot_script_path and self.eval_count % self.plot_interval == 0:
                try:
                    import subprocess
                    subprocess.run([sys.executable, self.plot_script_path], check=False)
                except Exception as e:
                    print(f"⚠️ Plotting failed: {e}")
        
        return True


class LossRecorderCallback(BaseCallback):
    """
    Callback do zapisywania loss values do modelu
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Zapisz losses jeśli są dostępne
        if hasattr(self.model, 'logger') and self.model.logger:
            try:
                # SB3 stores these in logger
                self.model._last_policy_loss = self.model.logger.name_to_value.get('train/policy_loss', None)
                self.model._last_value_loss = self.model.logger.name_to_value.get('train/value_loss', None)
                self.model._last_entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
            except:
                pass
        
        return True


class EntropySchedulerCallback(BaseCallback):
    """
    Callback dla liniowego zmniejszania współczynnika entropii
    """
    def __init__(self, initial_ent_coef, min_ent_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.min_ent_coef = min_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Liniowa interpolacja
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        new_ent_coef = self.initial_ent_coef - progress * (self.initial_ent_coef - self.min_ent_coef)
        self.model.ent_coef = max(new_ent_coef, self.min_ent_coef)
        
        return True


class WinTrackerCallback(BaseCallback):
    """
    Callback dla śledzenia wygranych i statystyk gry
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_games = 0
        self.total_wins = 0
        self.last_100_wins = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for info, done in zip(infos, dones):
            if done:
                self.total_games += 1
                won = info.get('won', False)
                if won:
                    self.total_wins += 1
                    self.last_100_wins.append(1)
                else:
                    self.last_100_wins.append(0)
                
                # Zachowaj tylko ostatnie 100
                if len(self.last_100_wins) > 100:
                    self.last_100_wins.pop(0)
        
        # Log co 10000 kroków
        if self.num_timesteps % 10000 == 0 and self.total_games > 0:
            win_rate = self.total_wins / self.total_games
            win_rate_100 = np.mean(self.last_100_wins) if self.last_100_wins else 0.0
            
            if self.verbose > 0:
                print(f"\n[Win Stats] Games: {self.total_games} | "
                      f"Overall Win Rate: {win_rate:.2%} | "
                      f"Last 100: {win_rate_100:.2%}")
        
        return True


class PeriodicSaveCallback(BaseCallback):
    """
    Callback do zapisywania modelu co N ewaluacji, niezależnie od wyniku
    """
    def __init__(self, save_path, eval_freq, save_interval=10, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.save_interval = save_interval  # Zapisz co N ewaluacji
        self.eval_count = 0
        self.last_save_eval = 0
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Sprawdzamy czy właśnie była ewaluacja
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            self.eval_count += 1
            
            # Zapisz model co save_interval ewaluacji
            if self.eval_count - self.last_save_eval >= self.save_interval:
                try:
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"✅ Model saved to {self.save_path} (eval #{self.eval_count})")
                    self.last_save_eval = self.eval_count
                except Exception as e:
                    print(f"❌ Failed to save model: {e}")
        
        return True
