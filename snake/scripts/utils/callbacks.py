import os
import csv
import numpy as np
import subprocess
import sys
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class TrainProgressCallback(BaseCallback):
    """Callback zapisujący postęp treningu do CSV"""
    def __init__(self, csv_path, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.initial_timesteps = initial_timesteps
        self.header_written = False
        self.last_logged = 0
        self.episode_scores = []
        self.episode_snake_lengths = []
        self.episode_steps_per_apple = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and 'score' in info:
                self.episode_scores.append(info['score'])
                self.episode_snake_lengths.append(info.get('snake_length', 3))
                self.episode_steps_per_apple.append(info.get('steps_per_apple', 0))
        
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            if (self.num_timesteps + self.initial_timesteps) - self.last_logged >= 1000:
                ep_rew_mean = self.model.ep_info_buffer and np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) or None
                ep_len_mean = self.model.ep_info_buffer and np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]) or None
                
                mean_score = np.mean(self.episode_scores) if self.episode_scores else 0
                max_score = np.max(self.episode_scores) if self.episode_scores else 0
                mean_snake_length = np.mean(self.episode_snake_lengths) if self.episode_snake_lengths else 3
                mean_steps_per_apple = np.mean(self.episode_steps_per_apple) if self.episode_steps_per_apple else 0
                
                progress_score = ep_rew_mean + 0.1 * mean_snake_length - 0.05 * mean_steps_per_apple if ep_rew_mean is not None else 0
                
                policy_loss = getattr(self.model, '_last_policy_loss', None)
                value_loss = getattr(self.model, '_last_value_loss', None)
                entropy_loss = getattr(self.model, '_last_entropy_loss', None)
                
                try:
                    write_header = not os.path.exists(self.csv_path)
                    with open(self.csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if write_header:
                            writer.writerow(['timesteps', 'mean_reward', 'mean_ep_length', 'mean_score', 'max_score', 'mean_snake_length', 'mean_steps_per_apple', 'progress_score', 'policy_loss', 'value_loss', 'entropy_loss'])
                        writer.writerow([
                            self.num_timesteps + self.initial_timesteps, 
                            ep_rew_mean, 
                            ep_len_mean, 
                            mean_score,
                            max_score,
                            mean_snake_length,
                            mean_steps_per_apple,
                            progress_score,
                            policy_loss,
                            value_loss,
                            entropy_loss
                        ])
                    self.last_logged = self.num_timesteps + self.initial_timesteps
                    self.episode_scores = []
                    self.episode_snake_lengths = []
                    self.episode_steps_per_apple = []
                except Exception as e:
                    print(f"Błąd zapisu train_progress.csv: {e}")
        return True


class CustomEvalCallback(EvalCallback):
    """Callback ewaluacyjny z automatycznym generowaniem wykresów"""
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
            continue_training = super()._on_step()
            self.eval_count += 1
            total_timesteps = self.model.num_timesteps + self.initial_timesteps
            
            # Import z training_utils
            from utils.training_utils import save_training_state
            save_training_state(self.model, self.model.env, self.eval_env, total_timesteps, self.best_model_save_path)
            print(f"Zaktualizowano najnowszy model po {total_timesteps} krokach z mean_reward={self.last_mean_reward}")

            if self.best_mean_reward < self.last_mean_reward:
                self.model.save(os.path.join(self.best_model_save_path, f'best_model_{total_timesteps}.zip'))
                print(f"New best model saved at {total_timesteps} timesteps with mean reward {self.last_mean_reward}")
            
            if self.eval_count % self.plot_interval == 0:
                try:
                    subprocess.run([sys.executable, self.plot_script_path], check=True)
                    print(f"Wygenerowano wykres po {self.eval_count} walidacji.")
                except Exception as e:
                    print(f"Błąd podczas generowania wykresu: {e}")
            
            if not continue_training:
                print(f"\n{'='*70}")
                print(f"🛑 TRENING ZATRZYMANY przez StopTrainingOnNoModelImprovement")
                print(f"{'='*70}\n")
                return False
        
        return True


class LossRecorderCallback(BaseCallback):
    """Callback zapisujący wartości loss dla train_progress"""
    def _on_step(self) -> bool:
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                if hasattr(self.model.logger, 'name_to_value'):
                    losses = self.model.logger.name_to_value
                    self.model._last_policy_loss = losses.get('train/policy_gradient_loss', None)
                    self.model._last_value_loss = losses.get('train/value_loss', None)
                    self.model._last_entropy_loss = losses.get('train/entropy_loss', None)
            except Exception:
                pass
        return True