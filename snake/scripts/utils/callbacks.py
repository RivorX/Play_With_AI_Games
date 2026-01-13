import os
import csv
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class TrainProgressCallback(BaseCallback):
    """
    ✅ FIXED: Przy wznowieniu treningu dołącza dane zamiast nadpisywać
    """
    def __init__(self, csv_path, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.initial_timesteps = initial_timesteps
        self.last_logged = 0
        
        # Pre-allocated lists
        self.episode_scores = []
        self.episode_snake_lengths = []
        self.episode_steps_per_apple = []
        self.episode_map_occupancy = []
        
        # ✅ FIX: Sprawdź czy CSV istnieje (wznowienie treningu)
        self._csv_exists = os.path.exists(csv_path)

    def _on_step(self) -> bool:
        """Zbiera dane z zakończonych epizodów"""
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for info, done in zip(infos, dones):
            if done and 'score' in info:
                score = info['score']
                snake_length = info.get('snake_length', 3)
                grid_size = info.get('grid_size', 16)
                
                self.episode_scores.append(score)
                self.episode_snake_lengths.append(snake_length)
                self.episode_steps_per_apple.append(info.get('steps_per_apple', 0))
                
                # Map occupancy
                map_occupancy = (snake_length / (grid_size * grid_size)) * 100.0
                self.episode_map_occupancy.append(map_occupancy)
        
        # Logowanie co 1000 kroków
        if any(dones) and (self.num_timesteps + self.initial_timesteps) - self.last_logged >= 1000:
            ep_buffer = self.model.ep_info_buffer
            
            ep_rew_mean = np.mean([ep['r'] for ep in ep_buffer]) if ep_buffer else None
            ep_len_mean = np.mean([ep['l'] for ep in ep_buffer]) if ep_buffer else None
            
            mean_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
            max_score = np.max(self.episode_scores) if self.episode_scores else 0.0
            mean_snake_length = np.mean(self.episode_snake_lengths) if self.episode_snake_lengths else 3.0
            mean_steps_per_apple = np.mean(self.episode_steps_per_apple) if self.episode_steps_per_apple else 0.0
            mean_map_occupancy = np.mean(self.episode_map_occupancy) if self.episode_map_occupancy else 0.0
            
            # Pobierz losses
            policy_loss = getattr(self.model, '_last_policy_loss', None)
            value_loss = getattr(self.model, '_last_value_loss', None)
            entropy_loss = getattr(self.model, '_last_entropy_loss', None)
            
            # ✅ FIX: Nagłówek tylko jeśli CSV nie istnieje
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
                            'mean_snake_length', 
                            'mean_steps_per_apple', 
                            'mean_map_occupancy',
                            'policy_loss', 
                            'value_loss', 
                            'entropy_loss'
                        ])
                        self._csv_exists = True  # Oznacz że istnieje
                    
                    writer.writerow([
                        self.num_timesteps + self.initial_timesteps, 
                        ep_rew_mean, 
                        ep_len_mean, 
                        mean_score,
                        max_score,
                        mean_snake_length,
                        mean_steps_per_apple,
                        mean_map_occupancy,
                        policy_loss,
                        value_loss,
                        entropy_loss
                    ])
                
                # Reset agregacji
                self.last_logged = self.num_timesteps + self.initial_timesteps
                self.episode_scores.clear()
                self.episode_snake_lengths.clear()
                self.episode_steps_per_apple.clear()
                self.episode_map_occupancy.clear()
                
            except Exception as e:
                print(f"Błąd zapisu train_progress.csv: {e}")
        
        return True


class CustomEvalCallback(EvalCallback):
    """
    ✅ FIXED: 
    1. Wyłączona duplikacja zapisu best_model.zip (używa parent class)
    2. Dodane zapisywanie policy.pth w odpowiedniej lokalizacji
    3. Usunięte save_training_state() (duplikacja z parent)
    """
    def __init__(self, eval_env, callback_on_new_best=None, callback_after_eval=None, 
                 best_model_save_path=None, log_path=None, eval_freq=10000, 
                 deterministic=True, render=False, verbose=1, warn=True, 
                 n_eval_episodes=5, plot_interval=3, plot_script_path=None, initial_timesteps=0):
        super().__init__(
            eval_env, 
            callback_on_new_best=callback_on_new_best, 
            callback_after_eval=callback_after_eval,
            best_model_save_path=best_model_save_path, 
            log_path=log_path, 
            eval_freq=eval_freq,
            deterministic=deterministic, 
            render=render, 
            verbose=verbose, 
            warn=warn,
            n_eval_episodes=n_eval_episodes
        )
        self.eval_count = 0
        self.plot_interval = plot_interval
        self.plot_script_path = plot_script_path
        self.initial_timesteps = initial_timesteps

    def _on_step(self) -> bool:
        """Wykonuje ewaluację i zapisuje modele"""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 🔍 Zapisz stary best_mean_reward PRZED wywołaniem parent
            old_best_reward = self.best_mean_reward
            
            # ✅ Parent class zapisuje best_model.zip automatycznie
            continue_training = super()._on_step()
            self.eval_count += 1
            total_timesteps = self.model.num_timesteps + self.initial_timesteps
            
            # ✅ Zapisz bieżący model (snake_ppo_model.zip) po KAŻDEJ ewaluacji
            current_model_path = os.path.join(self.best_model_save_path, 'snake_ppo_model.zip')
            try:
                self.model.save(current_model_path)
                # 💾 Zapisz też timesteps do _state.pkl
                state_path = current_model_path.replace('.zip', '_state.pkl')
                import pickle
                with open(state_path, 'wb') as f:
                    pickle.dump({'total_timesteps': total_timesteps}, f)
                if self.verbose > 0:
                    print(f"💾 Zapisano bieżący model: {current_model_path}")
                    print(f"💾 Zapisano timesteps: {total_timesteps:,}")
            except Exception as e:
                print(f"⚠️ Błąd zapisu snake_ppo_model.zip: {e}")
            
            # ✅ FIXED: Sprawdź czy parent zaktualizował best (nowy rekord!)
            if self.best_mean_reward > old_best_reward:
                # Parent class już zapisał best_model.zip, teraz zapisz policy.pth
                policy_pth_path = os.path.join(self.best_model_save_path, 'policy.pth')
                try:
                    torch.save(self.model.policy.state_dict(), policy_pth_path)
                    print(f"✅ Nowy BEST! Reward={self.best_mean_reward:.2f} (poprz: {old_best_reward:.2f}) | Timesteps={total_timesteps:,}")
                    print(f"💾 Zapisano policy.pth: {policy_pth_path}")
                except Exception as e:
                    print(f"⚠️ Błąd zapisu policy.pth: {e}")
            
            # ✅ Generuj wykresy co N ewaluacji
            if self.eval_count % self.plot_interval == 0:
                self._generate_plots()
            
            # Stop training jeśli callback zwrócił False
            if not continue_training:
                print(f"\n{'='*70}")
                print(f"🛑 TRENING ZATRZYMANY przez StopTrainingOnNoModelImprovement")
                print(f"{'='*70}\n")
                return False
        
        return True
    
    def _generate_plots(self):
        """Generuj wykresy treningu i gradientów"""
        # === 1. WYKRES TRENINGU ===
        try:
            if self.plot_script_path:
                import importlib.util
                spec = importlib.util.spec_from_file_location("plot_module", self.plot_script_path)
                if spec and spec.loader:
                    plot_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(plot_module)
                    
                    logs_dir = os.path.join(os.path.dirname(self.plot_script_path), '..', '..', 'logs')
                    csv_path = os.path.normpath(os.path.join(logs_dir, 'train_progress.csv'))
                    output_path = os.path.normpath(os.path.join(logs_dir, 'training_progress.png'))
                    
                    if os.path.exists(csv_path):
                        plot_module.plot_train_progress(csv_path, output_path)
                        print(f"📊 Wygenerowano wykres treningu po {self.eval_count} walidacji.")
                    else:
                        print(f"⚠️ CSV nie istnieje jeszcze: {csv_path}")
        
        except Exception as e:
            print(f"⚠️ Błąd podczas generowania wykresu treningu: {e}")
        
        # === 2. WYKRES GRADIENT MONITORA ===
        try:
            from utils.gradient_monitor import plot_gradient_monitor
            
            logs_dir = os.path.join(os.path.dirname(self.plot_script_path), '..', '..', 'logs')
            gradient_csv = os.path.normpath(os.path.join(logs_dir, 'gradient_monitor.csv'))
            gradient_plot = os.path.normpath(os.path.join(logs_dir, 'gradient_monitor.png'))
            
            if os.path.exists(gradient_csv):
                plot_gradient_monitor(gradient_csv, gradient_plot)
                print(f"📊 Wygenerowano wykres gradientów po {self.eval_count} walidacji.")
        
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Błąd podczas generowania wykresu gradientów: {e}")


class LossRecorderCallback(BaseCallback):
    """Callback zapisujący wartości loss"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._has_logger = False
        self._checked_logger = False

    def _on_step(self) -> bool:
        """Zapisuje losses do modelu"""
        if not self._checked_logger:
            self._has_logger = (
                hasattr(self.model, 'logger') and 
                self.model.logger is not None and
                hasattr(self.model.logger, 'name_to_value')
            )
            self._checked_logger = True
        
        if self._has_logger:
            try:
                losses = self.model.logger.name_to_value
                self.model._last_policy_loss = losses.get('train/policy_gradient_loss')
                self.model._last_value_loss = losses.get('train/value_loss')
                self.model._last_entropy_loss = losses.get('train/entropy_loss')
            except (AttributeError, KeyError):
                self._has_logger = False
        
        return True


class EntropySchedulerCallback(BaseCallback):
    """Callback aktualizujący współczynnik entropii"""
    def __init__(self, entropy_schedule_fn, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.entropy_schedule_fn = entropy_schedule_fn
        self.initial_timesteps = initial_timesteps
        self._last_logged = 0
    
    def _on_step(self) -> bool:
        """Aktualizuje ent_coef na podstawie progress_remaining"""
        total_timesteps = self.num_timesteps + self.initial_timesteps
        
        if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps > 0:
            total_target = self.model._total_timesteps
        elif hasattr(self.model, 'num_timesteps'):
            total_target = max(total_timesteps, 1)
        else:
            total_target = 1
        
        progress_remaining = max(0.0, min(1.0, 1.0 - (total_timesteps / total_target)))
        new_ent_coef = self.entropy_schedule_fn(progress_remaining)
        self.model.ent_coef = new_ent_coef
        
        # Loguj co 100'000 kroków
        if total_timesteps - self._last_logged >= 100000:
            print(f"[ENTROPY SCHEDULE] timesteps={total_timesteps}, ent_coef={new_ent_coef:.6f}, progress={progress_remaining:.2%}")
            self._last_logged = total_timesteps
        
        return True


class VictoryTrackerCallback(BaseCallback):
    """
    ✅ FIXED: Przy wznowieniu treningu dołącza do logu zamiast nadpisywać
    """
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'victories.log')
        os.makedirs(log_dir, exist_ok=True)
        
        # ✅ FIX: Inicjalizuj TYLKO jeśli plik nie istnieje
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("SNAKE AI - VICTORY LOG\n")
                f.write("Full Board Completions Tracker\n")
                f.write("="*70 + "\n\n")
    
    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals['infos']):
            if 'snake_length' in info and 'grid_size' in info:
                snake_len = info['snake_length']
                grid_size = info['grid_size']
                
                if snake_len == grid_size * grid_size:
                    self._log_victory(info, env_idx=i)
        
        return True
    
    def _log_victory(self, info: dict, env_idx: int):
        """Zapisz zwycięstwo do pliku"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_timesteps = self.num_timesteps
        victory_count = self._count_victories() + 1
        
        grid_size = info['grid_size']
        snake_length = info['snake_length']
        total_reward = info.get('total_reward', 0)
        steps_per_apple = info.get('steps_per_apple', 0)
        
        log_entry = (
            f"\n{'='*70}\n"
            f"🎉 VICTORY #{victory_count}\n"
            f"{'='*70}\n"
            f"Timestamp:         {timestamp}\n"
            f"Total Timesteps:   {total_timesteps:,}\n"
            f"Environment:       #{env_idx}\n"
            f"Grid Size:         {grid_size}x{grid_size}\n"
            f"Snake Length:      {snake_length} / {grid_size * grid_size} (FULL BOARD!)\n"
            f"Steps per Apple:   {steps_per_apple:.2f}\n"
            f"Total Reward:      {total_reward:.2f}\n"
            f"{'='*70}\n"
        )
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            print("\n" + "🎉" * 35)
            print(log_entry)
            print("🎉" * 35 + "\n")
            
            if self.verbose > 0:
                print(f"Victory logged to {self.log_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to write victory log: {e}")
    
    def _count_victories(self) -> int:
        """Policz ile było zwycięstw"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return content.count('🎉 VICTORY #')
        except:
            return 0