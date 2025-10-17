import os
import csv
import numpy as np
import subprocess
import sys
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class TrainProgressCallback(BaseCallback):
    """
    ZOPTYMALIZOWANY Callback zapisujący postęp treningu do CSV
    - Dodano tracking średniej zajętości mapy (map_occupancy)
    - Usunięto progress_score
    - Optymalizacja: cache list comprehensions, mniej operacji
    """
    def __init__(self, csv_path, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.initial_timesteps = initial_timesteps
        self.last_logged = 0
        
        # Listy do agregacji (pre-allocated dla szybkości)
        self.episode_scores = []
        self.episode_snake_lengths = []
        self.episode_steps_per_apple = []
        self.episode_map_occupancy = []  # NOWE: % zajętości mapy

    def _on_step(self) -> bool:
        """Zbiera dane z zakończonych epizodów"""
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        # OPTYMALIZACJA: zip zamiast enumerate gdy nie używamy i
        for info, done in zip(infos, dones):
            if done and 'score' in info:
                score = info['score']
                snake_length = info.get('snake_length', 3)
                grid_size = info.get('grid_size', 16)
                
                # Append do list (szybsze niż extend)
                self.episode_scores.append(score)
                self.episode_snake_lengths.append(snake_length)
                self.episode_steps_per_apple.append(info.get('steps_per_apple', 0))
                
                # NOWE: Oblicz zajętość mapy (% wypełnienia)
                # snake_length / (grid_size * grid_size) * 100
                map_occupancy = (snake_length / (grid_size * grid_size)) * 100.0
                self.episode_map_occupancy.append(map_occupancy)
        
        # Logowanie co 1000 kroków (tylko jeśli są zakończone epizody)
        if any(dones) and (self.num_timesteps + self.initial_timesteps) - self.last_logged >= 1000:
            # OPTYMALIZACJA: Cache ep_info_buffer raz
            ep_buffer = self.model.ep_info_buffer
            
            # Oblicz średnie (z cache)
            ep_rew_mean = np.mean([ep['r'] for ep in ep_buffer]) if ep_buffer else None
            ep_len_mean = np.mean([ep['l'] for ep in ep_buffer]) if ep_buffer else None
            
            # Agregacja epizodów (defaulty gdy puste)
            mean_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
            max_score = np.max(self.episode_scores) if self.episode_scores else 0.0
            mean_snake_length = np.mean(self.episode_snake_lengths) if self.episode_snake_lengths else 3.0
            mean_steps_per_apple = np.mean(self.episode_steps_per_apple) if self.episode_steps_per_apple else 0.0
            
            # NOWE: Średnia zajętość mapy
            mean_map_occupancy = np.mean(self.episode_map_occupancy) if self.episode_map_occupancy else 0.0
            
            # Pobierz losses (jeśli dostępne)
            policy_loss = getattr(self.model, '_last_policy_loss', None)
            value_loss = getattr(self.model, '_last_value_loss', None)
            entropy_loss = getattr(self.model, '_last_entropy_loss', None)
            
            # Zapis do CSV
            try:
                write_header = not os.path.exists(self.csv_path)
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
                            'mean_map_occupancy',  # NOWE zamiast progress_score
                            'policy_loss', 
                            'value_loss', 
                            'entropy_loss'
                        ])
                    writer.writerow([
                        self.num_timesteps + self.initial_timesteps, 
                        ep_rew_mean, 
                        ep_len_mean, 
                        mean_score,
                        max_score,
                        mean_snake_length,
                        mean_steps_per_apple,
                        mean_map_occupancy,  # NOWE
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
    ZOPTYMALIZOWANY Callback ewaluacyjny z generowaniem wykresów
    - Dodano generowanie wykresu gradient monitora
    - Mniej duplikacji kodu
    - Bezpieczniejsze operacje I/O
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
            # Wywołaj ewaluację z parent class
            continue_training = super()._on_step()
            self.eval_count += 1
            total_timesteps = self.model.num_timesteps + self.initial_timesteps
            
            # Import lokalny (tylko gdy potrzebny)
            from utils.training_utils import save_training_state
            
            # Zapisz najnowszy stan
            save_training_state(
                self.model, 
                self.model.env, 
                self.eval_env, 
                total_timesteps, 
                self.best_model_save_path
            )
            print(f"Zaktualizowano najnowszy model po {total_timesteps} krokach z mean_reward={self.last_mean_reward:.2f}")

            # Zapisz best model jeśli poprawił się
            if self.best_mean_reward < self.last_mean_reward:
                best_path = os.path.join(self.best_model_save_path, f'best_model_{total_timesteps}.zip')
                self.model.save(best_path)
                print(f"✅ Nowy najlepszy model! Zapisano: {best_path} (reward={self.last_mean_reward:.2f})")
            
            # Generuj wykresy co N ewaluacji
            if self.eval_count % self.plot_interval == 0:
                # ===========================
                # 1. WYKRES TRENINGU
                # ===========================
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
                        else:
                            print(f"⚠️ Nie udało się załadować modułu plot")
                    
                except FileNotFoundError as e:
                    print(f"⚠️ Nie znaleziono pliku CSV: {e}")
                except Exception as e:
                    print(f"⚠️ Błąd podczas generowania wykresu treningu: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ===========================
                # 2. WYKRES GRADIENT MONITORA
                # ===========================
                try:
                    from utils.gradient_monitor import plot_gradient_monitor
                    
                    logs_dir = os.path.join(os.path.dirname(self.plot_script_path), '..', '..', 'logs')
                    gradient_csv = os.path.normpath(os.path.join(logs_dir, 'gradient_monitor.csv'))
                    gradient_plot = os.path.normpath(os.path.join(logs_dir, 'gradient_monitor.png'))
                    
                    if os.path.exists(gradient_csv):
                        plot_gradient_monitor(gradient_csv, gradient_plot)
                        print(f"📊 Wygenerowano wykres gradientów po {self.eval_count} walidacji.")
                    else:
                        if self.verbose > 0:
                            print(f"ℹ️ Gradient monitor CSV jeszcze nie istnieje: {gradient_csv}")
                
                except ImportError:
                    if self.verbose > 0:
                        print(f"⚠️ Nie udało się zaimportować gradient_monitor")
                except Exception as e:
                    print(f"⚠️ Błąd podczas generowania wykresu gradientów: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Stop training jeśli callback zwrócił False
            if not continue_training:
                print(f"\n{'='*70}")
                print(f"🛑 TRENING ZATRZYMANY przez StopTrainingOnNoModelImprovement")
                print(f"{'='*70}\n")
                return False
        
        return True


class LossRecorderCallback(BaseCallback):
    """
    ZOPTYMALIZOWANY Callback zapisujący wartości loss
    - Bezpieczniejszy dostęp do loggera
    - Mniej try-except (performance)
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._has_logger = False
        self._checked_logger = False

    def _on_step(self) -> bool:
        """Zapisuje losses do modelu (jeśli logger dostępny)"""
        # OPTYMALIZACJA: Sprawdź logger tylko raz
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
                # OPTYMALIZACJA: .get() zwraca None domyślnie (brak default arg)
                self.model._last_policy_loss = losses.get('train/policy_gradient_loss')
                self.model._last_value_loss = losses.get('train/value_loss')
                self.model._last_entropy_loss = losses.get('train/entropy_loss')
            except (AttributeError, KeyError):
                # Jeśli struktura się zmieniła, wyłącz flagę
                self._has_logger = False
        
        return True


class EntropySchedulerCallback(BaseCallback):
    """
    Callback aktualizujący współczynnik entropii według harmonogramu
    Podobnie jak learning rate, entropia maleje w czasie treningu
    """
    def __init__(self, entropy_schedule_fn, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.entropy_schedule_fn = entropy_schedule_fn
        self.initial_timesteps = initial_timesteps
        self._last_logged = 0
    
    def _on_step(self) -> bool:
        """Aktualizuje ent_coef na podstawie progress_remaining"""
        # Oblicz progress_remaining (1.0 na początku, 0.0 na końcu)
        total_timesteps = self.num_timesteps + self.initial_timesteps
        
        # Pobierz całkowitą liczbę kroków do treningu
        if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps > 0:
            total_target = self.model._total_timesteps
        elif hasattr(self.model, 'num_timesteps'):
            # Fallback - używamy aktualnej liczby kroków jako odniesienia
            total_target = max(total_timesteps, 1)
        else:
            total_target = 1
        
        progress_remaining = max(0.0, min(1.0, 1.0 - (total_timesteps / total_target)))
        
        # Oblicz nową wartość entropii
        new_ent_coef = self.entropy_schedule_fn(progress_remaining)
        
        # Zaktualizuj model
        self.model.ent_coef = new_ent_coef
        
        # Loguj co 10000 kroków
        if total_timesteps - self._last_logged >= 10000:
            print(f"[ENTROPY SCHEDULE] timesteps={total_timesteps}, ent_coef={new_ent_coef:.6f}, progress={progress_remaining:.2%}")
            self._last_logged = total_timesteps
        
        return True