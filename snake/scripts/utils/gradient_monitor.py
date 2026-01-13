"""
✅ FIXED Gradient & Weight Monitor
- Przy wznowieniu treningu DOŁĄCZA dane zamiast nadpisywać CSV
- Silent logging + plotting
- Kontynuacja wykresu przy wznowieniu treningu

GŁÓWNE ZMIANY:
1. self._csv_exists - sprawdza czy CSV istnieje (wznowienie)
2. _on_training_start() - header TYLKO jeśli plik nowy
3. _log_to_csv() - append mode (dołączanie)
4. ✅ initial_timesteps - timesteps kontynuowane przy wznowieniu!
"""

import os
import csv
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GradientWeightMonitor(BaseCallback):
    """
    📊 CICHY MONITOR GRADIENTÓW I WAG (FIXED - kontynuacja przy wznowieniu)
    
    Zbiera dane co log_freq kroków i zapisuje do CSV:
    - Normy gradientów (gradient flow)
    - Normy wag (weight decay effect)
    - Learning rate
    - L2 penalty (weight decay * ||weights||²)
    
    ✅ FIX: Przy wznowieniu treningu dołącza dane do istniejącego CSV
    ✅ FIX: Timesteps kontynuowane przy wznowieniu (initial_timesteps)
    """
    def __init__(self, csv_path, log_freq=1000, initial_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.log_freq = log_freq
        self.initial_timesteps = initial_timesteps
        self.last_logged = 0
        
        # ✅ FIX: Sprawdź czy CSV już istnieje (wznowienie treningu)
        self._csv_exists = os.path.exists(csv_path)
    
    def _on_training_start(self):
        """Inicjalizuj CSV przy starcie treningu"""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # ✅ FIX: Zapisz header TYLKO jeśli plik nie istnieje
        if not self._csv_exists:
            self._write_csv_header()
            if self.verbose > 0:
                print(f"✅ Gradient monitor CSV (NOWY): {self.csv_path}")
        else:
            if self.verbose > 0:
                print(f"✅ Gradient monitor CSV (WZNOWIENIE): {self.csv_path}")
    
    def _on_step(self) -> bool:
        """Monitoruj co log_freq kroków"""
        # ✅ FIX: Użyj total_timesteps (z initial_timesteps)
        total_timesteps = self.num_timesteps + self.initial_timesteps
        
        if total_timesteps - self.last_logged >= self.log_freq:
            self._log_to_csv()
            self.last_logged = total_timesteps
        return True
    
    def _write_csv_header(self):
        """Zapisz nagłówek CSV (tylko raz przy tworzeniu pliku)"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timesteps',
                'grad_norm_total', 'grad_norm_cnn', 'grad_norm_mlp',
                'weight_norm_total', 'weight_norm_cnn', 'weight_norm_mlp',
                'learning_rate', 'l2_penalty'
            ])
    
    def _log_to_csv(self):
        """Zbierz dane i zapisz wiersz do CSV (append mode), ignorując puste pomiary (same zera)"""
        policy = self.model.policy
        grad_norms = self._compute_gradient_norms(policy)
        weight_norms = self._compute_weight_norms(policy)
        lr = self._get_current_lr()
        l2_penalty = self._compute_l2_penalty(policy)
        total_timesteps = self.num_timesteps + self.initial_timesteps

        # IGNORUJ jeśli wszystkie grad_norm i weight_norm są równe 0.0 (pusty pomiar)
        if all(
            abs(grad_norms[k]) < 1e-8 for k in ['total', 'cnn', 'mlp']
        ) and all(
            abs(weight_norms[k]) < 1e-8 for k in ['total', 'cnn', 'mlp']
        ):
            if self.verbose > 0:
                print(f"⏩ Pominięto pusty pomiar gradientów/wag (timesteps={total_timesteps})")
            return

        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    total_timesteps,
                    grad_norms['total'], grad_norms['cnn'], grad_norms['mlp'],
                    weight_norms['total'], weight_norms['cnn'], weight_norms['mlp'],
                    lr if lr is not None else 0.0,
                    l2_penalty
                ])
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Błąd zapisu gradient_monitor.csv: {e}")
    
    def _compute_gradient_norms(self, policy):
        """Oblicz L2 normy gradientów"""
        grad_norms = {'total': 0.0, 'cnn': 0.0, 'mlp': 0.0}
        total_sq = cnn_sq = mlp_sq = 0.0
        for name, param in policy.named_parameters():
            if param.grad is not None:
                param_norm_sq = param.grad.data.norm(2).item() ** 2
                total_sq += param_norm_sq
                if 'features_extractor' in name:
                    if 'conv' in name or 'bn' in name:
                        cnn_sq += param_norm_sq
                    else:
                        mlp_sq += param_norm_sq
                else:
                    mlp_sq += param_norm_sq
        grad_norms['total'] = np.sqrt(total_sq)
        grad_norms['cnn'] = np.sqrt(cnn_sq)
        grad_norms['mlp'] = np.sqrt(mlp_sq)
        return grad_norms
    
    def _compute_weight_norms(self, policy):
        """Oblicz L2 normy wag"""
        weight_norms = {'total': 0.0, 'cnn': 0.0, 'mlp': 0.0}
        total_sq = cnn_sq = mlp_sq = 0.0
        for name, param in policy.named_parameters():
            if param.requires_grad:
                param_norm_sq = param.data.norm(2).item() ** 2
                total_sq += param_norm_sq
                if 'features_extractor' in name:
                    if 'conv' in name or 'bn' in name:
                        cnn_sq += param_norm_sq
                    else:
                        mlp_sq += param_norm_sq
                else:
                    mlp_sq += param_norm_sq
        weight_norms['total'] = np.sqrt(total_sq)
        weight_norms['cnn'] = np.sqrt(cnn_sq)
        weight_norms['mlp'] = np.sqrt(mlp_sq)
        return weight_norms
    
    def _get_current_lr(self):
        """Pobierz aktualny learning rate"""
        if hasattr(self.model.policy, 'optimizer'):
            return self.model.policy.optimizer.param_groups[0]['lr']
        return None
    
    def _compute_l2_penalty(self, policy):
        """Oblicz karę L2 (weight_decay * ||weights||²)"""
        if not hasattr(self.model.policy, 'optimizer'):
            return 0.0
        
        weight_decay = self.model.policy.optimizer.param_groups[0].get('weight_decay', 0.0)
        
        if weight_decay == 0.0:
            return 0.0
        
        l2_sum = sum(param.data.norm(2).item() ** 2 
                     for param in policy.parameters() 
                     if param.requires_grad)
        
        return weight_decay * l2_sum


def plot_gradient_monitor(csv_path, output_path='logs/gradient_monitor.png'):
    """
    📊 Generuje wygładzony wykres z monitoringu gradientów
    Styl analogiczny do training_progress.png
    
    Args:
        csv_path: Ścieżka do gradient_monitor.csv
        output_path: Gdzie zapisać wykres
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Wczytaj dane
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f'⚠️ Nie znaleziono: {csv_path}')
        return
    except Exception as e:
        print(f'⚠️ Błąd wczytywania CSV: {e}')
        return
    
    if len(df) == 0:
        print('⚠️ Brak danych w CSV')
        return
    
    # Parametry wygładzania (jak w training_progress)
    window = 15
    
    # Setup figure - 3 rzędy, 2 kolumny
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Gradient & Weight Monitor - AdamW Regularization', 
                 fontsize=18, fontweight='bold')
    timesteps = df['timesteps'].values

    # === WYKRES 1: Total Gradient Norm ===
    ax = axes[0, 0]
    grad_total = df['grad_norm_total'].values
    ax.plot(timesteps, grad_total, color='#3498db', linewidth=1, alpha=0.5, label='Raw')
    if len(grad_total) >= window:
        grad_smooth = np.convolve(grad_total, np.ones(window)/window, mode='valid')
        ax.plot(timesteps[window-1:], grad_smooth, color='black', linewidth=2, 
                label=f'Rolling Mean (w={window})')
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(10.0, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_title('Total Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()

    # === WYKRES 2: Gradient Norms by Component (bez LSTM) ===
    ax = axes[0, 1]
    for component, color in [('cnn', '#e74c3c'), ('mlp', '#2ecc71')]:
        grad_comp = df[f'grad_norm_{component}'].values
        ax.plot(timesteps, grad_comp, color=color, linewidth=1, alpha=0.4)
        if len(grad_comp) >= window:
            grad_comp_smooth = np.convolve(grad_comp, np.ones(window)/window, mode='valid')
            ax.plot(timesteps[window-1:], grad_comp_smooth, color=color, linewidth=2, 
                    label=component.upper())
    ax.set_title('Gradient Flow by Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()

    # === WYKRES 3: Total Weight Norm ===
    ax = axes[1, 0]
    weight_total = df['weight_norm_total'].values
    ax.plot(timesteps, weight_total, color='#f39c12', linewidth=1, alpha=0.5, label='Raw')
    if len(weight_total) >= window:
        weight_smooth = np.convolve(weight_total, np.ones(window)/window, mode='valid')
        ax.plot(timesteps[window-1:], weight_smooth, color='black', linewidth=2, 
                label=f'Rolling Mean (w={window})')
    ax.set_title('Total Weight Norm (AdamW should decrease)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === WYKRES 4: Weight Norms by Component (bez LSTM) ===
    ax = axes[1, 1]
    for component, color in [('cnn', '#e74c3c'), ('mlp', '#2ecc71')]:
        weight_comp = df[f'weight_norm_{component}'].values
        ax.plot(timesteps, weight_comp, color=color, linewidth=1, alpha=0.4)
        if len(weight_comp) >= window:
            weight_comp_smooth = np.convolve(weight_comp, np.ones(window)/window, mode='valid')
            ax.plot(timesteps[window-1:], weight_comp_smooth, color=color, linewidth=2, 
                    label=component.upper())
    ax.set_title('Weight Norms by Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # === WYKRES 5: Learning Rate ===
    ax = axes[2, 0]
    lr = df['learning_rate'].values
    ax.plot(timesteps, lr, color='#3498db', linewidth=2)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # === WYKRES 6: L2 Penalty (AdamW effect) ===
    ax = axes[2, 1]
    l2_penalty = df['l2_penalty'].values
    ax.plot(timesteps, l2_penalty, color='#e74c3c', linewidth=1, alpha=0.5, label='Raw')
    if len(l2_penalty) >= window:
        l2_smooth = np.convolve(l2_penalty, np.ones(window)/window, mode='valid')
        ax.plot(timesteps[window-1:], l2_smooth, color='black', linewidth=2, 
                label=f'Rolling Mean (w={window})')
    ax.set_title('L2 Penalty (weight_decay × ||W||²)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Penalty')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Zapisz wykres
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'✅ Gradient monitor plot: {output_path}')
    except Exception as e:
        print(f'⚠️ Błąd zapisu wykresu: {e}')
    finally:
        plt.close()


if __name__ == "__main__":
    # Standalone plotting
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'gradient_monitor.png'
        plot_gradient_monitor(csv_path, output_path)
    else:
        print("Użycie: python gradient_monitor.py <csv_path> [output_path]")
        print("\nLub jako callback w train.py:")
        print("from utils.gradient_monitor import GradientWeightMonitor")
        print("gradient_monitor = GradientWeightMonitor('logs/gradient_monitor.csv', log_freq=1000)")
        print("model.learn(..., callback=[..., gradient_monitor])")