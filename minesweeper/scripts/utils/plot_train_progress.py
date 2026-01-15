import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Opcjonalny import scipy (fallback do rolling average)
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy nie zainstalowane - używam prostszego wygładzania")

def plot_train_progress(csv_path, output_path):
    """
    Plotowanie dla Minesweeper:
    - Win rate, Loss rate, Invalid rate
    - Mean reward i episode length
    - Policy, Value, Entropy losses
    """
    # Pre-allocate lists
    data = {
        'timesteps': [],
        'mean_reward': [],
        'mean_ep_length': [],
        'win_rate': [],
        'loss_rate': [],
        'invalid_rate': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy_loss': []
    }
    
    # Wczytaj CSV (z error handling)
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get('timesteps'):
                    continue
                    
                try:
                    # Parse timesteps (zawsze wymagane)
                    timesteps = int(row['timesteps'])
                    
                    # Parse metryki
                    mean_reward = float(row.get('mean_reward', 0))
                    mean_ep_length = float(row.get('mean_ep_length', 0))
                    win_rate = float(row.get('win_rate', 0))
                    loss_rate = float(row.get('loss_rate', 0))
                    invalid_rate = float(row.get('invalid_rate', 0))
                    
                    # Losses (mogą być None/puste)
                    def parse_loss(val):
                        """Helper do parsowania loss (może być None/pusty string)"""
                        if val and val != 'None':
                            try:
                                return float(val)
                            except ValueError:
                                return np.nan
                        return np.nan
                    
                    policy_loss_val = parse_loss(row.get('policy_loss', ''))
                    value_loss_val = parse_loss(row.get('value_loss', ''))
                    entropy_loss_val = parse_loss(row.get('entropy_loss', ''))
                    
                    # Append do dict
                    data['timesteps'].append(timesteps)
                    data['mean_reward'].append(mean_reward)
                    data['mean_ep_length'].append(mean_ep_length)
                    data['win_rate'].append(win_rate)
                    data['loss_rate'].append(loss_rate)
                    data['invalid_rate'].append(invalid_rate)
                    data['policy_loss'].append(policy_loss_val)
                    data['value_loss'].append(value_loss_val)
                    data['entropy_loss'].append(entropy_loss_val)
                    
                except (ValueError, TypeError):
                    continue
                    
    except FileNotFoundError:
        print(f'Nie znaleziono pliku: {csv_path}')
        return
    except Exception as e:
        print(f'Błąd wczytywania CSV: {e}')
        return
    
    if not data['timesteps']:
        print('Brak danych do wykresu!')
        return
    
    # Konwertuj do numpy arrays (szybsze operacje)
    timesteps = np.array(data['timesteps'])
    mean_reward_arr = np.array(data['mean_reward'])
    mean_ep_length_arr = np.array(data['mean_ep_length'])
    win_rate_arr = np.array(data['win_rate'])
    loss_rate_arr = np.array(data['loss_rate'])
    invalid_rate_arr = np.array(data['invalid_rate'])
    
    # Setup figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle('Training Progress - Minesweeper PPO', fontsize=18, fontweight='bold')
    
    # Window dla rolling average
    window = 15
    
    # === WYKRES 1: Mean Reward ===
    axes[0, 0].plot(timesteps, mean_reward_arr, color='blue', linewidth=1, alpha=0.5, label='Raw')
    if len(mean_reward_arr) >= window:
        mean_reward_smooth = np.convolve(mean_reward_arr, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(timesteps[window-1:], mean_reward_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[0, 0].set_title('Mean Reward', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # === WYKRES 2: Mean Episode Length ===
    axes[0, 1].plot(timesteps, mean_ep_length_arr, color='green', linewidth=1, alpha=0.5, label='Raw')
    if len(mean_ep_length_arr) >= window:
        mean_ep_length_smooth = np.convolve(mean_ep_length_arr, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(timesteps[window-1:], mean_ep_length_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[0, 1].set_title('Mean Episode Length', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # === WYKRES 3: Win Rate ===
    axes[0, 2].plot(timesteps, win_rate_arr, color='#2ecc71', linewidth=1, alpha=0.5, label='Raw')
    if len(win_rate_arr) >= window:
        win_rate_smooth = np.convolve(win_rate_arr, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(timesteps[window-1:], win_rate_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[0, 2].set_title('Win Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Timesteps')
    axes[0, 2].set_ylabel('Win Rate (%)')
    axes[0, 2].set_ylim(0, 100)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # === WYKRES 4: Loss Rate ===
    axes[1, 0].plot(timesteps, loss_rate_arr, color='#e74c3c', linewidth=1, alpha=0.5, label='Raw')
    if len(loss_rate_arr) >= window:
        loss_rate_smooth = np.convolve(loss_rate_arr, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(timesteps[window-1:], loss_rate_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[1, 0].set_title('Loss Rate (%) - Hit Mine', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Loss Rate (%)')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # === WYKRES 5: Invalid Action Rate ===
    axes[1, 1].plot(timesteps, invalid_rate_arr, color='#f39c12', linewidth=1, alpha=0.5, label='Raw')
    if len(invalid_rate_arr) >= window:
        invalid_rate_smooth = np.convolve(invalid_rate_arr, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(timesteps[window-1:], invalid_rate_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[1, 1].set_title('Invalid Action Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Invalid Rate (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # === WYKRES 6: Policy Loss ===
    policy_loss_arr = np.array(data['policy_loss'])
    axes[1, 2].plot(timesteps, policy_loss_arr, color='#e74c3c', linewidth=1, alpha=0.5, label='Raw')
    if len(policy_loss_arr) >= window:
        policy_loss_clean = np.nan_to_num(policy_loss_arr, nan=0.0)
        policy_loss_smooth = np.convolve(policy_loss_clean, np.ones(window)/window, mode='valid')
        axes[1, 2].plot(timesteps[window-1:], policy_loss_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[1, 2].set_title('Policy Loss', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Timesteps')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # === WYKRES 7: Value Loss (log scale) ===
    value_loss_arr = np.array(data['value_loss'])
    axes[2, 0].plot(timesteps, value_loss_arr, color='#3498db', linewidth=1, alpha=0.5, label='Raw')
    if len(value_loss_arr) >= window:
        value_loss_clean = np.nan_to_num(value_loss_arr, nan=0.0)
        value_loss_smooth = np.convolve(value_loss_clean, np.ones(window)/window, mode='valid')
        axes[2, 0].plot(timesteps[window-1:], value_loss_smooth, color='black', linewidth=2, 
                       label=f'Rolling Mean (w={window})')
    axes[2, 0].set_title('Value Loss', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Timesteps')
    axes[2, 0].set_ylabel('Loss (log scale)')
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3, which='both')
    axes[2, 0].legend()
    
    # === WYKRES 8: Entropy Loss ===
    axes[2, 1].plot(timesteps, data['entropy_loss'], color='#9b59b6', linewidth=2)
    axes[2, 1].set_title('Entropy Loss', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Timesteps')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].grid(True, alpha=0.3)
    
    # === WYKRES 9: Mean Reward vs Episode Length (trajectory) ===
    sc = axes[2, 2].scatter(mean_ep_length_arr, mean_reward_arr, 
                            c=timesteps, cmap='viridis', alpha=0.6, s=20, label='Raw')
    
    # Smoothing dla trajektorii (Savitzky-Golay lub rolling average)
    if len(mean_reward_arr) >= window:
        smooth_window = max(window * 3, 21)
        
        try:
            if HAS_SCIPY and len(mean_reward_arr) >= smooth_window:
                # Savitzky-Golay (lepsze wygładzanie)
                if smooth_window % 2 == 0:
                    smooth_window += 1
                sg_poly = 3 if smooth_window > 3 else 2
                mean_ep_length_smooth = savgol_filter(mean_ep_length_arr, smooth_window, sg_poly)
                mean_reward_smooth = savgol_filter(mean_reward_arr, smooth_window, sg_poly)
            else:
                # Rolling average (fallback)
                smooth_window = min(smooth_window, len(mean_reward_arr))
                mean_ep_length_smooth = np.convolve(mean_ep_length_arr, 
                                                    np.ones(smooth_window)/smooth_window, 
                                                    mode='valid')
                mean_reward_smooth = np.convolve(mean_reward_arr, 
                                               np.ones(smooth_window)/smooth_window, 
                                               mode='valid')
                # Adjust timesteps dla valid mode
                timesteps_smooth = timesteps[smooth_window-1:]
            
            # LineCollection dla kolorowej trajektorii
            from matplotlib.collections import LineCollection
            points = np.array([mean_ep_length_smooth, mean_reward_smooth]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            norm = plt.Normalize(timesteps.min(), timesteps.max())
            
            # Outline (czarny)
            lc_outline = LineCollection(segments, colors='black', linewidth=4, alpha=0.5, zorder=2)
            axes[2, 2].add_collection(lc_outline)
            
            # Główna linia (kolorowa)
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5, zorder=3)
            lc.set_array(timesteps if HAS_SCIPY else timesteps_smooth)
            axes[2, 2].add_collection(lc)
        except Exception as e:
            print(f"⚠️ Nie udało się wygładzić trajektorii: {e}")
    
    axes[2, 2].set_title('Mean Reward vs Episode Length', fontsize=12, fontweight='bold')
    axes[2, 2].set_xlabel('Mean Episode Length')
    axes[2, 2].set_ylabel('Mean Reward')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Colorbar
    if len(axes[2, 2].collections) > 0:
        cbar = plt.colorbar(sc, ax=axes[2, 2])
        cbar.set_label('Timesteps', rotation=270, labelpad=15)
    axes[2, 2].legend()
    
    # Zapisz wykres
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'✅ Wykres zapisany do {output_path}')
    except Exception as e:
        print(f'❌ Błąd zapisu wykresu: {e}')
    finally:
        plt.close()


if __name__ == "__main__":
    import traceback
    
    # Bezpieczniejsze ścieżki - absolutne zamiast relatywnych
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.normpath(os.path.join(script_dir, '..', '..', 'logs'))
    csv_path = os.path.join(logs_dir, 'train_progress.csv')
    output_path = os.path.join(logs_dir, 'training_progress.png')
    
    try:
        plot_train_progress(csv_path, output_path)
    except Exception as e:
        # Zapisz szczegółowy błąd do pliku
        error_log = os.path.join(logs_dir, 'plot_error.log')
        try:
            os.makedirs(logs_dir, exist_ok=True)  # Upewnij się że katalog istnieje
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"❌ BŁĄD GENEROWANIA WYKRESU\n")
                f.write(f"="*70 + "\n\n")
                f.write(f"Script dir: {script_dir}\n")
                f.write(f"Logs dir: {logs_dir}\n")
                f.write(f"CSV path: {csv_path}\n")
                f.write(f"CSV exists: {os.path.exists(csv_path)}\n\n")
                f.write(f"Błąd: {e}\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
            print(f"❌ BŁĄD PLOTU: {e}")
            print(f"📝 Szczegóły zapisane w: {error_log}")
        except Exception as log_error:
            print(f"❌ KRYTYCZNY BŁĄD: {e}")
            print(f"❌ Nie udało się zapisać logu: {log_error}")
            traceback.print_exc()
        
        import sys
        sys.exit(1)