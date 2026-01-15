import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_train_progress(csv_path, output_path):
    """
    Plots training progress for Solitaire.
    """
    data = {
        'timesteps': [],
        'mean_reward': [],
        'mean_ep_length': [],
        'mean_score': [],
        'max_score': [],
        'win_rate': [],
        'mean_foundations': [],
        'mean_moves': []
    }
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get('timesteps'):
                    continue
                try:
                    # ✅ Walidacja - dodaj wiersz tylko jeśli kluczowe wartości są OK
                    timesteps_val = int(row['timesteps'])
                    
                    # Sprawdź czy mean_reward i mean_ep_length są dostępne (nie puste)
                    mean_reward_str = row.get('mean_reward', '').strip()
                    mean_ep_length_str = row.get('mean_ep_length', '').strip()
                    
                    # Pomiń wiersz jeśli brakuje kluczowych danych
                    if not mean_reward_str or not mean_ep_length_str:
                        continue
                    
                    mean_reward_val = float(mean_reward_str)
                    mean_ep_length_val = float(mean_ep_length_str)
                    mean_score_val = float(row.get('mean_score') or 0)
                    max_score_val = float(row.get('max_score') or 0)
                    win_rate_val = float(row.get('win_rate') or 0)
                    mean_foundations_val = float(row.get('mean_foundations') or 0)
                    mean_moves_val = float(row.get('mean_moves') or 0)
                    
                    # Dodaj wszystkie na raz (atomowo)
                    data['timesteps'].append(timesteps_val)
                    data['mean_reward'].append(mean_reward_val)
                    data['mean_ep_length'].append(mean_ep_length_val)
                    data['mean_score'].append(mean_score_val)
                    data['max_score'].append(max_score_val)
                    data['win_rate'].append(win_rate_val)
                    data['mean_foundations'].append(mean_foundations_val)
                    data['mean_moves'].append(mean_moves_val)
                except (ValueError, TypeError) as e:
                    # Skip invalid rows
                    continue
    except FileNotFoundError:
        print(f'File not found: {csv_path}')
        return
    except Exception as e:
        print(f'Error reading CSV: {e}')
        return
    
    if not data['timesteps']:
        print('No data to plot!')
        return
    
    timesteps = np.array(data['timesteps'])
    
    # Setup figure
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle('Training Progress - Solitaire MaskablePPO', fontsize=16, fontweight='bold')
    
    # Window for rolling average
    window = min(50, len(timesteps) // 10 + 1)
    
    def rolling_avg(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')
    
    
    # === Plot 1: Mean Reward ===
    ax = axes[0, 0]
    ax.plot(timesteps, data['mean_reward'], color='blue', alpha=0.3, label='Raw')
    if len(data['mean_reward']) >= window:
        smooth = rolling_avg(data['mean_reward'], window)
        ax.plot(timesteps[window-1:], smooth, color='darkblue', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Mean Reward', fontsize=12)
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 2: Win Rate ===
    ax = axes[0, 1]
    ax.plot(timesteps, data['win_rate'], color='green', alpha=0.3, label='Raw')
    if len(data['win_rate']) >= window:
        smooth = rolling_avg(data['win_rate'], window)
        ax.plot(timesteps[window-1:], smooth, color='darkgreen', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Win Rate', fontsize=12)
    ax.set_ylabel('Win Rate (0-1)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 3: Mean Score ===
    ax = axes[1, 0]
    ax.plot(timesteps, data['mean_score'], color='orange', alpha=0.3, label='Raw')
    if len(data['mean_score']) >= window:
        smooth = rolling_avg(data['mean_score'], window)
        ax.plot(timesteps[window-1:], smooth, color='darkorange', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Mean Score', fontsize=12)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 4: Max Score ===
    ax = axes[1, 1]
    ax.plot(timesteps, data['max_score'], color='red', alpha=0.3, label='Raw')
    if len(data['max_score']) >= window:
        smooth = rolling_avg(data['max_score'], window)
        ax.plot(timesteps[window-1:], smooth, color='darkred', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Max Score', fontsize=12)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 5: Mean Foundations Filled ===
    ax = axes[2, 0]
    ax.plot(timesteps, data['mean_foundations'], color='purple', alpha=0.3, label='Raw')
    if len(data['mean_foundations']) >= window:
        smooth = rolling_avg(data['mean_foundations'], window)
        ax.plot(timesteps[window-1:], smooth, color='indigo', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Mean Foundations Filled', fontsize=12)
    ax.set_ylabel('Cards (0-52)')
    ax.set_ylim(0, 52)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 6: Mean Episode Length ===
    ax = axes[2, 1]
    ax.plot(timesteps, data['mean_ep_length'], color='cyan', alpha=0.3, label='Raw')
    if len(data['mean_ep_length']) >= window:
        smooth = rolling_avg(data['mean_ep_length'], window)
        ax.plot(timesteps[window-1:], smooth, color='teal', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Mean Episode Length', fontsize=12)
    ax.set_ylabel('Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 7: Mean Moves ===
    ax = axes[3, 0]
    ax.plot(timesteps, data['mean_moves'], color='magenta', alpha=0.3, label='Raw')
    if len(data['mean_moves']) >= window:
        smooth = rolling_avg(data['mean_moves'], window)
        ax.plot(timesteps[window-1:], smooth, color='purple', linewidth=2, label=f'Smooth (w={window})')
    ax.set_title('Mean Moves per Game', fontsize=12)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Moves')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 8: Combined Performance ===
    ax = axes[3, 1]
    # Normalize metrics to 0-1 for comparison
    if data['mean_score']:
        max_score = max(data['mean_score']) if max(data['mean_score']) > 0 else 1
        norm_score = [s / max_score for s in data['mean_score']]
        ax.plot(timesteps, norm_score, color='orange', alpha=0.5, label='Norm Score')
    
    ax.plot(timesteps, data['win_rate'], color='green', alpha=0.5, label='Win Rate')
    
    if data['mean_foundations']:
        norm_foundations = [f / 52.0 for f in data['mean_foundations']]
        ax.plot(timesteps, norm_foundations, color='purple', alpha=0.5, label='Norm Foundations')
    
    ax.set_title('Combined Performance (Normalized)', fontsize=12)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f'✅ Plot saved to {output_path}')
    except Exception as e:
        print(f'❌ Error saving plot: {e}')
    finally:
        plt.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.normpath(os.path.join(script_dir, '..', '..', 'logs'))
    csv_path = os.path.join(logs_dir, 'train_progress.csv')
    output_path = os.path.join(logs_dir, 'training_progress.png')
    
    plot_train_progress(csv_path, output_path)
