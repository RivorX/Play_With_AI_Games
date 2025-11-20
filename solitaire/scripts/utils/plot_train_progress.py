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
        'win_rate': []
    }
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get('timesteps'):
                    continue
                try:
                    data['timesteps'].append(int(row['timesteps']))
                    data['mean_reward'].append(float(row['mean_reward']))
                    data['mean_ep_length'].append(float(row['mean_ep_length']))
                    data['win_rate'].append(float(row['win_rate']))
                except (ValueError, TypeError):
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
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Training Progress - Solitaire MaskablePPO', fontsize=16, fontweight='bold')
    
    # Window for rolling average
    window = 50
    
    # === Plot 1: Mean Reward ===
    ax = axes[0]
    ax.plot(timesteps, data['mean_reward'], color='blue', alpha=0.3, label='Raw')
    if len(data['mean_reward']) >= window:
        smooth = np.convolve(data['mean_reward'], np.ones(window)/window, mode='valid')
        ax.plot(timesteps[window-1:], smooth, color='darkblue', linewidth=2, label=f'Rolling Mean (w={window})')
    ax.set_title('Mean Reward (Last 100 Episodes)', fontsize=12)
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 2: Mean Episode Length ===
    ax = axes[1]
    ax.plot(timesteps, data['mean_ep_length'], color='green', alpha=0.3, label='Raw')
    if len(data['mean_ep_length']) >= window:
        smooth = np.convolve(data['mean_ep_length'], np.ones(window)/window, mode='valid')
        ax.plot(timesteps[window-1:], smooth, color='darkgreen', linewidth=2, label=f'Rolling Mean (w={window})')
    ax.set_title('Mean Episode Length', fontsize=12)
    ax.set_ylabel('Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === Plot 3: Win Rate ===
    ax = axes[2]
    ax.plot(timesteps, data['win_rate'], color='orange', alpha=0.3, label='Raw')
    if len(data['win_rate']) >= window:
        smooth = np.convolve(data['win_rate'], np.ones(window)/window, mode='valid')
        ax.plot(timesteps[window-1:], smooth, color='darkorange', linewidth=2, label=f'Rolling Mean (w={window})')
    ax.set_title('Win Rate (Last 100 Episodes)', fontsize=12)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Win Rate (0-1)')
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
