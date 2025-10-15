import os
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

def plot_train_progress(csv_path, output_path):
    # Zbierz wszystkie dane
    data = {
        'timesteps': [],
        'mean_reward': [],
        'mean_ep_length': [],
        'mean_score': [],
        'max_score': [],
        'mean_snake_length': [],
        'mean_steps_per_apple': [],
        'progress_score': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy_loss': []
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            if not row or not row.get('timesteps'):
                continue
                
            try:
                timesteps = int(row['timesteps'])
                mean_reward = float(row.get('mean_reward', 0))
                mean_ep_length = float(row.get('mean_ep_length', 0))
                mean_score = float(row.get('mean_score', 0))
                max_score = float(row.get('max_score', 0))
                mean_snake_length = float(row.get('mean_snake_length', 3))
                mean_steps_per_apple = float(row.get('mean_steps_per_apple', 0))
                progress_score = float(row.get('progress_score', 0))
                
                policy_loss = row.get('policy_loss', '')
                value_loss = row.get('value_loss', '')
                entropy_loss = row.get('entropy_loss', '')
                policy_loss_val = float(policy_loss) if policy_loss and policy_loss != 'None' else float('nan')
                value_loss_val = float(value_loss) if value_loss and value_loss != 'None' else float('nan')
                entropy_loss_val = float(entropy_loss) if entropy_loss and entropy_loss != 'None' else float('nan')
                
                data['timesteps'].append(timesteps)
                data['mean_reward'].append(mean_reward)
                data['mean_ep_length'].append(mean_ep_length)
                data['mean_score'].append(mean_score)
                data['max_score'].append(max_score)
                data['mean_snake_length'].append(mean_snake_length)
                data['mean_steps_per_apple'].append(mean_steps_per_apple)
                data['progress_score'].append(progress_score)
                data['policy_loss'].append(policy_loss_val)
                data['value_loss'].append(value_loss_val)
                data['entropy_loss'].append(entropy_loss_val)
                
            except Exception as e:
                continue
    
    if not data['timesteps']:
        print('Brak danych do wykresu!')
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle('Training Progress - Snake RecurrentPPO', fontsize=18, fontweight='bold')
    
    timesteps = data['timesteps']
    
    # Wykres 1: Mean Reward
    axes[0, 0].plot(timesteps, data['mean_reward'], color='blue', linewidth=2)
    axes[0, 0].set_title('Mean Reward', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Wykres 2: Mean Episode Length
    axes[0, 1].plot(timesteps, data['mean_ep_length'], color='green', linewidth=2)
    axes[0, 1].set_title('Mean Episode Length', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Wykres 3: Mean Score
    window = 15
    axes[0, 2].plot(timesteps, data['mean_score'], color='red', linewidth=1, alpha=0.5, label='Mean Score (raw)')
    if len(data['mean_score']) >= window:
        mean_score_smooth = np.convolve(data['mean_score'], np.ones(window)/window, mode='valid')
        axes[0, 2].plot(timesteps[window-1:], mean_score_smooth, color='black', linewidth=2, label=f'Rolling Mean (window={window})')
    axes[0, 2].set_title('Mean Score (Apples Eaten)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Timesteps')
    axes[0, 2].set_ylabel('Apples')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Wykres 4: Max Score
    axes[1, 0].plot(timesteps, data['max_score'], color='orange', linewidth=1, alpha=0.5, label='Max Score (raw)')
    if len(data['max_score']) >= window:
        max_score_smooth = np.convolve(data['max_score'], np.ones(window)/window, mode='valid')
        axes[1, 0].plot(timesteps[window-1:], max_score_smooth, color='black', linewidth=2, label=f'Rolling Mean (window={window})')
    axes[1, 0].set_title('Max Score (Best Episode)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Apples')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Wykres 5: Progress Score
    axes[1, 1].plot(timesteps, data['progress_score'], color='#2ecc71', linewidth=2)
    axes[1, 1].set_title('Progress Score (Composite Metric)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Progress Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Wykres 6: Policy Loss
    axes[1, 2].plot(timesteps, data['policy_loss'], color='#e74c3c', linewidth=1, alpha=0.5, label='Policy Loss (raw)')
    if len(data['policy_loss']) >= window:
        policy_loss_smooth = np.convolve(np.nan_to_num(data['policy_loss'], nan=0.0), np.ones(window)/window, mode='valid')
        axes[1, 2].plot(timesteps[window-1:], policy_loss_smooth, color='black', linewidth=2, label=f'Rolling Mean (window={window})')
    axes[1, 2].set_title('Policy Loss', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Timesteps')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # Wykres 7: Value Loss
    axes[2, 0].plot(timesteps, data['value_loss'], color='#3498db', linewidth=1, alpha=0.5, label='Value Loss (raw)')
    if len(data['value_loss']) >= window:
        value_loss_smooth = np.convolve(np.nan_to_num(data['value_loss'], nan=0.0), np.ones(window)/window, mode='valid')
        axes[2, 0].plot(timesteps[window-1:], value_loss_smooth, color='black', linewidth=2, label=f'Rolling Mean (window={window})')
    axes[2, 0].set_title('Value Loss', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Timesteps')
    axes[2, 0].set_ylabel('Loss (log scale)')
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3, which='both')
    axes[2, 0].legend()
    
    # Wykres 8: Entropy Loss
    axes[2, 1].plot(timesteps, data['entropy_loss'], color='#9b59b6', linewidth=2)
    axes[2, 1].set_title('Entropy Loss', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Timesteps')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Wykres 9: Score vs Episode Length
    sc = axes[2, 2].scatter(data['mean_ep_length'], data['mean_score'], 
                            c=timesteps, cmap='viridis', alpha=0.6, s=20, label='Raw')
    if len(data['mean_score']) >= window:
        sg_window = max(window * 3, 21)
        if sg_window % 2 == 0:
            sg_window += 1
        sg_poly = 3 if sg_window > 3 else 2
        mean_ep_length_arr = np.array(data['mean_ep_length'])
        mean_score_arr = np.array(data['mean_score'])
        timesteps_arr = np.array(timesteps)
        mean_ep_length_smooth = savgol_filter(mean_ep_length_arr, sg_window, sg_poly)
        mean_score_smooth = savgol_filter(mean_score_arr, sg_window, sg_poly)
        from matplotlib.collections import LineCollection
        points = np.array([mean_ep_length_smooth, mean_score_smooth]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(timesteps_arr.min(), timesteps_arr.max())
        lc_outline = LineCollection(segments, colors='black', linewidth=4, alpha=0.5, zorder=2)
        axes[2, 2].add_collection(lc_outline)
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5, zorder=3)
        lc.set_array(timesteps_arr)
        axes[2, 2].add_collection(lc)
    axes[2, 2].set_title('Score vs Episode Length', fontsize=12, fontweight='bold')
    axes[2, 2].set_xlabel('Mean Episode Length')
    axes[2, 2].set_ylabel('Mean Score')
    axes[2, 2].grid(True, alpha=0.3)
    if len(axes[2, 2].collections) > 0:
        cbar = plt.colorbar(sc, ax=axes[2, 2])
        cbar.set_label('Timesteps', rotation=270, labelpad=15)
    axes[2, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wykres zapisany do {output_path}')

if __name__ == "__main__":
    logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    csv_path = os.path.join(logs_dir, 'train_progress.csv')
    output_path = os.path.join(logs_dir, 'training_progress.png')
    plot_train_progress(csv_path, output_path)