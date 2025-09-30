import os
import csv
import matplotlib.pyplot as plt

def plot_train_progress(csv_path, output_path):
    timesteps, rewards, lengths = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row['mean_reward'] and row['mean_ep_length']:
                    t = int(row['timesteps'])
                    r = float(row['mean_reward'])
                    l = float(row['mean_ep_length'])
                    timesteps.append(t)
                    rewards.append(r)
                    lengths.append(l)
            except Exception:
                continue
    if timesteps:
        plt.figure(figsize=(8, 4))
        plt.plot(timesteps, rewards, label='mean_reward')
        plt.plot(timesteps, lengths, label='mean_ep_length')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.title('Training Progress')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f'Wykres zapisany do {output_path}')
    else:
        print('Brak danych do wykresu!')

if __name__ == "__main__":
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    csv_path = os.path.join(logs_dir, 'train_progress.csv')
    output_path = os.path.join(logs_dir, 'training_progress.png')
    plot_train_progress(csv_path, output_path)