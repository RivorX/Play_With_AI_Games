import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import csv
from cnn import CustomCNN
from model import make_env
from stable_baselines3 import PPO

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Ścieżka do najlepszego modelu
model_path = os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip')

# Utwórz katalog na wyniki bezpośrednio w logs/analize_model
output_dir = os.path.join(base_dir, 'logs', 'analize_model')
os.makedirs(output_dir, exist_ok=True)

# Załaduj model PPO
model = PPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

# Przygotuj środowisko
env = make_env(render_mode=None, grid_size=16)()

# Nazwy kanałów
channel_names = ['mapa', 'dx', 'dy', 'kierunek', 'grid_size', 'odleglosc', 'hist_dir_1', 'hist_dir_2', 'hist_dir_3', 'hist_dir_4']

# Nazwy akcji
action_names = ['lewo', 'prosto', 'prawo']

# Lista do przechowywania wyników dla różnych obserwacji
all_saliency_stats = []
action_probs_list = []

# Analiza dla 3 różnych obserwacji
for state_idx in range(3):
    # Pobierz obserwację
    obs, _ = env.reset()
    # Wykonaj kilka kroków, aby uzyskać różne stany
    for _ in range(state_idx * 5):  # Różne stany: 0, 5, 10 kroków
        action, _ = model.predict(obs, deterministic=True)
        # Upewnij się, że akcja jest w zakresie [0, 2]
        action = int(np.clip(action.cpu() if torch.is_tensor(action) else action, 0, len(action_names) - 1))
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    obs_tensor = torch.tensor(obs[None], dtype=torch.float32, requires_grad=True, device=policy.device)

    # Przepuść przez sieć i oblicz gradienty
    features = features_extractor(obs_tensor)
    logits = policy.mlp_extractor.forward_actor(features)
    score, action = torch.max(logits, dim=1)
    action = int(np.clip(action.cpu().numpy() if torch.is_tensor(action) else action, 0, len(action_names) - 1))  # Upewnij się, że akcja jest w zakresie
    score.backward()
    saliency = obs_tensor.grad.abs().detach().cpu().numpy()[0]  # shape: (4, 16, 16, 10)

    # Prawdopodobieństwa akcji
    action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    action_probs_list.append({
        'state': state_idx,
        'p_lewo': action_probs[0],
        'p_prosto': action_probs[1],
        'p_prawo': action_probs[2],
        'akcja': action
    })

    # Statystyki saliency dla każdego kanału
    saliency_stats = []
    for i, name in enumerate(channel_names):
        channel_saliency = saliency[:, :, :, i]
        mean_saliency = channel_saliency.mean()
        max_saliency = channel_saliency.max()
        sum_saliency = channel_saliency.sum()
        saliency_stats.append({
            'state': state_idx,
            'kanał': name,
            'średnia_gradientów': mean_saliency,
            'maks_gradientów': max_saliency,
            'suma_gradientów': sum_saliency
        })

        # Heatmapa dla kanału 'mapa' (tylko najnowsza ramka)
        if name == 'mapa':
            plt.figure(figsize=(6, 6))
            plt.imshow(channel_saliency[-1], cmap='hot', interpolation='nearest')
            plt.colorbar(label='|Gradient|')
            plt.title(f'Saliency Map - Kanał mapa, Stan {state_idx}')
            plt.xlabel('X')
            plt.ylabel('Y')
            heatmap_path = os.path.join(output_dir, f'saliency_mapa_state_{state_idx}.png')
            plt.savefig(heatmap_path)
            plt.close()
            print(f'Heatmapa dla kanału mapa zapisana do: {heatmap_path}')

    all_saliency_stats.extend(saliency_stats)

    # Wizualizacja obserwacji dla kanału 'mapa' (dla odniesienia)
    plt.figure(figsize=(6, 6))
    plt.imshow(obs[-1, :, :, 0], cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wartość')
    plt.title(f'Obserwacja - Kanał mapa, Stan {state_idx}')
    plt.xlabel('X')
    plt.ylabel('Y')
    obs_path = os.path.join(output_dir, f'obs_mapa_state_{state_idx}.png')
    plt.savefig(obs_path)
    plt.close()
    print(f'Obserwacja dla kanału mapa zapisana do: {obs_path}')

# Wykres sumy gradientów po kanałach (uśredniony dla wszystkich stanów)
saliency_per_channel = np.zeros(len(channel_names))
for stat in all_saliency_stats:
    saliency_per_channel[channel_names.index(stat['kanał'])] += stat['suma_gradientów']
saliency_per_channel /= 3  # Uśrednij po 3 stanach

plt.figure(figsize=(10, 6))
plt.bar(channel_names, saliency_per_channel)
plt.xlabel('Kanał wejściowy')
plt.ylabel('Średnia suma |gradientów|')
plt.title('Ważność kanałów wejściowych (uśredniona dla 3 stanów)')
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = os.path.join(output_dir, 'saliency_channels_avg.png')
plt.savefig(plot_path)
plt.close()
print(f'Wykres kanałów zapisany do: {plot_path}')

# Wykres prawdopodobieństw akcji
for state_idx in range(3):
    probs = action_probs_list[state_idx]
    selected_action = action_names[probs['akcja']]  # Poprawione wybieranie akcji
    plt.figure(figsize=(6, 4))
    plt.bar(action_names, [probs['p_lewo'], probs['p_prosto'], probs['p_prawo']])
    plt.xlabel('Akcja')
    plt.ylabel('Prawdopodobieństwo')
    plt.title(f'Prawdopodobieństwa akcji - Stan {state_idx} (Wybrano: {selected_action})')
    action_plot_path = os.path.join(output_dir, f'action_probs_state_{state_idx}.png')
    plt.savefig(action_plot_path)
    plt.close()
    print(f'Wykres prawdopodobieństw akcji zapisany do: {action_plot_path}')

# Zapisz statystyki saliency do CSV
csv_path = os.path.join(output_dir, 'saliency_stats.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['stan', 'kanał', 'średnia_gradientów', 'maks_gradientów', 'suma_gradientów'])
    for stat in all_saliency_stats:
        writer.writerow([
            stat['state'],
            stat['kanał'],
            stat['średnia_gradientów'],
            stat['maks_gradientów'],
            stat['suma_gradientów']
        ])
print(f'CSV statystyk zapisany do: {csv_path}')

# Zapisz prawdopodobieństwa akcji do CSV
action_csv_path = os.path.join(output_dir, 'action_probs.csv')
with open(action_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['stan', 'p_lewo', 'p_prosto', 'p_prawo', 'akcja'])
    for probs in action_probs_list:
        writer.writerow([
            probs['state'],
            probs['p_lewo'],
            probs['p_prosto'],
            probs['p_prawo'],
            probs['akcja']
        ])
print(f'CSV prawdopodobieństw akcji zapisany do: {action_csv_path}')

# Zamknij środowisko
env.close()