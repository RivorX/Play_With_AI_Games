import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from cnn import CustomCNN
from model import make_env
from stable_baselines3 import PPO

# Wczytaj konfigurację jak w innych plikach
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)

# Ścieżka do najlepszego modelu (best_model.zip w models)
model_path = os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip')

# Załaduj model PPO
model = PPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

# Przygotuj środowisko i pobierz przykładową obserwację
env = make_env(render_mode=None, grid_size=16)()
obs, _ = env.reset()
obs_tensor = torch.tensor(obs[None], dtype=torch.float32, requires_grad=True, device=policy.device)

# Przepuść przez sieć i wybierz akcję o największym logicie
features = features_extractor(obs_tensor)
logits = policy.mlp_extractor.forward_actor(features)
score, action = torch.max(logits, dim=1)

# Oblicz gradient względem wejścia
score.backward()
saliency = obs_tensor.grad.abs().detach().cpu().numpy()[0]  # shape: (4, 16, 16, 10)

# Sumuj gradienty po przestrzeni i ramkach, by zobaczyć ważność kanałów
saliency_per_channel = saliency.sum(axis=(0, 1, 2))  # shape: (10,)

# Nazwy kanałów zgodnie z kolejnością w _get_obs
channel_names = [
	'mapa', 'dx', 'dy', 'kierunek', 'grid_size', 'odleglosc',
	'hist_dir_1', 'hist_dir_2', 'hist_dir_3', 'hist_dir_4'
]

# Utwórz katalog na wyniki jeśli nie istnieje
output_dir = os.path.join(base_dir, 'logs', 'analize_saliency')
os.makedirs(output_dir, exist_ok=True)

# Zapisz wykres
plt.figure(figsize=(8,5))
plt.bar(channel_names, saliency_per_channel)
plt.xlabel('Kanał wejściowy')
plt.ylabel('Suma |gradientów|')
plt.title('Saliency map - ważność kanałów wejściowych')
plt.tight_layout()
plot_path = os.path.join(output_dir, 'saliency_map.png')
plt.savefig(plot_path)
print(f'Wykres zapisany do: {plot_path}')

# Zapisz CSV
import csv
csv_path = os.path.join(output_dir, 'saliency_map.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
	writer = csv.writer(f)
	writer.writerow(['kanał', 'suma_gradientów'])
	for name, val in zip(channel_names, saliency_per_channel):
		writer.writerow([name, val])
print(f'CSV zapisany do: {csv_path}')