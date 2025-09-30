import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import csv
from model import make_env
from stable_baselines3 import PPO

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Ścieżka do najlepszego modelu
model_path = os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip')

# Utwórz katalog na wyniki
output_dir = os.path.join(base_dir, 'logs', 'analize_model')
os.makedirs(output_dir, exist_ok=True)

# Załaduj model PPO
model = PPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

# Przygotuj środowisko
env = make_env(render_mode=None, grid_size=16)()

# Nazwy komponentów obserwacji
component_names = ['frame_1', 'frame_2', 'frame_3', 'frame_4', 'direction', 'grid_size', 'dx_head', 'dy_head']

# Nazwy akcji
action_names = ['lewo', 'prosto', 'prawo']

# Lista do przechowywania wyników
all_saliency_stats = []
action_probs_list = []

# Analiza dla 3 różnych obserwacji
for state_idx in range(3):
    # Pobierz obserwację
    obs, _ = env.reset()
    # Wykonaj kilka kroków, aby uzyskać różne stany
    for _ in range(state_idx * 5):
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.clip(action.item() if torch.is_tensor(action) else action, 0, len(action_names) - 1))
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    # Przygotuj obs_tensor jako dict z tensorami
    obs_tensor = {}
    for k, v in obs.items():
        v_np = v if isinstance(v, np.ndarray) else np.array(v)
        if k == 'image':
            # Ensure image is [batch, channels, height, width]
            v_tensor = torch.tensor(v_np, dtype=torch.float32, requires_grad=True, device=policy.device)
            if v_tensor.ndim == 3:  # [channels, height, width]
                v_tensor = v_tensor.unsqueeze(0)  # Add batch dim: [1, channels, height, width]
        else:
            # Scalars: ensure [batch]
            v_tensor = torch.tensor(v_np, dtype=torch.float32, requires_grad=True, device=policy.device)
            if v_tensor.ndim == 1:  # [value]
                v_tensor = v_tensor.unsqueeze(0)  # [batch]
        obs_tensor[k] = v_tensor
        obs_tensor[k].retain_grad()  # Ensure gradients are retained for all tensors

    # Przepuść przez sieć i oblicz gradienty
    features = features_extractor(obs_tensor)
    logits = policy.mlp_extractor.forward_actor(features)
    score, action = torch.max(logits, dim=1)
    action = int(np.clip(action.item(), 0, len(action_names) - 1))
    score.backward()

    # Prawdopodobieństwa akcji
    action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    action_probs_list.append({
        'state': state_idx,
        'p_lewo': action_probs[0],
        'p_prosto': action_probs[1],
        'p_prawo': action_probs[2],
        'akcja': action
    })

    # Statystyki saliency dla komponentów
    saliency_stats = []
    
    # Dla 'image' (4 kanały: stacked frames)
    if obs_tensor['image'].grad is None:
        print(f"Warning: No gradients for image in state {state_idx}")
        image_grad = np.zeros_like(obs_tensor['image'].detach().cpu().numpy()[0])
    else:
        image_grad = obs_tensor['image'].grad.abs().detach().cpu().numpy()[0]  # [channels, height, width]
    
    for i in range(4):
        name = component_names[i]
        channel_grad = image_grad[i, :, :]  # [height, width]
        mean_grad = channel_grad.mean()
        max_grad = channel_grad.max()
        sum_grad = channel_grad.sum()
        saliency_stats.append({
            'state': state_idx,
            'komponent': name,
            'średnia_gradientów': mean_grad,
            'maks_gradientów': max_grad,
            'suma_gradientów': sum_grad
        })

        # Heatmapa tylko dla frame_4
        if name == 'frame_4':
            plt.figure(figsize=(6, 6))
            plt.imshow(channel_grad, cmap='hot', interpolation='nearest')
            plt.colorbar(label='|Gradient|')
            plt.title(f'Saliency Map - {name}, Stan {state_idx}')
            plt.xlabel('X')
            plt.ylabel('Y')
            heatmap_path = os.path.join(output_dir, f'saliency_frame_4_state_{state_idx}.png')
            plt.savefig(heatmap_path)
            plt.close()
            print(f'Heatmapa dla {name} zapisana do: {heatmap_path}')

    # Dla skalarów
    scalar_names = ['direction', 'grid_size', 'dx_head', 'dy_head']
    for idx, scalar_key in enumerate(scalar_names):
        name = component_names[4 + idx]
        if obs_tensor[scalar_key].grad is None:
            print(f"Warning: No gradients for {scalar_key} in state {state_idx}")
            scalar_grad = 0.0
        else:
            scalar_grad = obs_tensor[scalar_key].grad.abs().detach().cpu().numpy()[0]
        saliency_stats.append({
            'state': state_idx,
            'komponent': name,
            'średnia_gradientów': scalar_grad,
            'maks_gradientów': scalar_grad,
            'suma_gradientów': scalar_grad
        })

    all_saliency_stats.extend(saliency_stats)

    # Wizualizacja obserwacji dla frame_4
    latest_frame = obs['image'][-1, :, :]  # [height, width]
    plt.figure(figsize=(6, 6))
    plt.imshow(latest_frame, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wartość')
    plt.title(f'Obserwacja - frame_4, Stan {state_idx}')
    plt.xlabel('X')
    plt.ylabel('Y')
    obs_path = os.path.join(output_dir, f'obs_frame_4_state_{state_idx}.png')
    plt.savefig(obs_path)
    plt.close()
    print(f'Obserwacja dla frame_4 zapisana do: {obs_path}')

# Wykres sumy gradientów po komponentach
saliency_per_component = np.zeros(len(component_names))
for stat in all_saliency_stats:
    idx = component_names.index(stat['komponent'])
    saliency_per_component[idx] += stat['suma_gradientów']
saliency_per_component /= 3  # Uśrednij po 3 stanach

plt.figure(figsize=(10, 6))
plt.bar(component_names, saliency_per_component)
plt.xlabel('Komponent obserwacji')
plt.ylabel('Średnia suma |gradientów|')
plt.title('Ważność komponentów obserwacji (uśredniona dla 3 stanów)')
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = os.path.join(output_dir, 'saliency_components_avg.png')
plt.savefig(plot_path)
plt.close()
print(f'Wykres komponentów zapisany do: {plot_path}')

# Wykres prawdopodobieństw akcji
for state_idx in range(3):
    probs = action_probs_list[state_idx]
    selected_action = action_names[probs['akcja']]
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
    writer.writerow(['stan', 'komponent', 'średnia_gradientów', 'maks_gradientów', 'suma_gradientów'])
    for stat in all_saliency_stats:
        writer.writerow([
            stat['state'],
            stat['komponent'],
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