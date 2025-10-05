import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import csv
from model import make_env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Ścieżka do najlepszego modelu
model_path = os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip')

# Utwórz katalogi na wyniki
output_dir = os.path.join(base_dir, 'logs', 'analyze_model')
conv_viz_dir = os.path.join(output_dir, 'conv_visualizations')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(conv_viz_dir, exist_ok=True)

# Załaduj model RecurrentPPO
print("Ładowanie modelu...")
model = RecurrentPPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

# Przygotuj środowisko
env = make_env(render_mode=None, grid_size=16)()

# Nazwy akcji
action_names = ['lewo', 'prosto', 'prawo']

# Lista do przechowywania wyników
all_importance_stats = []
action_probs_list = []

print("\n=== Analiza ważności komponentów ===")

# Inicjalizuj stany LSTM
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

# Analiza dla 3 różnych obserwacji
for state_idx in range(3):
    print(f"\nAnaliza stanu {state_idx}...")
    
    # Pobierz obserwację
    obs, _ = env.reset()
    lstm_states = None  # Reset LSTM na początku epizodu
    
    # Wykonaj kilka kroków, aby uzyskać różne stany
    for _ in range(state_idx * 5):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        action = int(np.clip(action.item() if torch.is_tensor(action) else action, 0, len(action_names) - 1))
        obs, _, done, _, _ = env.step(action)
        episode_starts = np.array([done], dtype=bool)
        if done:
            obs, _ = env.reset()
            lstm_states = None

    # Przygotuj obs_tensor jako dict z tensorami
    obs_tensor = {}
    for k, v in obs.items():
        v_np = v if isinstance(v, np.ndarray) else np.array(v)
        v_tensor = torch.tensor(v_np, dtype=torch.float32, device=policy.device)
        
        if k == 'image':
            # [H, W, C] -> [batch, H, W, C]
            if v_tensor.ndim == 3:
                v_tensor = v_tensor.unsqueeze(0)
        else:
            # Skalary: [value] -> [batch, value]
            if v_tensor.ndim == 1:
                v_tensor = v_tensor.unsqueeze(0)
        
        obs_tensor[k] = v_tensor

    # === UPROSZCZONE: Użyj model.predict + ręczny forward dla logitów ===
    with torch.no_grad():
        # 1. Pobierz akcję (i zaktualizuj lstm_states)
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        action_idx = int(action.item() if torch.is_tensor(action) else action)
        action_idx = int(np.clip(action_idx, 0, len(action_names) - 1))
        
        # 2. Ręczny forward dla prawdopodobieństw (bez LSTM - już było w predict)
        # Ekstrahuj cechy
        features = features_extractor(obs_tensor)
        
        # Przygotuj lstm_states jako tensory
        lstm_states_tensor = (
            torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device),
            torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device)
        )
        
        # Przepuść przez LSTM aktora (już zaktualizowane stany z predict)
        # features: [batch, features_dim] -> [batch, seq=1, features_dim]
        features_seq = features.unsqueeze(1)
        
        # Forward przez LSTM
        lstm_output, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
        latent_pi = lstm_output.squeeze(1)  # [batch, lstm_hidden_size]
        
        # MLP extractor dla policy
        latent_pi = policy.mlp_extractor.policy_net(latent_pi)
        
        # Logity akcji
        logits = policy.action_net(latent_pi)

    # === Prawdopodobieństwa akcji ===
    action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    action_probs_list.append({
        'state': state_idx,
        'p_lewo': action_probs[0],
        'p_prosto': action_probs[1],
        'p_prawo': action_probs[2],
        'akcja': action_idx
    })

    # === Wizualizacja CNN output ===
    image = obs_tensor['image']
    if len(image.shape) == 4:
        image = image.permute(0, 3, 1, 2)  # [batch, C, H, W]
    
    # Przepuść przez pierwszą warstwę CNN
    with torch.no_grad():
        first_conv = features_extractor.cnn[0]
        cnn_activation = first_conv(image).detach().cpu().numpy()[0]
    
    # Pokaż pierwsze 16 kanałów CNN
    num_channels_to_show = min(16, cnn_activation.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_channels_to_show):
        ax = axes[i // 4, i % 4]
        ax.imshow(cnn_activation[i], cmap='viridis')
        ax.set_title(f'CNN Ch{i}')
        ax.axis('off')
    for i in range(num_channels_to_show, 16):
        axes[i // 4, i % 4].axis('off')
    plt.suptitle(f'CNN Output (warstwa 1) - Stan {state_idx}')
    plt.tight_layout()
    cnn_viz_path = os.path.join(conv_viz_dir, f'cnn_output_state_{state_idx}.png')
    plt.savefig(cnn_viz_path, dpi=150)
    plt.close()
    print(f'  Wizualizacja CNN zapisana: {cnn_viz_path}')

    # === Wizualizacja viewport ===
    viewport = obs['image'][:, :, 0]
    plt.figure(figsize=(8, 8))
    plt.imshow(viewport, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wartość')
    plt.title(f'Viewport 16x16 - Stan {state_idx}')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', label='Ściana (-1.0)'),
        Patch(facecolor='black', label='Puste (0.0)'),
        Patch(facecolor='cyan', label='Ciało (0.5)'),
        Patch(facecolor='orange', label='Jedzenie (0.75)'),
        Patch(facecolor='yellow', label='Głowa (1.0)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    viewport_path = os.path.join(output_dir, f'viewport_state_{state_idx}.png')
    plt.savefig(viewport_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Viewport zapisany: {viewport_path}')

    print(f"  Wybrany action: {action_names[action_idx]}")
    print(f"  Prawdopodobieństwa: {action_probs}")

# === Wykres prawdopodobieństw akcji ===
print("\n=== Prawdopodobieństwa akcji ===")

for state_idx in range(3):
    probs = action_probs_list[state_idx]
    selected_action = action_names[probs['akcja']]
    
    print(f"Stan {state_idx}: Wybrano '{selected_action}' - p_lewo={probs['p_lewo']:.3f}, p_prosto={probs['p_prosto']:.3f}, p_prawo={probs['p_prawo']:.3f}")
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(action_names, [probs['p_lewo'], probs['p_prosto'], probs['p_prawo']])
    bars[probs['akcja']].set_color('green')
    plt.xlabel('Akcja')
    plt.ylabel('Prawdopodobieństwo')
    plt.title(f'Prawdopodobieństwa akcji - Stan {state_idx} (Wybrano: {selected_action})')
    plt.ylim(0, 1)
    action_plot_path = os.path.join(output_dir, f'action_probs_state_{state_idx}.png')
    plt.savefig(action_plot_path, dpi=150)
    plt.close()

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
print(f'Prawdopodobieństwa akcji zapisane: {action_csv_path}')

# === Wizualizacja wag CNN ===
print("\n=== Wizualizacja filtrów CNN ===")

first_conv = features_extractor.cnn[0]
conv_weights = first_conv.weight.data.cpu().numpy()

num_filters_to_show = min(16, conv_weights.shape[0])
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i in range(num_filters_to_show):
    ax = axes[i // 4, i % 4]
    filter_vis = conv_weights[i, 0, :, :]
    im = ax.imshow(filter_vis, cmap='coolwarm', interpolation='nearest')
    ax.set_title(f'Filter {i}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
for i in range(num_filters_to_show, 16):
    axes[i // 4, i % 4].axis('off')
plt.suptitle('Filtry CNN - Warstwa 1 (pierwsze 16)')
plt.tight_layout()
filters_path = os.path.join(conv_viz_dir, 'cnn_filters.png')
plt.savefig(filters_path, dpi=150)
plt.close()
print(f'Filtry CNN zapisane: {filters_path}')

env.close()

print("\n=== Analiza zakończona ===")
print(f"Wyniki zapisane w: {output_dir}")
print(f"Wizualizacje w: {conv_viz_dir}")
print("\nUwaga: Analiza gradientów jest wyłączona dla RecurrentPPO")
print("ze względu na złożoność backprop przez LSTM.")