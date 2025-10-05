import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import csv
from model import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

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

# Załaduj model PPO
print("Ładowanie modelu...")
model = PPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

# Przygotuj środowisko
sequence_length = config['environment']['sequence_length']
env = make_env(render_mode=None, grid_size=16, sequence_length=sequence_length)()

# Nazwy akcji
action_names = ['lewo', 'prosto', 'prawo']

# Lista do przechowywania wyników
all_importance_stats = []
action_probs_list = []

print("\n=== Analiza ważności komponentów ===")

# Analiza dla 3 różnych obserwacji
for state_idx in range(3):
    print(f"\nAnaliza stanu {state_idx}...")
    
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
        v_tensor = torch.tensor(v_np, dtype=torch.float32, requires_grad=True, device=policy.device)
        
        if k == 'image':
            # [seq_len, H, W, C] -> potrzebujemy [batch, seq_len, H, W, C]
            if v_tensor.ndim == 4:
                v_tensor = v_tensor.unsqueeze(0)
        else:
            # Skalary: [value] -> [batch, value]
            if v_tensor.ndim == 1:
                v_tensor = v_tensor.unsqueeze(0)
        
        obs_tensor[k] = v_tensor
        obs_tensor[k].retain_grad()

    # === Przepuść przez sieć i zbierz aktywacje ===
    
    # 1. ConvLSTM
    image = obs_tensor['image']
    if len(image.shape) == 5:
        image = image.permute(0, 1, 4, 2, 3)  # [batch, seq_len, C, H, W]
    
    convlstm = features_extractor.convlstm
    lstm_out, hidden_states = convlstm(image, hidden_state=None)
    last_frame_lstm = lstm_out[:, -1, :, :, :]  # [batch, hidden_channels, H, W]
    
    # 2. CNN po ConvLSTM
    cnn_out = features_extractor.cnn(last_frame_lstm)
    
    # 3. Skalary
    scalars = torch.cat([
        obs_tensor['direction'],
        obs_tensor['dx_head'],
        obs_tensor['dy_head'],
        obs_tensor['front_coll'],
        obs_tensor['left_coll'],
        obs_tensor['right_coll']
    ], dim=-1)
    scalar_features = features_extractor.scalar_linear(scalars)
    
    # 4. Finalne cechy
    features = torch.cat([cnn_out, scalar_features], dim=-1)
    final_features = features_extractor.final_linear(features)
    
    # 5. Decyzja
    logits = policy.mlp_extractor.forward_actor(final_features)
    score, action = torch.max(logits, dim=1)
    action_idx = int(np.clip(action.item(), 0, len(action_names) - 1))
    score.backward()

    # === Prawdopodobieństwa akcji ===
    action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    action_probs_list.append({
        'state': state_idx,
        'p_lewo': action_probs[0],
        'p_prosto': action_probs[1],
        'p_prawo': action_probs[2],
        'akcja': action_idx
    })

    # === Wizualizacja ConvLSTM output ===
    lstm_activation = last_frame_lstm[0].detach().cpu().numpy()  # [hidden_channels, H, W]
    
    # Pokaż pierwsze 16 kanałów ConvLSTM
    num_channels_to_show = min(16, lstm_activation.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_channels_to_show):
        ax = axes[i // 4, i % 4]
        ax.imshow(lstm_activation[i], cmap='viridis')
        ax.set_title(f'LSTM Ch{i}')
        ax.axis('off')
    plt.suptitle(f'ConvLSTM Output - Stan {state_idx}')
    plt.tight_layout()
    lstm_viz_path = os.path.join(conv_viz_dir, f'lstm_output_state_{state_idx}.png')
    plt.savefig(lstm_viz_path, dpi=150)
    plt.close()
    print(f'  Wizualizacja ConvLSTM zapisana: {lstm_viz_path}')

    # === Wizualizacja viewport ===
    viewport = obs['image'][-1, :, :, 0]  # Ostatnia ramka z sekwencji [H, W]
    plt.figure(figsize=(8, 8))
    plt.imshow(viewport, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wartość')
    plt.title(f'Viewport 16x16 - Stan {state_idx}')
    
    # Dodaj legendę
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

    # === Analiza ważności (gradienty) ===
    
    # Ważność ConvLSTM (gradient na wejściowym obrazie)
    if obs_tensor['image'].grad is not None:
        image_grad = obs_tensor['image'].grad.abs()
        # Uśrednij po wszystkich wymiarach poza batch
        convlstm_importance = image_grad.mean().item()
    else:
        convlstm_importance = 0.0

    all_importance_stats.append({
        'state': state_idx,
        'komponent': 'ConvLSTM_avg',
        'ważność': convlstm_importance
    })

    # Ważność poszczególnych skalarów
    scalar_names = ['direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']
    for scalar_key in scalar_names:
        if obs_tensor[scalar_key].grad is not None:
            scalar_grad = obs_tensor[scalar_key].grad.abs().mean().detach().cpu().item()
        else:
            scalar_grad = 0.0
        
        all_importance_stats.append({
            'state': state_idx,
            'komponent': scalar_key,
            'ważność': scalar_grad
        })

# === Agregacja ważności ===
print("\n=== Agregacja ważności komponentów ===")

component_importance = {}
for stat in all_importance_stats:
    comp = stat['komponent']
    if comp not in component_importance:
        component_importance[comp] = []
    component_importance[comp].append(stat['ważność'])

# Uśrednij
avg_importance = {k: np.mean(v) for k, v in component_importance.items()}
sorted_components = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

print("\nŚrednia ważność komponentów (posortowane):")
for comp, imp in sorted_components:
    print(f"  {comp}: {imp:.6f}")

# Wykres ważności
components = [c[0] for c in sorted_components]
importances = [c[1] for c in sorted_components]

plt.figure(figsize=(12, 6))
plt.bar(components, importances)
plt.xlabel('Komponent')
plt.ylabel('Średnia ważność (|gradient|)')
plt.title('Ważność komponentów obserwacji (uśredniona dla 3 stanów)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
importance_plot_path = os.path.join(output_dir, 'component_importance.png')
plt.savefig(importance_plot_path, dpi=150)
plt.close()
print(f'\nWykres ważności zapisany: {importance_plot_path}')

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

# === Zapisz statystyki do CSV ===
csv_path = os.path.join(output_dir, 'importance_stats.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['stan', 'komponent', 'ważność'])
    for stat in all_importance_stats:
        writer.writerow([stat['state'], stat['komponent'], stat['ważność']])
print(f'\nStatystyki zapisane: {csv_path}')

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

# === Wizualizacja wag ConvLSTM (pierwsze filtry) ===
print("\n=== Wizualizacja filtrów ConvLSTM ===")

# Filtry z pierwszej warstwy ConvLSTM
first_lstm_cell = features_extractor.convlstm.cell_list[0]
conv_weights = first_lstm_cell.conv.weight.data.cpu().numpy()  # [out_channels, in_channels, K, K]

# Pokaż pierwsze 16 filtrów
num_filters_to_show = min(16, conv_weights.shape[0])
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i in range(num_filters_to_show):
    ax = axes[i // 4, i % 4]
    # Weź pierwszy kanał wejściowy dla wizualizacji
    filter_vis = conv_weights[i, 0, :, :]
    im = ax.imshow(filter_vis, cmap='coolwarm', interpolation='nearest')
    ax.set_title(f'Filter {i}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('Filtry ConvLSTM - Warstwa 1 (pierwsze 16)')
plt.tight_layout()
filters_path = os.path.join(conv_viz_dir, 'convlstm_filters.png')
plt.savefig(filters_path, dpi=150)
plt.close()
print(f'Filtry ConvLSTM zapisane: {filters_path}')

# Zamknij środowisko
env.close()

print("\n=== Analiza zakończona ===")
print(f"Wyniki zapisane w: {output_dir}")
print(f"Wizualizacje konwolucji w: {conv_viz_dir}")