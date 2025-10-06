import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import csv
from model import make_env
from sb3_contrib import RecurrentPPO

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
gradient_importance = []
activation_importance = []

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
    
    # === ANALIZA WAŻNOŚCI PRZEZ GRADIENTY ===
    print(f"  Obliczanie ważności neuronów...")
    
    # PRZEŁĄCZ MODEL W TRYB TRENINGOWY dla backward pass
    model.policy.train()
    
    # Metoda 1: Bezpośrednie gradienty z retain_grad
    obs_grad = {}
    for k, v in obs.items():
        v_np = v if isinstance(v, np.ndarray) else np.array(v)
        v_tensor = torch.tensor(v_np, dtype=torch.float32, device=policy.device, requires_grad=True)
        
        if k == 'image':
            if v_tensor.ndim == 3:
                v_tensor = v_tensor.unsqueeze(0)
        else:
            if v_tensor.ndim == 1:
                v_tensor = v_tensor.unsqueeze(0)
        
        # KLUCZOWE: retain_grad() dla non-leaf tensors
        v_tensor.retain_grad()
        obs_grad[k] = v_tensor
    
    # Forward pass z gradientami
    features = features_extractor(obs_grad)
    features.retain_grad()
    
    # Przygotuj lstm_states jako tensory
    lstm_states_tensor = (
        torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device, requires_grad=False),
        torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device, requires_grad=False)
    )
    
    features_seq = features.unsqueeze(1)
    lstm_output, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
    latent_pi = lstm_output.squeeze(1)
    latent_pi = policy.mlp_extractor.policy_net(latent_pi)
    logits_grad = policy.action_net(latent_pi)
    
    # Gradient względem wybranej akcji
    selected_logit = logits_grad[0, action_idx]
    
    # Zero grad przed backward
    model.policy.zero_grad()
    selected_logit.backward(retain_graph=True)
    
    # Zbierz gradienty skalarów
    scalar_keys = ['direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']
    scalar_grads = {}
    scalar_values = {}
    
    for key in scalar_keys:
        if obs_grad[key].grad is not None:
            grad_magnitude = obs_grad[key].grad.abs().mean().item()
            scalar_grads[key] = grad_magnitude
        else:
            scalar_grads[key] = 0.0
        scalar_values[key] = obs_grad[key].detach().cpu().numpy()[0]
    
    # Gradient obrazu (CNN)
    image_grad_magnitude = 0.0
    if obs_grad['image'].grad is not None:
        image_grad_magnitude = obs_grad['image'].grad.abs().mean().item()
    
    # Gradient features (wewnętrzna reprezentacja)
    features_grad_magnitude = 0.0
    if features.grad is not None:
        features_grad_magnitude = features.grad.abs().mean().item()
    
    gradient_importance.append({
        'state': state_idx,
        'action': action_idx,
        'image_grad': image_grad_magnitude,
        'features_grad': features_grad_magnitude,
        **{f'{k}_grad': v for k, v in scalar_grads.items()},
        **{f'{k}_value': v for k, v in scalar_values.items()}
    })
    
    # WRÓĆ DO TRYBU EVAL
    model.policy.eval()
    
    # === ANALIZA AKTYWACJI WARSTW ===
    with torch.no_grad():
        # Ekstrahuj aktywacje z CNN
        image_input = obs_tensor['image']
        if len(image_input.shape) == 4:
            image_input = image_input.permute(0, 3, 1, 2)
        
        # DEBUGOWANIE: Sprawdź statystyki wejścia
        print(f"    DEBUG CNN Input - shape: {image_input.shape}, min: {image_input.min().item():.4f}, max: {image_input.max().item():.4f}, mean: {image_input.mean().item():.4f}")
        
        # Aktywacje każdej warstwy CNN
        cnn_activations = []
        x = image_input
        for i, layer in enumerate(features_extractor.cnn):
            x = layer(x)
            layer_name = layer.__class__.__name__
            if isinstance(layer, (torch.nn.ReLU, torch.nn.LeakyReLU)):
                # Średnia aktywacja po ReLU/LeakyReLU
                activation_mean = x.abs().mean().item()
                cnn_activations.append(activation_mean)
                print(f"    DEBUG CNN Layer {i} ({layer_name}): mean activation = {activation_mean:.6f}")
            elif isinstance(layer, torch.nn.Conv2d):
                print(f"    DEBUG CNN Layer {i} ({layer_name}): shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
            elif isinstance(layer, torch.nn.BatchNorm2d):
                print(f"    DEBUG CNN Layer {i} ({layer_name}): shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
        
        # Aktywacje skalarów
        scalars = torch.cat([
            obs_tensor['direction'],
            obs_tensor['dx_head'],
            obs_tensor['dy_head'],
            obs_tensor['front_coll'],
            obs_tensor['left_coll'],
            obs_tensor['right_coll']
        ], dim=-1)
        
        print(f"    DEBUG Scalars - shape: {scalars.shape}, values: {scalars[0].cpu().numpy()}")
        
        scalar_features_activation = features_extractor.scalar_linear(scalars)
        scalar_activation_mean = scalar_features_activation.abs().mean().item()
        print(f"    DEBUG Scalar features: mean activation = {scalar_activation_mean:.6f}")
        
        activation_importance.append({
            'state': state_idx,
            'cnn_layer1': cnn_activations[0] if len(cnn_activations) > 0 else 0,
            'cnn_layer2': cnn_activations[1] if len(cnn_activations) > 1 else 0,
            'cnn_layer3': cnn_activations[2] if len(cnn_activations) > 2 else 0,
            'scalar_activation': scalar_activation_mean
        })
    
    print(f"    Image gradient: {image_grad_magnitude:.6f}")
    print(f"    Features gradient: {features_grad_magnitude:.6f}")
    print(f"    Top scalar gradients: {sorted(scalar_grads.items(), key=lambda x: x[1], reverse=True)[:3]}")

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

# === WIZUALIZACJA WAŻNOŚCI GRADIENTÓW ===
print("\n=== Wizualizacja ważności neuronów ===")

# Przygotuj dane do wykresów
scalar_keys_display = ['direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']
avg_scalar_grads = {k: np.mean([g[f'{k}_grad'] for g in gradient_importance]) for k in scalar_keys_display}
avg_image_grad = np.mean([g['image_grad'] for g in gradient_importance])

# Wykres 1: Porównanie CNN vs Skalary (średnia ważność)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: CNN vs Skalary
components = ['CNN (Image)'] + scalar_keys_display
values = [avg_image_grad] + [avg_scalar_grads[k] for k in scalar_keys_display]
colors = ['#ff6b6b'] + ['#4ecdc4'] * len(scalar_keys_display)

bars1 = ax1.bar(range(len(components)), values, color=colors)
ax1.set_xticks(range(len(components)))
ax1.set_xticklabels(components, rotation=45, ha='right')
ax1.set_ylabel('Średnia wartość gradientu (ważność)')
ax1.set_title('Ważność komponentów obserwacji\n(gradient względem wybranej akcji)')
ax1.grid(axis='y', alpha=0.3)

# Dodaj wartości na słupkach
for i, (bar, val) in enumerate(zip(bars1, values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Subplot 2: Szczegóły skalarów
scalar_names_short = ['dir', 'dx', 'dy', 'front', 'left', 'right']
scalar_vals = [avg_scalar_grads[k] for k in scalar_keys_display]
bars2 = ax2.bar(range(len(scalar_names_short)), scalar_vals, color='#4ecdc4')
ax2.set_xticks(range(len(scalar_names_short)))
ax2.set_xticklabels(scalar_names_short)
ax2.set_ylabel('Średnia wartość gradientu')
ax2.set_title('Szczegółowa ważność skalarów')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars2, scalar_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scalar_vals)*0.01,
             f'{val:.5f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
importance_path = os.path.join(output_dir, 'neuron_importance.png')
plt.savefig(importance_path, dpi=150)
plt.close()
print(f'Wykres ważności neuronów zapisany: {importance_path}')

# Wykres 2: Ważność dla każdego stanu osobno
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for state_idx in range(3):
    ax = axes[state_idx]
    g = gradient_importance[state_idx]
    
    components = ['CNN'] + scalar_keys_display
    values = [g['image_grad']] + [g[f'{k}_grad'] for k in scalar_keys_display]
    colors = ['#ff6b6b'] + ['#4ecdc4'] * len(scalar_keys_display)
    
    bars = ax.bar(range(len(components)), values, color=colors)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.set_ylabel('Wartość gradientu')
    ax.set_title(f'Stan {state_idx}: Ważność komponentów (akcja: {action_names[g["action"]]})')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{val:.5f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
per_state_path = os.path.join(output_dir, 'importance_per_state.png')
plt.savefig(per_state_path, dpi=150)
plt.close()
print(f'Wykres ważności per stan zapisany: {per_state_path}')

# Wykres 3: Aktywacje warstw
fig, ax = plt.subplots(figsize=(12, 6))

states = [0, 1, 2]
cnn1_acts = [a['cnn_layer1'] for a in activation_importance]
cnn2_acts = [a['cnn_layer2'] for a in activation_importance]
cnn3_acts = [a['cnn_layer3'] for a in activation_importance]
scalar_acts = [a['scalar_activation'] for a in activation_importance]

x = np.arange(len(states))
width = 0.2

ax.bar(x - 1.5*width, cnn1_acts, width, label='CNN Layer 1', color='#e74c3c')
ax.bar(x - 0.5*width, cnn2_acts, width, label='CNN Layer 2', color='#e67e22')
ax.bar(x + 0.5*width, cnn3_acts, width, label='CNN Layer 3', color='#f39c12')
ax.bar(x + 1.5*width, scalar_acts, width, label='Scalar Layer', color='#3498db')

ax.set_xlabel('Stan')
ax.set_ylabel('Średnia aktywacja')
ax.set_title('Aktywacje warstw sieci neuronowej')
ax.set_xticks(x)
ax.set_xticklabels([f'Stan {s}' for s in states])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
activation_path = os.path.join(output_dir, 'layer_activations.png')
plt.savefig(activation_path, dpi=150)
plt.close()
print(f'Wykres aktywacji warstw zapisany: {activation_path}')

# Zapisz szczegółowe dane do CSV
importance_csv_path = os.path.join(output_dir, 'neuron_importance.csv')
with open(importance_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['stan', 'akcja', 'image_grad'] + [f'{k}_grad' for k in scalar_keys_display] + [f'{k}_value' for k in scalar_keys_display]
    writer.writerow(header)
    for g in gradient_importance:
        row = [g['state'], action_names[g['action']], g['image_grad']]
        row += [g[f'{k}_grad'] for k in scalar_keys_display]
        row += [str(g[f'{k}_value']) for k in scalar_keys_display]
        writer.writerow(row)
print(f'Dane ważności neuronów zapisane: {importance_csv_path}')

# Wyświetl podsumowanie
print("\n=== PODSUMOWANIE WAŻNOŚCI ===")
print(f"Średnia ważność CNN (image): {avg_image_grad:.6f}")
print("\nŚrednia ważność skalarów:")
for k in scalar_keys_display:
    print(f"  {k:12s}: {avg_scalar_grads[k]:.6f}")

# Ranking
all_components = [('CNN', avg_image_grad)] + [(k, avg_scalar_grads[k]) for k in scalar_keys_display]
all_components.sort(key=lambda x: x[1], reverse=True)
print("\nRanking najważniejszych komponentów:")
for i, (name, importance) in enumerate(all_components, 1):
    print(f"  {i}. {name:12s}: {importance:.6f}")

print("\n=== Analiza zakończona ===")
print(f"Wyniki zapisane w: {output_dir}")
print(f"Wizualizacje w: {conv_viz_dir}")
print(f"\nWygenerowane wykresy:")
print(f"  - neuron_importance.png (porównanie CNN vs skalary)")
print(f"  - importance_per_state.png (ważność dla każdego stanu)")
print(f"  - layer_activations.png (aktywacje warstw)")
print(f"  - neuron_importance.csv (szczegółowe dane)")
