import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import csv
from model import make_env
from sb3_contrib import RecurrentPPO
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from collections import defaultdict

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Ścieżka do najlepszego modelu
model_path = os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip')

# Utwórz katalogi na wyniki
output_dir = os.path.join(base_dir, 'logs', 'analyze_model')
conv_viz_dir = os.path.join(output_dir, 'conv_visualizations')
viewport_dir = os.path.join(output_dir, 'viewports')
action_probs_dir = os.path.join(output_dir, 'action_probs')
heatmap_dir = os.path.join(output_dir, 'attention_heatmaps')
lstm_dir = os.path.join(output_dir, 'lstm_analysis')
uncertainty_dir = os.path.join(output_dir, 'uncertainty_analysis')
confusion_dir = os.path.join(output_dir, 'confusion_matrix')

for dir_path in [output_dir, conv_viz_dir, viewport_dir, action_probs_dir, 
                 heatmap_dir, lstm_dir, uncertainty_dir, confusion_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Załaduj model RecurrentPPO
print("Ładowanie modelu...")
model = RecurrentPPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

print(f"\n=== Informacje o modelu ===")
print(f"CNN channels: {config['model']['convlstm']['cnn_channels']}")
print(f"Scalar hidden dims: {config['model']['convlstm']['scalar_hidden_dims']}")
print(f"Features dim: {config['model']['policy_kwargs']['features_extractor_kwargs']['features_dim']}")
print(f"LSTM hidden size: {config['model']['policy_kwargs']['lstm_hidden_size']}")
print(f"LSTM layers: {config['model']['policy_kwargs']['n_lstm_layers']}")
print(f"Dropout rate: {config['model'].get('dropout_rate', 0.1)}")
print(f"Actor network: {config['model']['policy_kwargs']['net_arch']['pi']}")
print(f"Critic network: {config['model']['policy_kwargs']['net_arch']['vf']}")
print(f"Critic LSTM enabled: {config['model']['policy_kwargs']['enable_critic_lstm']}")

# Przygotuj środowisko
env = make_env(render_mode=None, grid_size=16)()

# Nazwy akcji
action_names = ['lewo', 'prosto', 'prawo']

# Listy do przechowywania wyników
action_probs_list = []
detailed_activations = []
layer_gradients = []
attention_heatmaps = []
lstm_memory_data = []
uncertainty_data = []

# Dla confusion matrix - zbieramy dane z pełnych epizodów
confusion_data = []

print("\n=== Analiza ważności komponentów ===")

# Inicjalizuj stany LSTM
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

# ===================================================
# CZĘŚĆ 1: PODSTAWOWA ANALIZA (3 STANY)
# ===================================================
for state_idx in range(3):
    print(f"\nAnaliza stanu {state_idx}...")
    
    # Pobierz obserwację
    obs, _ = env.reset()
    lstm_states = None
    
    # Wykonaj kilka kroków, aby uzyskać różne stany
    for _ in range(state_idx * 5):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        if torch.is_tensor(action):
            action = int(action.item())
        elif isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        action = int(np.clip(action, 0, len(action_names) - 1))
        obs, _, done, _, _ = env.step(action)
        episode_starts = np.array([done], dtype=bool)
        if done:
            obs, _ = env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

    # Przygotuj obs_tensor jako dict z tensorami
    obs_tensor = {}
    for k, v in obs.items():
        v_np = v if isinstance(v, np.ndarray) else np.array(v)
        v_tensor = torch.tensor(v_np, dtype=torch.float32, device=policy.device)
        
        if k == 'image':
            if v_tensor.ndim == 3:
                v_tensor = v_tensor.unsqueeze(0)
        else:
            if v_tensor.ndim == 1:
                v_tensor = v_tensor.unsqueeze(0)
        
        obs_tensor[k] = v_tensor

    # === FORWARD PASS Z ZAPISYWANIEM AKTYWACJI ===
    with torch.no_grad():
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        if torch.is_tensor(action):
            action_idx = int(action.item())
        elif isinstance(action, np.ndarray):
            action_idx = int(action.item())
        else:
            action_idx = int(action)
        action_idx = int(np.clip(action_idx, 0, len(action_names) - 1))
        
        state_activations = {
            'state': state_idx,
            'action': action_idx
        }
        
        # === CNN FEATURES ===
        image = obs_tensor['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        cnn_features = features_extractor.cnn(image)
        state_activations['cnn_output_mean'] = cnn_features.abs().mean().item()
        state_activations['cnn_output_max'] = cnn_features.abs().max().item()
        state_activations['cnn_output_std'] = cnn_features.std().item()
        
        # === SCALAR FEATURES ===
        scalars = torch.cat([
            obs_tensor['direction'],
            obs_tensor['dx_head'],
            obs_tensor['dy_head'],
            obs_tensor['front_coll'],
            obs_tensor['left_coll'],
            obs_tensor['right_coll']
        ], dim=-1)
        
        scalar_features = features_extractor.scalar_linear(scalars)
        state_activations['scalar_output_mean'] = scalar_features.abs().mean().item()
        state_activations['scalar_output_max'] = scalar_features.abs().max().item()
        state_activations['scalar_output_std'] = scalar_features.std().item()
        
        # === COMBINED FEATURES ===
        features = features_extractor(obs_tensor)
        state_activations['features_mean'] = features.abs().mean().item()
        state_activations['features_max'] = features.abs().max().item()
        state_activations['features_std'] = features.std().item()
        
        # === LSTM ACTOR ===
        lstm_states_tensor = (
            torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device),
            torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device)
        )
        
        features_seq = features.unsqueeze(1)
        lstm_output, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_tensor)
        latent_pi = lstm_output.squeeze(1)
        
        state_activations['lstm_output_mean'] = latent_pi.abs().mean().item()
        state_activations['lstm_output_max'] = latent_pi.abs().max().item()
        state_activations['lstm_output_std'] = latent_pi.std().item()
        
        state_activations['lstm_hidden_mean'] = new_lstm_states[0].abs().mean().item()
        state_activations['lstm_hidden_max'] = new_lstm_states[0].abs().max().item()
        state_activations['lstm_cell_mean'] = new_lstm_states[1].abs().mean().item()
        state_activations['lstm_cell_max'] = new_lstm_states[1].abs().max().item()
        
        # === MLP EXTRACTOR ===
        latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
        state_activations['mlp_pi_mean'] = latent_pi_mlp.abs().mean().item()
        state_activations['mlp_pi_max'] = latent_pi_mlp.abs().max().item()
        
        # === ACTION NETWORK ===
        logits = policy.action_net(latent_pi_mlp)
        state_activations['logits_mean'] = logits.abs().mean().item()
        state_activations['logits_max'] = logits.abs().max().item()
        
        detailed_activations.append(state_activations)

    # === Prawdopodobieństwa akcji ===
    action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    action_probs_list.append({
        'state': state_idx,
        'p_lewo': action_probs[0],
        'p_prosto': action_probs[1],
        'p_prawo': action_probs[2],
        'akcja': action_idx
    })
    
    # === UNCERTAINTY ANALYSIS ===
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
    max_prob = np.max(action_probs)
    uncertainty_data.append({
        'state': state_idx,
        'entropy': entropy,
        'max_prob': max_prob,
        'certainty': max_prob,
        'action': action_names[action_idx]
    })

    # ===================================================
    # HEATMAPA ATTENTION/WAŻNOŚCI
    # ===================================================
    print(f"  Generowanie attention heatmap...")
    
    model.policy.train()
    
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
        
        obs_grad[k] = v_tensor
    
    # Forward pass - WAŻNE: permute PRZED retain_grad
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    # TERAZ retain_grad na przekształconym tensorze
    image_grad.retain_grad()
    
    # Całe features extraction
    cnn_output = features_extractor.cnn(image_grad)
    
    scalars_grad = torch.cat([
        obs_grad['direction'],
        obs_grad['dx_head'],
        obs_grad['dy_head'],
        obs_grad['front_coll'],
        obs_grad['left_coll'],
        obs_grad['right_coll']
    ], dim=-1)
    
    scalar_output = features_extractor.scalar_linear(scalars_grad)
    combined = torch.cat([cnn_output, scalar_output], dim=-1)
    features_final = features_extractor.final_linear(combined)
    
    # LSTM
    lstm_states_tensor = (
        torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device, requires_grad=False),
        torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device, requires_grad=False)
    )
    
    features_seq = features_final.unsqueeze(1)
    lstm_out, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
    latent_pi_grad = lstm_out.squeeze(1)
    
    # MLP i action
    latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi_grad)
    logits_grad = policy.action_net(latent_pi_mlp)
    selected_logit = logits_grad[0, action_idx]
    
    # Backward
    model.policy.zero_grad()
    selected_logit.backward()
    
    # Gradient względem input image
    if image_grad.grad is not None:
        # Shape: [1, 1, 16, 16]
        grad_map = image_grad.grad[0, 0].abs().detach().cpu().numpy()
        
        # Normalizacja
        if grad_map.max() > 0:
            grad_map = grad_map / grad_map.max()
        
        # Smooth dla lepszej wizualizacji
        grad_map_smooth = gaussian_filter(grad_map, sigma=0.8)
        
        attention_heatmaps.append({
            'state': state_idx,
            'action': action_idx,
            'heatmap': grad_map_smooth,
            'viewport': obs['image'][:, :, 0]
        })
        
        # Wizualizacja
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Viewport
        axes[0].imshow(obs['image'][:, :, 0], cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Viewport (Stan {state_idx})')
        axes[0].axis('off')
        
        # Heatmapa
        im1 = axes[1].imshow(grad_map_smooth, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f'Attention Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(obs['image'][:, :, 0], cmap='viridis', interpolation='nearest', alpha=0.7)
        axes[2].imshow(grad_map_smooth, cmap='hot', interpolation='bilinear', alpha=0.5)
        axes[2].set_title(f'Overlay (Akcja: {action_names[action_idx]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        heatmap_path = os.path.join(heatmap_dir, f'attention_heatmap_state_{state_idx}.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Attention heatmap zapisana: {heatmap_path}")
    
    model.policy.eval()
    
    # === ANALIZA GRADIENTÓW DLA KAŻDEJ WARSTWY (z oryginalnego kodu) ===
    print(f"  Obliczanie gradientów warstw...")
    
    model.policy.train()
    
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
        
        v_tensor.retain_grad()
        obs_grad[k] = v_tensor
    
    # Forward pass z zachowaniem intermediate activations
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    # CNN layers z gradient tracking
    cnn_intermediates = []
    x_cnn = image_grad
    for i, layer in enumerate(features_extractor.cnn):
        x_cnn = layer(x_cnn)
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.LeakyReLU)):
            x_cnn.retain_grad()
            cnn_intermediates.append(('cnn', i, layer.__class__.__name__, x_cnn))
    
    cnn_output = x_cnn
    
    # Scalar layers
    scalars_grad = torch.cat([
        obs_grad['direction'],
        obs_grad['dx_head'],
        obs_grad['dy_head'],
        obs_grad['front_coll'],
        obs_grad['left_coll'],
        obs_grad['right_coll']
    ], dim=-1)
    
    scalar_intermediates = []
    x_scalar = scalars_grad
    for i, layer in enumerate(features_extractor.scalar_linear):
        x_scalar = layer(x_scalar)
        if isinstance(layer, (torch.nn.Linear, torch.nn.LeakyReLU)):
            x_scalar.retain_grad()
            scalar_intermediates.append(('scalar', i, layer.__class__.__name__, x_scalar))
    
    scalar_output = x_scalar
    
    # Combined features
    combined = torch.cat([cnn_output, scalar_output], dim=-1)
    combined.retain_grad()
    features_final = features_extractor.final_linear(combined)
    features_final.retain_grad()
    
    # LSTM
    lstm_states_tensor = (
        torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device, requires_grad=False),
        torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device, requires_grad=False)
    )
    
    features_seq = features_final.unsqueeze(1)
    lstm_out, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
    lstm_out.retain_grad()
    latent_pi_grad = lstm_out.squeeze(1)
    latent_pi_grad.retain_grad()
    
    # MLP layers
    mlp_intermediates = []
    x_mlp = latent_pi_grad
    for i, layer in enumerate(policy.mlp_extractor.policy_net):
        x_mlp = layer(x_mlp)
        if isinstance(layer, (torch.nn.Linear, torch.nn.ReLU)):
            x_mlp.retain_grad()
            mlp_intermediates.append(('mlp', i, layer.__class__.__name__, x_mlp))
    
    latent_pi_final = x_mlp
    
    # Action logits
    logits_grad = policy.action_net(latent_pi_final)
    selected_logit = logits_grad[0, action_idx]
    
    # Backward pass
    model.policy.zero_grad()
    selected_logit.backward(retain_graph=True)
    
    # Zbierz gradienty dla wszystkich warstw
    state_layer_grads = {
        'state': state_idx,
        'action': action_idx,
        'layers': []
    }
    
    # CNN gradienty
    for layer_type, layer_idx, layer_name, activation in cnn_intermediates:
        if activation.grad is not None:
            grad_mean = activation.grad.abs().mean().item()
            grad_max = activation.grad.abs().max().item()
            activation_mean = activation.abs().mean().item()
            
            state_layer_grads['layers'].append({
                'network': 'CNN',
                'layer_idx': layer_idx,
                'layer_name': layer_name,
                'activation_mean': activation_mean,
                'gradient_mean': grad_mean,
                'gradient_max': grad_max
            })
    
    # Scalar gradienty
    for layer_type, layer_idx, layer_name, activation in scalar_intermediates:
        if activation.grad is not None:
            grad_mean = activation.grad.abs().mean().item()
            grad_max = activation.grad.abs().max().item()
            activation_mean = activation.abs().mean().item()
            
            state_layer_grads['layers'].append({
                'network': 'Scalar',
                'layer_idx': layer_idx,
                'layer_name': layer_name,
                'activation_mean': activation_mean,
                'gradient_mean': grad_mean,
                'gradient_max': grad_max
            })
    
    # Combined features
    if features_final.grad is not None:
        state_layer_grads['layers'].append({
            'network': 'Features',
            'layer_idx': 0,
            'layer_name': 'Combined',
            'activation_mean': features_final.abs().mean().item(),
            'gradient_mean': features_final.grad.abs().mean().item(),
            'gradient_max': features_final.grad.abs().max().item()
        })
    
    # LSTM
    if latent_pi_grad.grad is not None:
        state_layer_grads['layers'].append({
            'network': 'LSTM',
            'layer_idx': 0,
            'layer_name': 'Output',
            'activation_mean': latent_pi_grad.abs().mean().item(),
            'gradient_mean': latent_pi_grad.grad.abs().mean().item(),
            'gradient_max': latent_pi_grad.grad.abs().max().item()
        })
    
    # MLP gradienty
    for layer_type, layer_idx, layer_name, activation in mlp_intermediates:
        if activation.grad is not None:
            grad_mean = activation.grad.abs().mean().item()
            grad_max = activation.grad.abs().max().item()
            activation_mean = activation.abs().mean().item()
            
            state_layer_grads['layers'].append({
                'network': 'MLP',
                'layer_idx': layer_idx,
                'layer_name': layer_name,
                'activation_mean': activation_mean,
                'gradient_mean': grad_mean,
                'gradient_max': grad_max
            })
    
    layer_gradients.append(state_layer_grads)
    
    model.policy.eval()
    
    # === Wizualizacja CNN output ===
    with torch.no_grad():
        first_conv = features_extractor.cnn[0]
        cnn_activation = first_conv(image).detach().cpu().numpy()[0]
    
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
    
    legend_elements = [
        Patch(facecolor='purple', label='Ściana (-1.0)'),
        Patch(facecolor='black', label='Puste (0.0)'),
        Patch(facecolor='cyan', label='Ciało (0.5)'),
        Patch(facecolor='orange', label='Jedzenie (0.75)'),
        Patch(facecolor='yellow', label='Głowa (1.0)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    viewport_path = os.path.join(viewport_dir, f'viewport_state_{state_idx}.png')
    plt.savefig(viewport_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Viewport zapisany: {viewport_path}')

    print(f"  Wybrany action: {action_names[action_idx]}")
    print(f"  Prawdopodobieństwa: lewo={action_probs[0]:.3f}, prosto={action_probs[1]:.3f}, prawo={action_probs[2]:.3f}")

# ===================================================
# CZĘŚĆ 2: ANALIZA KONSYSTENCJI LSTM (PAMIĘĆ)
# ===================================================
print("\n=== Analiza konsystencji LSTM (pamięć) ===")

# Zbieramy dane z pełnego epizodu
obs, _ = env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

episode_lstm_data = []
step_count = 0
max_steps = 50  # Analizujemy 50 kroków

while step_count < max_steps:
    # Predict
    with torch.no_grad():
        action, new_lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        
        if torch.is_tensor(action):
            action_idx = int(action.item())
        elif isinstance(action, np.ndarray):
            action_idx = int(action.item())
        else:
            action_idx = int(action)
        action_idx = int(np.clip(action_idx, 0, len(action_names) - 1))
        
        # Zapisz dane LSTM
        if lstm_states is not None:
            hidden_state = lstm_states[0]  # Shape: [n_layers, batch, hidden_size]
            cell_state = lstm_states[1]
            
            # Mean magnitude per layer
            hidden_mean = np.abs(hidden_state).mean(axis=(1, 2))  # [n_layers]
            cell_mean = np.abs(cell_state).mean(axis=(1, 2))
            
            # Change from previous step (if available)
            if len(episode_lstm_data) > 0:
                prev_hidden = episode_lstm_data[-1]['hidden_state']
                hidden_change = np.abs(hidden_state - prev_hidden).mean()
            else:
                hidden_change = 0.0
            
            episode_lstm_data.append({
                'step': step_count,
                'action': action_names[action_idx],
                'hidden_state': hidden_state.copy(),
                'cell_state': cell_state.copy(),
                'hidden_mean_layer0': hidden_mean[0] if len(hidden_mean) > 0 else 0,
                'cell_mean_layer0': cell_mean[0] if len(cell_mean) > 0 else 0,
                'hidden_change': hidden_change
            })
        
        lstm_states = new_lstm_states
    
    # Step
    obs, reward, done, truncated, info = env.step(action_idx)
    episode_starts = np.array([done or truncated], dtype=bool)
    step_count += 1
    
    if done or truncated:
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

# Wizualizacja LSTM memory evolution
if len(episode_lstm_data) > 0:
    steps = [d['step'] for d in episode_lstm_data]
    hidden_means = [d['hidden_mean_layer0'] for d in episode_lstm_data]
    cell_means = [d['cell_mean_layer0'] for d in episode_lstm_data]
    hidden_changes = [d['hidden_change'] for d in episode_lstm_data]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Hidden state magnitude over time
    axes[0].plot(steps, hidden_means, label='Hidden State (Layer 0)', color='#9b59b6', linewidth=2)
    axes[0].set_xlabel('Krok')
    axes[0].set_ylabel('Średnia magnitude')
    axes[0].set_title('Ewolucja Hidden State LSTM')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Cell state magnitude over time
    axes[1].plot(steps, cell_means, label='Cell State (Layer 0)', color='#8e44ad', linewidth=2)
    axes[1].set_xlabel('Krok')
    axes[1].set_ylabel('Średnia magnitude')
    axes[1].set_title('Ewolucja Cell State LSTM')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Hidden state change (memory update rate)
    axes[2].plot(steps[1:], hidden_changes[1:], label='Zmiana Hidden State', color='#e74c3c', linewidth=2)
    axes[2].set_xlabel('Krok')
    axes[2].set_ylabel('Wielkość zmiany')
    axes[2].set_title('Tempo aktualizacji pamięci LSTM')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    lstm_evolution_path = os.path.join(lstm_dir, 'lstm_memory_evolution.png')
    plt.savefig(lstm_evolution_path, dpi=150)
    plt.close()
    print(f'✓ LSTM memory evolution zapisana: {lstm_evolution_path}')
    
    # Heatmapa hidden state w czasie (dla wszystkich neuronów)
    hidden_states_matrix = np.array([d['hidden_state'][0, 0, :] for d in episode_lstm_data])  # [steps, hidden_size]
    
    plt.figure(figsize=(16, 8))
    plt.imshow(hidden_states_matrix.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Wartość aktywacji')
    plt.xlabel('Krok czasowy')
    plt.ylabel('Neuron LSTM')
    plt.title('Aktywacja wszystkich neuronów LSTM w czasie')
    plt.tight_layout()
    lstm_heatmap_path = os.path.join(lstm_dir, 'lstm_neurons_heatmap.png')
    plt.savefig(lstm_heatmap_path, dpi=150)
    plt.close()
    print(f'✓ LSTM neurons heatmap zapisana: {lstm_heatmap_path}')
    
    # Zapisz dane LSTM do CSV
    lstm_csv_path = os.path.join(lstm_dir, 'lstm_memory_data.csv')
    with open(lstm_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'action', 'hidden_mean_layer0', 'cell_mean_layer0', 'hidden_change'])
        for d in episode_lstm_data:
            writer.writerow([
                d['step'],
                d['action'],
                d['hidden_mean_layer0'],
                d['cell_mean_layer0'],
                d['hidden_change']
            ])
    print(f'✓ LSTM memory data zapisana: {lstm_csv_path}')

# ===================================================
# CZĘŚĆ 3: CONFUSION MATRIX (zbieranie danych z epizodów)
# ===================================================
print("\n=== Zbieranie danych dla Confusion Matrix ===")

num_episodes = 20
confusion_matrix = np.zeros((3, 3))  # [expected_action, actual_action]
action_history = []

for episode_idx in range(num_episodes):
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    step = 0
    
    while not done and step < 100:
        # Predict
        with torch.no_grad():
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            if torch.is_tensor(action):
                action_idx = int(action.item())
            elif isinstance(action, np.ndarray):
                action_idx = int(action.item())
            else:
                action_idx = int(action)
            action_idx = int(np.clip(action_idx, 0, len(action_names) - 1))
        
        # Heurystyka: "oczekiwana akcja" na podstawie prostej logiki
        # (możesz to zastąpić swoją logiką ekspercką)
        dx = obs['dx_head'][0]
        dy = obs['dy_head'][0]
        direction = obs['direction'][0]
        front_coll = obs['front_coll'][0]
        left_coll = obs['left_coll'][0]
        right_coll = obs['right_coll'][0]
        
        # Prosta heurystyka: jeśli jedzenie jest z przodu i nie ma kolizji -> prosto
        # jeśli jedzenie z lewej -> lewo, z prawej -> prawo
        expected_action = 1  # default: prosto
        
        if front_coll > 0.5:  # kolizja z przodu
            if left_coll < 0.5:
                expected_action = 0  # lewo
            elif right_coll < 0.5:
                expected_action = 2  # prawo
        else:
            # Skręcaj w stronę jedzenia
            food_angle = np.arctan2(dy, dx) * 180 / np.pi
            # Normalizuj względem kierunku węża
            if abs(food_angle) < 30:
                expected_action = 1  # prosto
            elif food_angle < -30:
                expected_action = 0  # lewo
            elif food_angle > 30:
                expected_action = 2  # prawo
        
        confusion_matrix[expected_action, action_idx] += 1
        action_history.append({
            'episode': episode_idx,
            'step': step,
            'expected': action_names[expected_action],
            'actual': action_names[action_idx],
            'match': expected_action == action_idx
        })
        
        # Step
        obs, reward, done, truncated, info = env.step(action_idx)
        episode_starts = np.array([done or truncated], dtype=bool)
        done = done or truncated
        step += 1
    
    if (episode_idx + 1) % 5 == 0:
        print(f"  Przetworzono {episode_idx + 1}/{num_episodes} epizodów")

# Wizualizacja Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')

# Dodaj wartości do komórek
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, int(confusion_matrix[i, j]),
                      ha="center", va="center", color="black", fontsize=14, fontweight='bold')

ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(action_names)
ax.set_yticklabels(action_names)
ax.set_xlabel('Akcja Modelu (Actual)', fontsize=12)
ax.set_ylabel('Oczekiwana Akcja (Expected)', fontsize=12)
ax.set_title('Confusion Matrix - Porównanie z Heurystyką', fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='Liczba przypadków')
plt.tight_layout()
confusion_path = os.path.join(confusion_dir, 'confusion_matrix.png')
plt.savefig(confusion_path, dpi=150)
plt.close()
print(f'✓ Confusion matrix zapisana: {confusion_path}')

# Accuracy
total = confusion_matrix.sum()
correct = np.trace(confusion_matrix)
accuracy = correct / total if total > 0 else 0

print(f"\n📊 Confusion Matrix Stats:")
print(f"   Zgodność z heurystyką: {accuracy*100:.1f}%")
print(f"   Całkowita liczba akcji: {int(total)}")
print(f"   Zgodnych akcji: {int(correct)}")

# Zapisz confusion matrix do CSV
confusion_csv_path = os.path.join(confusion_dir, 'confusion_matrix.csv')
confusion_df = np.vstack([
    [''] + action_names,
    *[[action_names[i]] + list(confusion_matrix[i, :]) for i in range(3)]
])
np.savetxt(confusion_csv_path, confusion_df, delimiter=',', fmt='%s')
print(f'✓ Confusion matrix CSV zapisana: {confusion_csv_path}')

# ===================================================
# CZĘŚĆ 4: UNCERTAINTY ANALYSIS (rozszerzona)
# ===================================================
print("\n=== Rozszerzona analiza uncertainty ===")

# Zbierz więcej danych uncertainty z dodatkowych epizodów
extended_uncertainty = []

for episode_idx in range(10):
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    step = 0
    
    while not done and step < 50:
        with torch.no_grad():
            # Przygotuj obs_tensor
            obs_tensor = {}
            for k, v in obs.items():
                v_np = v if isinstance(v, np.ndarray) else np.array(v)
                v_tensor = torch.tensor(v_np, dtype=torch.float32, device=policy.device)
                
                if k == 'image':
                    if v_tensor.ndim == 3:
                        v_tensor = v_tensor.unsqueeze(0)
                else:
                    if v_tensor.ndim == 1:
                        v_tensor = v_tensor.unsqueeze(0)
                
                obs_tensor[k] = v_tensor
            
            # Get features
            features = features_extractor(obs_tensor)
            
            # LSTM
            lstm_states_tensor = (
                torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device) if lstm_states is not None else None,
                torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device) if lstm_states is not None else None
            )
            
            if lstm_states_tensor[0] is not None:
                features_seq = features.unsqueeze(1)
                lstm_output, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_tensor)
                latent_pi = lstm_output.squeeze(1)
            else:
                # Initial state
                batch_size = 1
                n_layers = policy.lstm_actor.num_layers
                hidden_size = policy.lstm_actor.hidden_size
                device = policy.device
                lstm_states_init = (
                    torch.zeros(n_layers, batch_size, hidden_size, device=device),
                    torch.zeros(n_layers, batch_size, hidden_size, device=device)
                )
                features_seq = features.unsqueeze(1)
                lstm_output, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_init)
                latent_pi = lstm_output.squeeze(1)
            
            # MLP and action
            latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
            logits = policy.action_net(latent_pi_mlp)
            action_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            action_idx = np.argmax(action_probs)
            
            # Uncertainty metrics
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
            max_prob = np.max(action_probs)
            margin = np.partition(action_probs, -2)[-1] - np.partition(action_probs, -2)[-2]
            
            lstm_states = (new_lstm_states[0].cpu().numpy(), new_lstm_states[1].cpu().numpy())
        
        # Step
        obs, reward, done, truncated, info = env.step(action_idx)
        episode_starts = np.array([done or truncated], dtype=bool)
        
        extended_uncertainty.append({
            'episode': episode_idx,
            'step': step,
            'entropy': entropy,
            'max_prob': max_prob,
            'margin': margin,
            'action': action_names[action_idx],
            'reward': reward
        })
        
        done = done or truncated
        step += 1

# Wizualizacja Uncertainty
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Histogram entropii
entropies = [d['entropy'] for d in extended_uncertainty]
axes[0, 0].hist(entropies, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', linewidth=2, label=f'Średnia: {np.mean(entropies):.3f}')
axes[0, 0].set_xlabel('Entropia')
axes[0, 0].set_ylabel('Częstość')
axes[0, 0].set_title('Rozkład Entropii Decyzji')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Max probability distribution
max_probs = [d['max_prob'] for d in extended_uncertainty]
axes[0, 1].hist(max_probs, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, label=f'Średnia: {np.mean(max_probs):.3f}')
axes[0, 1].set_xlabel('Max Probability')
axes[0, 1].set_ylabel('Częstość')
axes[0, 1].set_title('Rozkład Pewności Modelu')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Entropy vs Reward
axes[1, 0].scatter([d['entropy'] for d in extended_uncertainty], 
                   [d['reward'] for d in extended_uncertainty],
                   alpha=0.5, c='#e74c3c', s=20)
axes[1, 0].set_xlabel('Entropia')
axes[1, 0].set_ylabel('Reward')
axes[1, 0].set_title('Entropia vs Reward (czy niepewność szkodzi?)')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Certainty categories
# Podziel na kategorie: High certainty (max_prob > 0.8), Medium (0.5-0.8), Low (<0.5)
high_cert = [d for d in extended_uncertainty if d['max_prob'] > 0.8]
medium_cert = [d for d in extended_uncertainty if 0.5 <= d['max_prob'] <= 0.8]
low_cert = [d for d in extended_uncertainty if d['max_prob'] < 0.5]

categories = ['High\n(>0.8)', 'Medium\n(0.5-0.8)', 'Low\n(<0.5)']
counts = [len(high_cert), len(medium_cert), len(low_cert)]
colors_cert = ['#2ecc71', '#f39c12', '#e74c3c']

bars = axes[1, 1].bar(categories, counts, color=colors_cert, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Liczba decyzji')
axes[1, 1].set_title('Kategorie Pewności Modelu')
axes[1, 1].grid(axis='y', alpha=0.3)

# Dodaj wartości na słupkach
for bar, count in zip(bars, counts):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/len(extended_uncertainty)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10)

plt.tight_layout()
uncertainty_path = os.path.join(uncertainty_dir, 'uncertainty_analysis.png')
plt.savefig(uncertainty_path, dpi=150)
plt.close()
print(f'✓ Uncertainty analysis zapisana: {uncertainty_path}')

# Zapisz uncertainty data do CSV
uncertainty_csv_path = os.path.join(uncertainty_dir, 'uncertainty_data.csv')
with open(uncertainty_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'step', 'entropy', 'max_prob', 'margin', 'action', 'reward'])
    for d in extended_uncertainty:
        writer.writerow([
            d['episode'],
            d['step'],
            d['entropy'],
            d['max_prob'],
            d['margin'],
            d['action'],
            d['reward']
        ])
print(f'✓ Uncertainty data zapisana: {uncertainty_csv_path}')

print(f"\n📊 Uncertainty Stats:")
print(f"   Średnia entropia: {np.mean(entropies):.3f}")
print(f"   Średnia pewność (max_prob): {np.mean(max_probs):.3f}")
print(f"   Decyzje wysokiej pewności: {len(high_cert)} ({len(high_cert)/len(extended_uncertainty)*100:.1f}%)")
print(f"   Decyzje średniej pewności: {len(medium_cert)} ({len(medium_cert)/len(extended_uncertainty)*100:.1f}%)")
print(f"   Decyzje niskiej pewności: {len(low_cert)} ({len(low_cert)/len(extended_uncertainty)*100:.1f}%)")

# ===================================================
# CZĘŚĆ 5: ORYGINALNY KOD - BOTTLENECK ANALYSIS
# ===================================================
print("\n=== Analiza bottlenecków (aktywacje vs gradienty) ===")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

bottleneck_report = []

for state_idx in range(3):
    ax = axes[state_idx]
    grad_data = layer_gradients[state_idx]
    
    layers_info = grad_data['layers']
    
    layer_labels = []
    activations = []
    gradients = []
    colors = []
    
    for layer_info in layers_info:
        network = layer_info['network']
        layer_name = f"{network}\n{layer_info['layer_name']}{layer_info['layer_idx']}"
        layer_labels.append(layer_name)
        activations.append(layer_info['activation_mean'])
        gradients.append(layer_info['gradient_mean'])
        
        if network == 'CNN':
            colors.append('#e74c3c')
        elif network == 'Scalar':
            colors.append('#3498db')
        elif network == 'Features':
            colors.append('#2ecc71')
        elif network == 'LSTM':
            colors.append('#9b59b6')
        elif network == 'MLP':
            colors.append('#f39c12')
        else:
            colors.append('#95a5a6')
    
    x = np.arange(len(layer_labels))
    width = 0.35
    
    act_max = max(activations) if max(activations) > 0 else 1
    grad_max = max(gradients) if max(gradients) > 0 else 1
    
    activations_norm = [a / act_max for a in activations]
    gradients_norm = [g / grad_max for g in gradients]
    
    bars1 = ax.bar(x - width/2, activations_norm, width, label='Aktywacja (norm)', color=colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, gradients_norm, width, label='Gradient (norm)', color=colors, alpha=0.4, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Warstwa')
    ax.set_ylabel('Wartość znormalizowana')
    ax.set_title(f'Stan {state_idx}: Aktywacje vs Gradienty (akcja: {action_names[grad_data["action"]]})')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for i, (act_n, grad_n, layer_info) in enumerate(zip(activations_norm, gradients_norm, layers_info)):
        bottleneck_type = None
        severity = None
        
        if act_n > 0.5 and grad_n < 0.2:
            bottleneck_type = 'Gradient Vanishing'
            severity = 'HIGH'
            ax.text(i, max(act_n, grad_n) + 0.05, '⚠️', ha='center', fontsize=16, color='red')
        elif act_n < 0.1:
            bottleneck_type = 'Dead/Underutilized'
            severity = 'MEDIUM'
            ax.text(i, 0.05, '⚠️', ha='center', fontsize=14, color='orange')
        elif act_n > 0.7 and grad_n > 0.7:
            bottleneck_type = 'Healthy (High Flow)'
            severity = 'GOOD'
        
        if bottleneck_type:
            bottleneck_report.append({
                'state': state_idx,
                'network': layer_info['network'],
                'layer': f"{layer_info['layer_name']}{layer_info['layer_idx']}",
                'activation': layer_info['activation_mean'],
                'gradient': layer_info['gradient_mean'],
                'activation_norm': act_n,
                'gradient_norm': grad_n,
                'bottleneck_type': bottleneck_type,
                'severity': severity
            })

plt.tight_layout()
bottleneck_path = os.path.join(output_dir, 'bottleneck_analysis.png')
plt.savefig(bottleneck_path, dpi=150)
plt.close()
print(f'Analiza bottlenecków zapisana: {bottleneck_path}')

# Raport bottlenecków
print("\n" + "="*80)
print("=== RAPORT BOTTLENECKÓW ===")
print("="*80)

high_severity = [b for b in bottleneck_report if b['severity'] == 'HIGH']
medium_severity = [b for b in bottleneck_report if b['severity'] == 'MEDIUM']

if high_severity:
    print("\n🔴 WYSOKIE RYZYKO BOTTLENECKÓW (Gradient Vanishing):")
    print("-" * 80)
    for b in high_severity:
        print(f"  Stan {b['state']} | {b['network']:8s} | {b['layer']:15s}")
        print(f"    Aktywacja: {b['activation']:.4f} (norm: {b['activation_norm']:.2f})")
        print(f"    Gradient:  {b['gradient']:.6f} (norm: {b['gradient_norm']:.2f})")
        print(f"    Problem: {b['bottleneck_type']}")
        print()

if medium_severity:
    print("\n🟠 ŚREDNIE RYZYKO BOTTLENECKÓW (Dead/Underutilized Neurons):")
    print("-" * 80)
    for b in medium_severity:
        print(f"  Stan {b['state']} | {b['network']:8s} | {b['layer']:15s}")
        print(f"    Aktywacja: {b['activation']:.4f} (norm: {b['activation_norm']:.2f})")
        print(f"    Gradient:  {b['gradient']:.6f} (norm: {b['gradient_norm']:.2f})")
        print(f"    Problem: {b['bottleneck_type']}")
        print()

if not high_severity and not medium_severity:
    print("\n✅ Nie wykryto żadnych bottlenecków o wysokim lub średnim ryzyku!")
    print("    Wszystkie warstwy działają w zdrowym zakresie.")

bottleneck_csv_path = os.path.join(output_dir, 'bottleneck_report.csv')
if bottleneck_report:
    with open(bottleneck_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=bottleneck_report[0].keys())
        writer.writeheader()
        writer.writerows(bottleneck_report)
    print(f"\nRaport bottlenecków zapisany: {bottleneck_csv_path}")

# Wykres: Przegląd aktywacji neuronów
print("\n=== Generowanie wykresu przeglądu aktywacji ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

states = [0, 1, 2]
cnn_means = [d['cnn_output_mean'] for d in detailed_activations]
scalar_means = [d['scalar_output_mean'] for d in detailed_activations]
features_means = [d['features_mean'] for d in detailed_activations]

x = np.arange(len(states))
width = 0.25

axes[0].bar(x - width, cnn_means, width, label='CNN Output', color='#e74c3c')
axes[0].bar(x, scalar_means, width, label='Scalar Output', color='#3498db')
axes[0].bar(x + width, features_means, width, label='Combined Features', color='#2ecc71')
axes[0].set_xlabel('Stan')
axes[0].set_ylabel('Średnia aktywacja')
axes[0].set_title('Porównanie CNN vs Scalar Features')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'Stan {s}' for s in states])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

lstm_output_means = [d['lstm_output_mean'] for d in detailed_activations]
lstm_hidden_means = [d['lstm_hidden_mean'] for d in detailed_activations]
lstm_cell_means = [d['lstm_cell_mean'] for d in detailed_activations]

axes[1].bar(x - width, lstm_output_means, width, label='LSTM Output', color='#9b59b6')
axes[1].bar(x, lstm_hidden_means, width, label='Hidden State', color='#8e44ad')
axes[1].bar(x + width, lstm_cell_means, width, label='Cell State', color='#6c3483')
axes[1].set_xlabel('Stan')
axes[1].set_ylabel('Średnia aktywacja')
axes[1].set_title('Aktywacje LSTM')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'Stan {s}' for s in states])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

mlp_means = [d['mlp_pi_mean'] for d in detailed_activations]
logits_means = [d['logits_mean'] for d in detailed_activations]

axes[2].bar(x - width/2, mlp_means, width, label='MLP Output', color='#f39c12')
axes[2].bar(x + width/2, logits_means, width, label='Logits', color='#e67e22')
axes[2].set_xlabel('Stan')
axes[2].set_ylabel('Średnia aktywacja')
axes[2].set_title('Aktywacje MLP i Logits')
axes[2].set_xticks(x)
axes[2].set_xticklabels([f'Stan {s}' for s in states])
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
neuron_activation_path = os.path.join(output_dir, 'neuron_activations_overview.png')
plt.savefig(neuron_activation_path, dpi=150)
plt.close()
print(f'Wykres aktywacji neuronów zapisany: {neuron_activation_path}')

# Wykres: Prawdopodobieństwa akcji
print("\n=== Generowanie wykresu prawdopodobieństw akcji ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for state_idx in range(3):
    ax = axes[state_idx]
    probs = action_probs_list[state_idx]
    selected_action = action_names[probs['akcja']]
    
    bars = ax.bar(action_names, [probs['p_lewo'], probs['p_prosto'], probs['p_prawo']], color=['#95a5a6', '#95a5a6', '#95a5a6'])
    bars[probs['akcja']].set_color('#2ecc71')
    
    ax.set_xlabel('Akcja')
    ax.set_ylabel('Prawdopodobieństwo')
    ax.set_title(f'Stan {state_idx}: Wybrano "{selected_action}"')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, prob) in enumerate(zip(bars, [probs['p_lewo'], probs['p_prosto'], probs['p_prawo']])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    print(f"Stan {state_idx}: Wybrano '{selected_action}' - p_lewo={probs['p_lewo']:.3f}, p_prosto={probs['p_prosto']:.3f}, p_prawo={probs['p_prawo']:.3f}")

plt.tight_layout()
action_probs_combined_path = os.path.join(action_probs_dir, 'action_probs_combined.png')
plt.savefig(action_probs_combined_path, dpi=150)
plt.close()
print(f'Wykres prawdopodobieństw akcji zapisany: {action_probs_combined_path}')

# Zapisz prawdopodobieństwa akcji do CSV
action_csv_path = os.path.join(action_probs_dir, 'action_probs.csv')
with open(action_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['stan', 'p_lewo', 'p_prosto', 'p_prawo', 'wybrana_akcja'])
    for probs in action_probs_list:
        writer.writerow([
            probs['state'],
            probs['p_lewo'],
            probs['p_prosto'],
            probs['p_prawo'],
            action_names[probs['akcja']]
        ])
print(f'CSV prawdopodobieństw akcji zapisany: {action_csv_path}')

# Zapisz szczegółowe dane aktywacji do CSV
activations_csv_path = os.path.join(output_dir, 'detailed_activations.csv')
with open(activations_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['stan', 'akcja', 'cnn_output_mean', 'scalar_output_mean', 'features_mean',
              'lstm_output_mean', 'lstm_hidden_mean', 'lstm_cell_mean', 'mlp_pi_mean', 'logits_mean']
    writer.writerow(header)
    for d in detailed_activations:
        row = [
            d['state'], 
            action_names[d['action']], 
            d['cnn_output_mean'],
            d['scalar_output_mean'],
            d['features_mean'],
            d['lstm_output_mean'],
            d['lstm_hidden_mean'],
            d['lstm_cell_mean'],
            d['mlp_pi_mean'],
            d['logits_mean']
        ]
        writer.writerow(row)
print(f'Szczegółowe aktywacje zapisane: {activations_csv_path}')

env.close()

# ===================================================
# PODSUMOWANIE KOŃCOWE
# ===================================================
print("\n" + "="*80)
print("=== ANALIZA ZAKOŃCZONA ===")
print("="*80)
print(f"\n📂 Struktura katalogów:")
print(f"   {output_dir}/")
print(f"   ├── bottleneck_analysis.png          ⚠️  Analiza bottlenecków")
print(f"   ├── neuron_activations_overview.png")
print(f"   ├── bottleneck_report.csv            📊 Raport bottlenecków")
print(f"   ├── detailed_activations.csv")
print(f"   │")
print(f"   ├── 🔥 attention_heatmaps/")
print(f"   │   ├── attention_heatmap_state_0.png")
print(f"   │   ├── attention_heatmap_state_1.png")
print(f"   │   └── attention_heatmap_state_2.png")
print(f"   │")
print(f"   ├── 🧠 lstm_analysis/")
print(f"   │   ├── lstm_memory_evolution.png")
print(f"   │   ├── lstm_neurons_heatmap.png")
print(f"   │   └── lstm_memory_data.csv")
print(f"   │")
print(f"   ├── 🎲 uncertainty_analysis/")
print(f"   │   ├── uncertainty_analysis.png")
print(f"   │   └── uncertainty_data.csv")
print(f"   │")
print(f"   ├── 📊 confusion_matrix/")
print(f"   │   ├── confusion_matrix.png")
print(f"   │   └── confusion_matrix.csv")
print(f"   │")
print(f"   ├── conv_visualizations/")
print(f"   │   ├── cnn_output_state_0.png")
print(f"   │   ├── cnn_output_state_1.png")
print(f"   │   └── cnn_output_state_2.png")
print(f"   │")
print(f"   ├── viewports/")
print(f"   │   ├── viewport_state_0.png")
print(f"   │   ├── viewport_state_1.png")
print(f"   │   └── viewport_state_2.png")
print(f"   │")
print(f"   └── action_probs/")
print(f"       ├── action_probs_combined.png")
print(f"       └── action_probs.csv")

print("\n" + "="*80)
print("=== KLUCZOWE WYNIKI ===")
print("="*80)

print("\n🔥 ATTENTION HEATMAPS:")
print("   - Pokazują które regiony viewport są najważniejsze dla decyzji")
print("   - Czerwone obszary = wysoka uwaga modelu")
print("   - Sprawdź czy model patrzy na jedzenie, ściany, czy własne ciało")

print("\n🧠 LSTM MEMORY ANALYSIS:")
print("   - lstm_memory_evolution.png: jak zmienia się pamięć w czasie")
print("   - lstm_neurons_heatmap.png: aktywacja wszystkich neuronów LSTM")
print("   - Sprawdź czy LSTM faktycznie wykorzystuje pamięć długoterminową")

print("\n🎲 UNCERTAINTY ANALYSIS:")
print(f"   - Średnia entropia: {np.mean(entropies):.3f}")
print(f"   - Średnia pewność: {np.mean(max_probs):.3f}")
print(f"   - Decyzje wysokiej pewności: {len(high_cert)/len(extended_uncertainty)*100:.1f}%")
print("   - Sprawdź czy model jest pewny swoich decyzji")

print("\n📊 CONFUSION MATRIX:")
print(f"   - Zgodność z heurystyką: {accuracy*100:.1f}%")
print(f"   - Pokazuje jakie błędy model popełnia najczęściej")
print("   - Porównuje decyzje modelu z prostą logiką ekspercką")

print("\n⚠️  BOTTLENECKS:")
if high_severity:
    print(f"   - 🔴 WYSOKIE RYZYKO: {len(high_severity)} przypadków")
if medium_severity:
    print(f"   - 🟠 ŚREDNIE RYZYKO: {len(medium_severity)} przypadków")
if not high_severity and not medium_severity:
    print("   - ✅ Brak krytycznych bottlenecków")

print("\n" + "="*80)
print("🎯 REKOMENDACJE:")
print("="*80)
print("1. Sprawdź attention heatmaps - czy model patrzy na właściwe rzeczy?")
print("2. Przeanalizuj LSTM memory evolution - czy pamięć jest wykorzystywana?")
print("3. Zbadaj uncertainty - wysokie wartości = model się waha")
print("4. Confusion matrix - jakie błędy popełnia model najczęściej?")
print("5. Jeśli są bottlenecki - rozważ:")
print("   - Zwiększenie learning rate")
print("   - Gradient clipping")
print("   - Batch normalization")
print("   - Residual connections")
print("\n" + "="*80)