import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter


def analyze_basic_states(model, env, output_dirs, action_names, config):
    """
    Analiza podstawowych stanów: aktywacje, attention heatmaps, gradienty
    🆕 UPDATED: Pure Bottleneck (NO Skip)
    """
    policy = model.policy
    features_extractor = policy.features_extractor
    
    action_probs_list = []
    detailed_activations = []
    layer_gradients = []
    attention_heatmaps = []
    
    # Inicjalizuj stany LSTM
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    # Analiza 3 stanów
    for state_idx in range(3):
        print(f"\nAnaliza stanu {state_idx}...")
        
        # Pobierz obserwację
        obs, _ = env.reset()
        lstm_states = None
        
        # Wykonaj kilka kroków
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
            
            # 🆕 CNN FEATURES - Moderate Compression + Residual
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # CNN layers (bez input_bn)
            x = image
            
            # Layer 1
            x = features_extractor.conv1(x)
            x = features_extractor.bn1(x)
            x = torch.nn.functional.gelu(x)
            
            # Layer 2
            x = features_extractor.conv2(x)
            x = features_extractor.bn2(x)
            x = torch.nn.functional.gelu(x)
            x = features_extractor.dropout2(x)
            
            # Flatten
            cnn_raw = features_extractor.flatten(x)
            cnn_raw = cnn_raw.float()
            
            # 🆕 PURE BOTTLENECK (NO SKIP)
            cnn_features = features_extractor.cnn_bottleneck(cnn_raw)
            
            state_activations['cnn_raw_mean'] = cnn_raw.abs().mean().item()
            state_activations['cnn_raw_max'] = cnn_raw.abs().max().item()
            state_activations['cnn_bottleneck_mean'] = cnn_features.abs().mean().item()
            state_activations['cnn_output_mean'] = cnn_features.abs().mean().item()
            state_activations['cnn_output_max'] = cnn_features.abs().max().item()
            state_activations['cnn_output_std'] = cnn_features.std().item()
            
            # SCALAR FEATURES
            scalars = torch.cat([
                obs_tensor['direction'],
                obs_tensor['dx_head'],
                obs_tensor['dy_head'],
                obs_tensor['front_coll'],
                obs_tensor['left_coll'],
                obs_tensor['right_coll']
            ], dim=-1)
            
            scalars = features_extractor.scalar_input_dropout(scalars)
            scalar_features = features_extractor.scalar_linear(scalars)
            
            state_activations['scalar_output_mean'] = scalar_features.abs().mean().item()
            state_activations['scalar_output_max'] = scalar_features.abs().max().item()
            state_activations['scalar_output_std'] = scalar_features.std().item()
            
            # COMBINED FEATURES
            features = features_extractor(obs_tensor)
            state_activations['features_mean'] = features.abs().mean().item()
            state_activations['features_max'] = features.abs().max().item()
            state_activations['features_std'] = features.std().item()
            
            # LSTM ACTOR
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
            
            # MLP EXTRACTOR
            latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
            state_activations['mlp_pi_mean'] = latent_pi_mlp.abs().mean().item()
            state_activations['mlp_pi_max'] = latent_pi_mlp.abs().max().item()
            
            # ACTION NETWORK
            logits = policy.action_net(latent_pi_mlp)
            state_activations['logits_mean'] = logits.abs().mean().item()
            state_activations['logits_max'] = logits.abs().max().item()
            
            detailed_activations.append(state_activations)

        # Prawdopodobieństwa akcji
        action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        action_probs_list.append({
            'state': state_idx,
            'p_lewo': action_probs[0],
            'p_prosto': action_probs[1],
            'p_prawo': action_probs[2],
            'akcja': action_idx
        })
        
        # === ATTENTION HEATMAP ===
        print(f"  Generowanie attention heatmap...")
        attention_map = generate_attention_heatmap(
            model, obs, obs_tensor, lstm_states, action_idx,
            output_dirs['heatmap'], state_idx, action_names
        )
        if attention_map is not None:
            attention_heatmaps.append({
                'state': state_idx,
                'action': action_idx,
                'heatmap': attention_map,
                'viewport': obs['image'][:, :, 0]
            })
        
        # === ANALIZA GRADIENTÓW WARSTW ===
        print(f"  Obliczanie gradientów warstw...")
        state_layer_grads = compute_layer_gradients(
            model, obs, obs_tensor, lstm_states, action_idx, features_extractor
        )
        layer_gradients.append(state_layer_grads)
        
        # === WIZUALIZACJA CNN OUTPUT ===
        visualize_cnn_output(obs_tensor, features_extractor, output_dirs['conv_viz'], state_idx)
        
        # === WIZUALIZACJA VIEWPORT ===
        visualize_viewport(obs, output_dirs['viewport'], state_idx)
        
        print(f"  Wybrany action: {action_names[action_idx]}")
        print(f"  Prawdopodobieństwa: lewo={action_probs[0]:.3f}, prosto={action_probs[1]:.3f}, prawo={action_probs[2]:.3f}")
        print(f"  🆕 CNN raw: {state_activations['cnn_raw_mean']:.4f}, bottleneck: {state_activations['cnn_bottleneck_mean']:.4f}, output: {state_activations['cnn_output_mean']:.4f}")
    
    return action_probs_list, detailed_activations, layer_gradients, attention_heatmaps


def generate_attention_heatmap(model, obs, obs_tensor, lstm_states, action_idx, output_dir, state_idx, action_names):
    """
    Generuje attention heatmap używając gradientów
    🆕 UPDATED: Pure Bottleneck (NO Skip)
    """
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # Ustaw tryby
    features_extractor.eval()
    policy.lstm_actor.train()
    policy.mlp_extractor.train()
    policy.action_net.train()
    
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
    
    # Forward pass
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    image_grad.retain_grad()
    
    # 🆕 CNN (bez input_bn)
    x = image_grad
    
    # Layer 1
    x = features_extractor.conv1(x)
    x = features_extractor.bn1(x)
    x = torch.nn.functional.gelu(x)
    
    # Layer 2
    x = features_extractor.conv2(x)
    x = features_extractor.bn2(x)
    x = torch.nn.functional.gelu(x)
    x = features_extractor.dropout2(x)
    
    # Flatten
    cnn_raw = features_extractor.flatten(x)
    cnn_raw = cnn_raw.float()
    
    # 🆕 PURE BOTTLENECK (NO SKIP)
    cnn_features = features_extractor.cnn_bottleneck(cnn_raw)
    
    # Scalars
    scalars_grad = torch.cat([
        obs_grad['direction'],
        obs_grad['dx_head'],
        obs_grad['dy_head'],
        obs_grad['front_coll'],
        obs_grad['left_coll'],
        obs_grad['right_coll']
    ], dim=-1)
    
    scalars_grad = features_extractor.scalar_input_dropout(scalars_grad)
    scalar_features = features_extractor.scalar_linear(scalars_grad)
    
    # Fusion
    combined = torch.cat([cnn_features, scalar_features], dim=-1)
    features_final = features_extractor.final_linear(combined)
    
    # LSTM
    lstm_states_tensor = (
        torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device, requires_grad=False),
        torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device, requires_grad=False)
    )
    
    features_seq = features_final.unsqueeze(1)
    lstm_out, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
    latent_pi_grad = lstm_out.squeeze(1)
    
    latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi_grad)
    logits_grad = policy.action_net(latent_pi_mlp)
    selected_logit = logits_grad[0, action_idx]
    
    # Backward
    model.policy.zero_grad()
    selected_logit.backward()
    
    # Gradient względem input image
    if image_grad.grad is not None:
        grad_map = image_grad.grad[0, 0].abs().detach().cpu().numpy()
        
        if grad_map.max() > 0:
            grad_map = grad_map / grad_map.max()
        
        grad_map_smooth = gaussian_filter(grad_map, sigma=0.8)
        
        # Wizualizacja
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(obs['image'][:, :, 0], cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Viewport (Stan {state_idx})')
        axes[0].axis('off')
        
        im1 = axes[1].imshow(grad_map_smooth, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f'Attention Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        axes[2].imshow(obs['image'][:, :, 0], cmap='viridis', interpolation='nearest', alpha=0.7)
        axes[2].imshow(grad_map_smooth, cmap='hot', interpolation='bilinear', alpha=0.5)
        axes[2].set_title(f'Overlay (Akcja: {action_names[action_idx]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'attention_heatmap_state_{state_idx}.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Attention heatmap zapisana: {heatmap_path}")
        
        model.policy.eval()
        return grad_map_smooth
    
    model.policy.eval()
    return None


def compute_layer_gradients(model, obs, obs_tensor, lstm_states, action_idx, features_extractor):
    """
    Oblicza gradienty dla wszystkich warstw
    🆕 UPDATED: Pure Bottleneck (NO Skip)
    """
    policy = model.policy
    
    features_extractor.eval()
    policy.lstm_actor.train()
    
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
    
    # Forward z intermediate activations
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    # 🆕 CNN layers (bez input_bn)
    cnn_intermediates = []
    x_cnn = image_grad
    
    # Layer 1
    x_cnn = features_extractor.conv1(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 0, 'Conv2d-1', x_cnn))
    x_cnn = features_extractor.bn1(x_cnn)
    x_cnn = torch.nn.functional.gelu(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 1, 'GELU-1', x_cnn))
    
    # Layer 2
    x_cnn = features_extractor.conv2(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 2, 'Conv2d-2', x_cnn))
    x_cnn = features_extractor.bn2(x_cnn)
    x_cnn = torch.nn.functional.gelu(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 3, 'GELU-2', x_cnn))
    x_cnn = features_extractor.dropout2(x_cnn)
    
    # Flatten
    cnn_raw = features_extractor.flatten(x_cnn)
    cnn_raw = cnn_raw.float()
    cnn_raw.retain_grad()
    cnn_intermediates.append(('cnn', 4, 'Flatten', cnn_raw))
    
    # 🆕 PURE BOTTLENECK (NO SKIP)
    cnn_features = features_extractor.cnn_bottleneck(cnn_raw)
    cnn_features.retain_grad()
    cnn_intermediates.append(('cnn', 5, 'Bottleneck', cnn_features))
    
    # Scalar layers
    scalars_grad = torch.cat([
        obs_grad['direction'],
        obs_grad['dx_head'],
        obs_grad['dy_head'],
        obs_grad['front_coll'],
        obs_grad['left_coll'],
        obs_grad['right_coll']
    ], dim=-1)
    
    scalars_grad = features_extractor.scalar_input_dropout(scalars_grad)
    
    scalar_intermediates = []
    x_scalar = scalars_grad
    for i, layer in enumerate(features_extractor.scalar_linear):
        x_scalar = layer(x_scalar)
        if isinstance(layer, (torch.nn.Linear, torch.nn.GELU, torch.nn.LayerNorm)):
            x_scalar.retain_grad()
            scalar_intermediates.append(('scalar', i, layer.__class__.__name__, x_scalar))
    
    scalar_features = x_scalar
    scalar_features.retain_grad()
    scalar_intermediates.append(('scalar', len(scalar_intermediates), 'Output', scalar_features))
    
    # Combined features
    combined = torch.cat([cnn_features, scalar_features], dim=-1)
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
    
    # Zbierz gradienty
    state_layer_grads = {
        'state': obs_grad.get('state', 0),
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
    
    model.policy.eval()
    return state_layer_grads


def visualize_cnn_output(obs_tensor, features_extractor, output_dir, state_idx):
    """
    Wizualizuje output wszystkich warstw CNN
    🆕 UPDATED: Pure Bottleneck (NO Skip)
    """
    with torch.no_grad():
        image = obs_tensor['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        # ==================== WSZYSTKIE WARSTWY CNN ====================
        activations = {}
        
        # Warstwa 1: Conv1 + BN + GELU
        x = features_extractor.conv1(image)
        x = features_extractor.bn1(x)
        x = torch.nn.functional.gelu(x)
        activations['conv1_output'] = x.detach().cpu().numpy()[0]  # [32, H, W]
        
        # Warstwa 2: Conv2 + BN + GELU + Dropout
        x = features_extractor.conv2(x)
        x = features_extractor.bn2(x)
        x = torch.nn.functional.gelu(x)
        x = features_extractor.dropout2(x)
        activations['conv2_output'] = x.detach().cpu().numpy()[0]  # [64, H/2, W/2]
        
        # Flatten
        cnn_raw = features_extractor.flatten(x)
        cnn_raw = cnn_raw.float()
        activations['cnn_raw'] = cnn_raw.detach().cpu().numpy()[0]  # [4608]
        
        # Pure Bottleneck
        cnn_features = features_extractor.cnn_bottleneck(cnn_raw)
        activations['bottleneck'] = cnn_features.detach().cpu().numpy()[0]  # [output_dim]
    
    # ==================== WIZUALIZACJA ====================
    
    # 1. CONV1 OUTPUT
    visualize_conv_layer(
        activations['conv1_output'], 
        layer_name='Conv1 Output',
        output_path=os.path.join(output_dir, f'cnn_conv1_state_{state_idx}.png'),
        num_channels=min(16, activations['conv1_output'].shape[0])
    )
    
    # 2. CONV2 OUTPUT
    visualize_conv_layer(
        activations['conv2_output'], 
        layer_name='Conv2 Output',
        output_path=os.path.join(output_dir, f'cnn_conv2_state_{state_idx}.png'),
        num_channels=min(16, activations['conv2_output'].shape[0])
    )
    
    # 3. RAW vs BOTTLENECK (1D features)
    visualize_1d_features(
        {
            f'CNN Raw ({len(activations["cnn_raw"])})': activations['cnn_raw'],
            f'Bottleneck ({len(activations["bottleneck"])})': activations['bottleneck']
        },
        output_path=os.path.join(output_dir, f'cnn_bottleneck_state_{state_idx}.png'),
        state_idx=state_idx
    )
    
    # 4. HEATMAPA WSZYSTKICH KANAŁÓW CONV2
    visualize_all_channels_heatmap(
        activations['conv2_output'],
        layer_name='Conv2 All Channels',
        output_path=os.path.join(output_dir, f'cnn_conv2_all_channels_state_{state_idx}.png')
    )
    
    print(f'  ✅ CNN visualization (all layers) zapisana dla stanu {state_idx}')


def visualize_conv_layer(activation, layer_name, output_path, num_channels=16):
    """Wizualizuje wybrane kanały warstwy konwolucyjnej (grid 4x4)"""
    num_channels = min(num_channels, activation.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_channels):
        ax = axes[i // 4, i % 4]
        im = ax.imshow(activation[i], cmap='viridis', interpolation='nearest')
        ax.set_title(f'Ch{i}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Wyłącz puste subploty
    for i in range(num_channels, 16):
        axes[i // 4, i % 4].axis('off')
    
    plt.suptitle(f'{layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_1d_features(features_dict, output_path, state_idx):
    """
    Wizualizuje 1D features (bottleneck, residual, final output)
    jako heatmapy poziome
    """
    fig, axes = plt.subplots(len(features_dict), 1, figsize=(16, 6))
    
    if len(features_dict) == 1:
        axes = [axes]
    
    for idx, (name, features) in enumerate(features_dict.items()):
        ax = axes[idx]
        
        # Normalizacja dla lepszej wizualizacji
        features_norm = features / (np.abs(features).max() + 1e-8)
        
        # Heatmapa pozioma
        im = ax.imshow(features_norm.reshape(1, -1), 
                       cmap='coolwarm', 
                       aspect='auto', 
                       interpolation='nearest',
                       vmin=-1, vmax=1)
        
        ax.set_title(f'{name} - Stan {state_idx}', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel('Neuron Index', fontsize=10)
        plt.colorbar(im, ax=ax, label='Normalized Activation')
        
        # Statystyki
        stats_text = f'Mean: {features.mean():.3f} | Std: {features.std():.3f} | Max: {features.max():.3f} | Min: {features.min():.3f}'
        ax.text(0.5, -0.3, stats_text, ha='center', transform=ax.transAxes, fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_all_channels_heatmap(activation, layer_name, output_path):
    """
    Wizualizuje WSZYSTKIE kanały jako heatmapę (channels x spatial)
    Użyteczne dla Conv2: 64 channels x 8x8 = 64 rows x 64 cols
    """
    num_channels, height, width = activation.shape
    
    # Flatten spatial dimensions: [64, 8, 8] → [64, 64]
    activation_flat = activation.reshape(num_channels, -1)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(activation_flat, cmap='viridis', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('Spatial Position (flattened)', fontsize=12)
    ax.set_ylabel('Channel Index', fontsize=12)
    ax.set_title(f'{layer_name} - All Channels Heatmap', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Activation Value')
    
    # Dodaj linie co 8 pikseli (dla 8x8 spatial)
    if width == 8:
        for i in range(1, height):
            ax.axvline(i * width - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_viewport(obs, output_dir, state_idx):
    """Wizualizuje viewport 16x16"""
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
    viewport_path = os.path.join(output_dir, f'viewport_state_{state_idx}.png')
    plt.savefig(viewport_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_bottlenecks(layer_gradients, action_names, output_dir):
    """
    Analiza bottlenecków w sieci
    🆕 UPDATED: Dodano etykiety dla residual connections
    """
    print("\n=== Analiza bottlenecków (aktywacje vs gradienty) ===")
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    
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
        print("\n🟡 ŚREDNIE RYZYKO BOTTLENECKÓW (Dead/Underutilized Neurons):")
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
    
    return bottleneck_report

def analyze_channel_specialization(model, env, output_dir, num_samples=50):
    """
    Analiza specjalizacji kanałów CNN
    - Które kanały są aktywne?
    - Które kanały są "dead" (zawsze ~0)?
    - Jaka jest różnorodność między kanałami?
    """
    print("\n=== Analiza specjalizacji kanałów CNN ===")
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # Zbierz aktywacje z wielu stanów
    conv1_activations = []
    conv2_activations = []
    
    for _ in range(num_samples):
        obs, _ = env.reset()
        
        # Random steps
        for _ in range(np.random.randint(0, 20)):
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
        
        # Get activations
        with torch.no_grad():
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
            
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # Conv1
            x = features_extractor.conv1(image)
            x = features_extractor.bn1(x)
            x = torch.nn.functional.gelu(x)
            conv1_activations.append(x.detach().cpu().numpy()[0])  # [32, 16, 16]
            
            # Conv2
            x = features_extractor.conv2(x)
            x = features_extractor.bn2(x)
            x = torch.nn.functional.gelu(x)
            conv2_activations.append(x.detach().cpu().numpy()[0])  # [64, 8, 8]
    
    # Stack: [num_samples, channels, height, width]
    conv1_activations = np.array(conv1_activations)  # [50, 32, 16, 16]
    conv2_activations = np.array(conv2_activations)  # [50, 64, 8, 8]
    
    # Analiza per channel
    def analyze_channels(activations, layer_name):
        num_channels = activations.shape[1]
        channel_stats = []
        
        for ch in range(num_channels):
            ch_data = activations[:, ch, :, :].flatten()
            
            mean_activation = ch_data.mean()
            std_activation = ch_data.std()
            max_activation = ch_data.max()
            sparsity = (np.abs(ch_data) < 0.01).sum() / len(ch_data)
            
            # Dead neuron detection
            is_dead = (max_activation < 0.01) or (std_activation < 0.001)
            
            channel_stats.append({
                'channel': ch,
                'mean': mean_activation,
                'std': std_activation,
                'max': max_activation,
                'sparsity': sparsity,
                'is_dead': is_dead
            })
        
        return channel_stats
    
    conv1_stats = analyze_channels(conv1_activations, 'Conv1')
    conv2_stats = analyze_channels(conv2_activations, 'Conv2')
    
    # Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Conv1 - Mean activation per channel
    channels_conv1 = [s['channel'] for s in conv1_stats]
    means_conv1 = [s['mean'] for s in conv1_stats]
    dead_conv1 = [s['is_dead'] for s in conv1_stats]
    
    colors_conv1 = ['red' if d else 'green' for d in dead_conv1]
    axes[0, 0].bar(channels_conv1, means_conv1, color=colors_conv1, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean Activation')
    axes[0, 0].set_title(f'Conv1 - Channel Activity (Red = Dead)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Conv2 - Mean activation per channel
    channels_conv2 = [s['channel'] for s in conv2_stats]
    means_conv2 = [s['mean'] for s in conv2_stats]
    dead_conv2 = [s['is_dead'] for s in conv2_stats]
    
    colors_conv2 = ['red' if d else 'green' for d in dead_conv2]
    axes[0, 1].bar(channels_conv2, means_conv2, color=colors_conv2, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Mean Activation')
    axes[0, 1].set_title(f'Conv2 - Channel Activity (Red = Dead)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Sparsity (ile neuronów ~0)
    sparsity_conv1 = [s['sparsity'] for s in conv1_stats]
    axes[1, 0].bar(channels_conv1, sparsity_conv1, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Sparsity (% near-zero)')
    axes[1, 0].set_title('Conv1 - Sparsity per Channel')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Sparsity Conv2
    sparsity_conv2 = [s['sparsity'] for s in conv2_stats]
    axes[1, 1].bar(channels_conv2, sparsity_conv2, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Sparsity (% near-zero)')
    axes[1, 1].set_title('Conv2 - Sparsity per Channel')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    specialization_path = os.path.join(output_dir, 'channel_specialization.png')
    plt.savefig(specialization_path, dpi=150)
    plt.close()
    
    # Raport
    dead_conv1_count = sum(dead_conv1)
    dead_conv2_count = sum(dead_conv2)
    
    print(f"\n📊 Channel Specialization Report:")
    print(f"   Conv1: {dead_conv1_count}/32 dead channels ({dead_conv1_count/32*100:.1f}%)")
    print(f"   Conv2: {dead_conv2_count}/64 dead channels ({dead_conv2_count/64*100:.1f}%)")
    print(f"\n   âš ï¸ Jeśli >30% kanałów jest dead, sieć jest UNDERUTILIZED!")
    
    print(f"\n✅ Channel specialization zapisana: {specialization_path}")
    
def plot_activation_overview(detailed_activations, action_probs_list, action_names, output_dirs):
    """Generuje wykresy przeglądu aktywacji"""
    
    # Wykres 1: Przegląd aktywacji neuronów
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
    neuron_activation_path = os.path.join(output_dirs['main'], 'neuron_activations_overview.png')
    plt.savefig(neuron_activation_path, dpi=150)
    plt.close()
    print(f'Wykres aktywacji neuronów zapisany: {neuron_activation_path}')
    
    # Wykres 2: Prawdopodobieństwa akcji
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
    action_probs_combined_path = os.path.join(output_dirs['action_probs'], 'action_probs_combined.png')
    plt.savefig(action_probs_combined_path, dpi=150)
    plt.close()
    
    # Zapisz prawdopodobieństwa akcji do CSV
    action_csv_path = os.path.join(output_dirs['action_probs'], 'action_probs.csv')
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
    
    # Zapisz szczegółowe dane aktywacji do CSV
    activations_csv_path = os.path.join(output_dirs['main'], 'detailed_activations.csv')
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