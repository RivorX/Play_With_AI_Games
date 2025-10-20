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
    🆕 FIXED: Simplified Pre-LN (no skip connections, 2 CNN layers, GELU)
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
            
            # 🆕 CNN FEATURES - Simplified Pre-LN (NO SKIP, 2 LAYERS)
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # Input BN
            if hasattr(features_extractor, 'input_bn'):
                x = features_extractor.input_bn(image)
            else:
                x = image
            
            # Layer 1
            x = features_extractor.conv1(x)
            x = features_extractor.bn1(x)
            x = torch.nn.functional.gelu(x)  # 🆕 GELU
            
            # Layer 2 with Pre-Norm (NO SKIP)
            if hasattr(features_extractor, 'pre_bn2'):
                x = features_extractor.pre_bn2(x)
            x = features_extractor.conv2(x)
            x = features_extractor.bn2(x)
            x = torch.nn.functional.gelu(x)  # 🆕 GELU
            x = features_extractor.dropout2(x)
            
            # No Conv3 - REMOVED
            
            cnn_features = features_extractor.flatten(x)
            cnn_features = cnn_features.float()
            
            # Pre-LN norm
            if hasattr(features_extractor, 'cnn_pre_norm'):
                cnn_features = features_extractor.cnn_pre_norm(cnn_features)
            
            state_activations['cnn_output_mean'] = cnn_features.abs().mean().item()
            state_activations['cnn_output_max'] = cnn_features.abs().max().item()
            state_activations['cnn_output_std'] = cnn_features.std().item()
            
            # SCALAR FEATURES (unchanged)
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
            
            if hasattr(features_extractor, 'scalar_pre_norm'):
                scalar_features = features_extractor.scalar_pre_norm(scalar_features)
            
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
    
    return action_probs_list, detailed_activations, layer_gradients, attention_heatmaps


def generate_attention_heatmap(model, obs, obs_tensor, lstm_states, action_idx, output_dir, state_idx, action_names):
    """
    Generuje attention heatmap używając gradientów
    🆕 FIXED: Simplified Pre-LN (no skip, 2 layers, GELU)
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
    
    # 🆕 Simplified Pre-LN CNN (NO SKIP)
    if hasattr(features_extractor, 'input_bn'):
        x = features_extractor.input_bn(image_grad)
    else:
        x = image_grad
    
    # Layer 1
    x = features_extractor.conv1(x)
    x = features_extractor.bn1(x)
    x = torch.nn.functional.gelu(x)
    
    # Layer 2 (NO SKIP)
    if hasattr(features_extractor, 'pre_bn2'):
        x = features_extractor.pre_bn2(x)
    x = features_extractor.conv2(x)
    x = features_extractor.bn2(x)
    x = torch.nn.functional.gelu(x)
    x = features_extractor.dropout2(x)
    
    # No Conv3
    
    cnn_output = features_extractor.flatten(x)
    cnn_output = cnn_output.float()
    
    if hasattr(features_extractor, 'cnn_pre_norm'):
        cnn_output = features_extractor.cnn_pre_norm(cnn_output)
    
    scalars_grad = torch.cat([
        obs_grad['direction'],
        obs_grad['dx_head'],
        obs_grad['dy_head'],
        obs_grad['front_coll'],
        obs_grad['left_coll'],
        obs_grad['right_coll']
    ], dim=-1)
    
    scalars_grad = features_extractor.scalar_input_dropout(scalars_grad)
    scalar_output = features_extractor.scalar_linear(scalars_grad)
    
    if hasattr(features_extractor, 'scalar_pre_norm'):
        scalar_output = features_extractor.scalar_pre_norm(scalar_output)
    
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
    🆕 FIXED: Simplified Pre-LN (no skip, 2 layers, GELU)
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
    
    # 🆕 CNN layers (SIMPLIFIED - NO SKIP, 2 LAYERS)
    cnn_intermediates = []
    x_cnn = image_grad
    
    if hasattr(features_extractor, 'input_bn'):
        x_cnn = features_extractor.input_bn(x_cnn)
        x_cnn.retain_grad()
        cnn_intermediates.append(('cnn', 0, 'InputBN', x_cnn))
    
    # Layer 1
    x_cnn = features_extractor.conv1(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 1, 'Conv2d-1', x_cnn))
    x_cnn = features_extractor.bn1(x_cnn)
    x_cnn = torch.nn.functional.gelu(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 2, 'GELU-1', x_cnn))
    
    # Layer 2 (NO SKIP)
    if hasattr(features_extractor, 'pre_bn2'):
        x_cnn = features_extractor.pre_bn2(x_cnn)
        x_cnn.retain_grad()
        cnn_intermediates.append(('cnn', 3, 'PreBN2', x_cnn))
    
    x_cnn = features_extractor.conv2(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 4, 'Conv2d-2', x_cnn))
    x_cnn = features_extractor.bn2(x_cnn)
    x_cnn = torch.nn.functional.gelu(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 5, 'GELU-2', x_cnn))
    x_cnn = features_extractor.dropout2(x_cnn)
    
    # No Conv3
    
    cnn_output = features_extractor.flatten(x_cnn)
    cnn_output = cnn_output.float()
    
    if hasattr(features_extractor, 'cnn_pre_norm'):
        cnn_output = features_extractor.cnn_pre_norm(cnn_output)
    
    cnn_output.retain_grad()
    cnn_intermediates.append(('cnn', len(cnn_intermediates), 'Flatten+Norm', cnn_output))
    
    # Scalar layers (unchanged)
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
        if isinstance(layer, (torch.nn.Linear, torch.nn.GELU, torch.nn.LayerNorm)):  # 🆕 GELU
            x_scalar.retain_grad()
            scalar_intermediates.append(('scalar', i, layer.__class__.__name__, x_scalar))
    
    if hasattr(features_extractor, 'scalar_pre_norm'):
        scalar_output = features_extractor.scalar_pre_norm(x_scalar)
    
    scalar_output.retain_grad()
    scalar_intermediates.append(('scalar', len(scalar_intermediates), 'Norm', scalar_output))
    
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
    """Wizualizuje output CNN (pierwsze 16 kanałów)"""
    with torch.no_grad():
        image = obs_tensor['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        cnn_activation = features_extractor.conv1(image).detach().cpu().numpy()[0]
    
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
    cnn_viz_path = os.path.join(output_dir, f'cnn_output_state_{state_idx}.png')
    plt.savefig(cnn_viz_path, dpi=150)
    plt.close()
    print(f'  Wizualizacja CNN zapisana: {cnn_viz_path}')


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
    """Analiza bottlenecków w sieci"""
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