"""
🔍 BASIC ANALYSIS MODULE
- analyze_basic_states: Analiza podstawowych stanów, aktywacje, attention heatmaps
- visualize_cnn_output: Wizualizacja warstw CNN (Conv1, Conv2, Conv3)
- visualize_viewport: Wizualizacja viewport 16x16
- generate_attention_heatmap: Generowanie attention heatmap
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
        obs, _ = env.reset()
        
        # Konwersja obs do tensora
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.tensor(value, dtype=torch.float32, device=policy.device).unsqueeze(0)
            else:
                obs_tensor[key] = torch.tensor([value], dtype=torch.float32, device=policy.device).unsqueeze(0)
        
        episode_starts_tensor = torch.tensor(episode_starts, dtype=torch.bool, device=policy.device)
        
        # Forward pass (bez gradów)
        with torch.no_grad():
            # CNN features
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # CNN - Multi-Scale Architecture
            x = image
            x = features_extractor.conv1(x)
            x = features_extractor.bn1(x)
            x = torch.nn.functional.gelu(x)
            
            x = features_extractor.conv2(x)
            x = features_extractor.bn2(x)
            x = features_extractor.dropout2(x)
            x = torch.nn.functional.gelu(x)
            
            if features_extractor.has_conv3:
                identity = features_extractor.residual_proj(x)
                
                x_local = features_extractor.conv3_local(x)
                x_local = features_extractor.bn3_local(x_local)
                
                x_global = features_extractor.conv3_global(x)
                x_global = features_extractor.bn3_global(x_global)
                
                x_combined = torch.cat([x_local, x_global], dim=1)
                x_combined = features_extractor.dropout3(x_combined)
                x_combined = torch.nn.functional.gelu(x_combined)
                x = x_combined + identity  # Residual
            
            cnn_raw = features_extractor.flatten(x)
            cnn_raw = cnn_raw.float()
            
            # Bottleneck
            cnn_features = features_extractor.cnn_bottleneck(cnn_raw)
            
            # Scalars
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
            
            # Fusion
            combined = torch.cat([cnn_features, scalar_features], dim=-1)
            features_final = features_extractor.fusion_main(combined)
            
            # LSTM - inicjalizacja stanów jeśli potrzeba
            if lstm_states is None:
                # Standard PyTorch LSTM: (num_layers, batch_size, hidden_size)
                batch_size = features_final.shape[0]
                num_layers = policy.lstm_actor.num_layers
                hidden_size = policy.lstm_actor.hidden_size
                
                lstm_states = (
                    np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32),
                    np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
                )
            
            lstm_states_tensor = (
                torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device),
                torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device)
            )
            
            features_seq = features_final.unsqueeze(1)
            lstm_out, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_tensor)
            lstm_states = (new_lstm_states[0].cpu().numpy(), new_lstm_states[1].cpu().numpy())
            
            latent_pi = lstm_out.squeeze(1)
            latent_vf = lstm_out.squeeze(1)
            
            # MLP
            latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
            latent_vf_mlp = policy.mlp_extractor.value_net(latent_vf)
            
            # Action distribution
            action_logits = policy.action_net(latent_pi_mlp)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1)
            
            # Value
            value = policy.value_net(latent_vf_mlp)
        
        # Zapisz dane
        action_probs_np = action_probs[0].cpu().numpy()
        action_probs_list.append({
            'state': state_idx,
            'probs': action_probs_np,
            'action': action.item(),
            'value': value.item()
        })
        
        detailed_activations.append({
            'state': state_idx,
            'cnn_output_mean': cnn_features[0].cpu().numpy().mean(),
            'cnn_output_std': cnn_features[0].cpu().numpy().std(),
            'scalar_output_mean': scalar_features[0].cpu().numpy().mean(),
            'scalar_output_std': scalar_features[0].cpu().numpy().std(),
            'features_mean': features_final[0].cpu().numpy().mean(),
            'features_std': features_final[0].cpu().numpy().std(),
            'lstm_output_mean': latent_pi[0].cpu().numpy().mean(),
            'lstm_output_std': latent_pi[0].cpu().numpy().std(),
            'lstm_hidden_mean': lstm_states[0][0].mean(),
            'lstm_hidden_std': lstm_states[0][0].std(),
            'lstm_cell_mean': lstm_states[1][0].mean(),
            'lstm_cell_std': lstm_states[1][0].std(),
        })
        
        # Wizualizacja CNN output
        visualize_cnn_output(obs_tensor, features_extractor, output_dirs['conv_viz'], state_idx)
        
        # Wizualizacja viewport
        visualize_viewport(obs, output_dirs['viewport'], state_idx)
        
        # Compute layer gradients
        from utils.analyze_gradients import compute_layer_gradients
        state_layer_grads = compute_layer_gradients(
            model, obs, obs_tensor, lstm_states, action.item(), features_extractor
        )
        layer_gradients.append(state_layer_grads)
        
        # Generate attention heatmap
        attention_map = generate_attention_heatmap(
            model, obs, obs_tensor, lstm_states, action.item(),
            output_dirs['heatmap'], state_idx, action_names
        )
        if attention_map is not None:
            attention_heatmaps.append(attention_map)
        
        print(f"   Stan {state_idx}: Akcja={action_names[action.item()]} (probs={action_probs_np})")
    
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
        if isinstance(v, np.ndarray):
            obs_grad[k] = torch.tensor(v, dtype=torch.float32, device=policy.device, requires_grad=True).unsqueeze(0)
        else:
            obs_grad[k] = torch.tensor([v], dtype=torch.float32, device=policy.device, requires_grad=True).unsqueeze(0)
    
    # Forward pass
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    image_grad.retain_grad()
    
    # 🆕 CNN - Multi-Scale Architecture
    x = image_grad

    # Layer 1
    x = features_extractor.conv1(x)
    x = features_extractor.bn1(x)
    x = torch.nn.functional.gelu(x)

    # Layer 2
    x = features_extractor.conv2(x)
    x = features_extractor.bn2(x)
    x = features_extractor.dropout2(x)
    x = torch.nn.functional.gelu(x)

    # Layer 3 - MULTI-SCALE (opcjonalna)
    if features_extractor.has_conv3:
        identity = features_extractor.residual_proj(x)
        
        x_local = features_extractor.conv3_local(x)
        x_local = features_extractor.bn3_local(x_local)
        
        x_global = features_extractor.conv3_global(x)
        x_global = features_extractor.bn3_global(x_global)
        
        x_combined = torch.cat([x_local, x_global], dim=1)
        x_combined = features_extractor.dropout3(x_combined)
        x_combined = torch.nn.functional.gelu(x_combined)
        x = x_combined + identity  # Residual

    # Flatten
    cnn_raw = features_extractor.flatten(x)
    cnn_raw = cnn_raw.float()

    # 🆕 BOTTLENECK
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
    features_final = features_extractor.fusion_main(combined)
    
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
        grad_map = image_grad.grad[0, 0].cpu().numpy()
        
        # Normalizuj gradient
        grad_map = np.abs(grad_map)
        if grad_map.max() > 0:
            grad_map = grad_map / grad_map.max()
        
        # Wizualizacja
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original viewport
        viewport = obs['image'][:, :, 0]
        axes[0].imshow(viewport, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Original Viewport - Stan {state_idx}')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(grad_map, cmap='hot', interpolation='bilinear', alpha=0.8)
        axes[1].set_title(f'Attention Heatmap - Akcja: {action_names[action_idx]}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Gradient Magnitude')
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'attention_state_{state_idx}.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        
        print(f'  ✅ Attention heatmap zapisana: {heatmap_path}')
    
    model.policy.eval()
    return None


def visualize_cnn_output(obs_tensor, features_extractor, output_dir, state_idx):
    """
    Wizualizuje output wszystkich warstw CNN
    🆕 UPDATED: Multi-Scale Architecture with Local + Global paths
    """
    with torch.no_grad():
        image = obs_tensor['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)

        x = image

        # Conv1
        x = features_extractor.conv1(x)
        x = features_extractor.bn1(x)
        x = torch.nn.functional.gelu(x)
        conv1_output = x[0].cpu().numpy()

        # Conv2
        x = features_extractor.conv2(x)
        x = features_extractor.bn2(x)
        x = features_extractor.dropout2(x)
        x = torch.nn.functional.gelu(x)
        conv2_output = x[0].cpu().numpy()

        activations = {
            'conv1_output': conv1_output,
            'conv2_output': conv2_output
        }

        # Conv3 (if exists)
        if features_extractor.has_conv3:
            identity = features_extractor.residual_proj(x)

            x_local = features_extractor.conv3_local(x)
            x_local = features_extractor.bn3_local(x_local)

            x_global = features_extractor.conv3_global(x)
            x_global = features_extractor.bn3_global(x_global)

            x_combined = torch.cat([x_local, x_global], dim=1)
            x_combined = features_extractor.dropout3(x_combined)
            x_combined = torch.nn.functional.gelu(x_combined)

            x = x_combined + identity  # Residual

            activations['conv3_output'] = x[0].cpu().numpy()
            activations['conv3_local'] = x_local[0].cpu().numpy()
            activations['conv3_global'] = x_global[0].cpu().numpy()

        # Flatten + Bottleneck
        cnn_raw = features_extractor.flatten(x)
        cnn_raw = cnn_raw.float()
        cnn_features = features_extractor.cnn_bottleneck(cnn_raw)

        activations['cnn_raw'] = cnn_raw[0].cpu().numpy()
        activations['bottleneck'] = cnn_features[0].cpu().numpy()

        # Scalars
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
        activations['scalar'] = scalar_features[0].cpu().numpy()

        # Fusion
        combined = torch.cat([cnn_features, scalar_features], dim=-1)
        fusion = features_extractor.fusion_main(combined)
        activations['fusion'] = fusion[0].cpu().numpy()

        # LSTM (jeśli dostępny)
        lstm_out = None
        policy = getattr(features_extractor, 'policy', None)
        if policy is not None and hasattr(policy, 'lstm_actor'):
            lstm_states = (
                torch.zeros((1, policy.lstm_actor.hidden_size), device=cnn_raw.device),
                torch.zeros((1, policy.lstm_actor.hidden_size), device=cnn_raw.device)
            )
            features_seq = fusion.unsqueeze(1)
            lstm_out, _ = policy.lstm_actor(features_seq, lstm_states)
            activations['lstm'] = lstm_out.squeeze(1)[0].cpu().numpy()

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

    # 3. CONV3 - Multi-Scale Branches (jeśli istnieje)
    if 'conv3_output' in activations:
        visualize_conv_layer(
            activations['conv3_output'],
            layer_name='Conv3 Output (Local + Global + Residual)',
            output_path=os.path.join(output_dir, f'cnn_conv3_state_{state_idx}.png'),
            num_channels=min(16, activations['conv3_output'].shape[0])
        )

        # Conv3 Local branch
        if 'conv3_local' in activations:
            visualize_conv_layer(
                activations['conv3_local'],
                layer_name='Conv3 Local Branch',
                output_path=os.path.join(output_dir, f'cnn_conv3_local_state_{state_idx}.png'),
                num_channels=min(16, activations['conv3_local'].shape[0])
            )

        # Conv3 Global branch
        if 'conv3_global' in activations:
            visualize_conv_layer(
                activations['conv3_global'],
                layer_name='Conv3 Global Branch',
                output_path=os.path.join(output_dir, f'cnn_conv3_global_state_{state_idx}.png'),
                num_channels=min(16, activations['conv3_global'].shape[0])
            )

    # 4. RAW vs BOTTLENECK (1D features)
    visualize_1d_features(
        {
            f'CNN Raw ({len(activations["cnn_raw"])})': activations['cnn_raw'],
            f'Bottleneck ({len(activations["bottleneck"])})': activations['bottleneck']
        },
        output_path=os.path.join(output_dir, f'cnn_bottleneck_state_{state_idx}.png'),
        state_idx=state_idx
    )

    # 5. FUSION, SCALAR, LSTM (1D features)
    fusion_dict = {
        f'Scalar ({len(activations["scalar"])})': activations['scalar'],
        f'Fusion ({len(activations["fusion"])})': activations['fusion']
    }
    if 'lstm' in activations:
        fusion_dict[f'LSTM ({len(activations["lstm"])})'] = activations['lstm']
    visualize_1d_features(
        fusion_dict,
        output_path=os.path.join(output_dir, f'fusions_state_{state_idx}.png'),
        state_idx=state_idx
    )

    # 6. HEATMAPA WSZYSTKICH KANAŁÓW CONV2
    visualize_all_channels_heatmap(
        activations['conv2_output'],
        layer_name='Conv2 All Channels',
        output_path=os.path.join(output_dir, f'cnn_conv2_all_channels_state_{state_idx}.png')
    )

    layer_info = 'Multi-Scale 3-layer' if 'conv3_output' in activations else '2-layer'
    print(f'  ✅ CNN+Fusion+Scalar+LSTM visualization ({layer_info}) zapisana dla stanu {state_idx}')


def visualize_conv_layer(activation, layer_name, output_path, num_channels=16):
    """Wizualizuje wybrane kanały warstwy konwolucyjnej (grid 4x4)"""
    num_channels = min(num_channels, activation.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_channels):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(activation[i], cmap='viridis', interpolation='nearest')
        axes[row, col].set_title(f'Ch {i}', fontsize=10)
        axes[row, col].axis('off')
    
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
        # Reshape do 2D dla imshow
        features_2d = features.reshape(1, -1)
        
        im = axes[idx].imshow(features_2d, cmap='viridis', aspect='auto', interpolation='nearest')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_yticks([])
        
        # Colorbar
        plt.colorbar(im, ax=axes[idx], orientation='horizontal', pad=0.1)
        
        # Statystyki
        mean_val = features.mean()
        std_val = features.std()
        max_val = features.max()
        min_val = features.min()
        axes[idx].text(0.02, 0.95, f'Mean: {mean_val:.3f}, Std: {std_val:.3f}, Max: {max_val:.3f}, Min: {min_val:.3f}',
                      transform=axes[idx].transAxes, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
    
    action_probs = [a['probs'] for a in action_probs_list]
    action_probs_np = np.array(action_probs)
    
    for action_idx in range(len(action_names)):
        axes[2].plot(states, action_probs_np[:, action_idx], marker='o', label=action_names[action_idx], linewidth=2)
    
    axes[2].set_xlabel('Stan')
    axes[2].set_ylabel('Prawdopodobieństwo')
    axes[2].set_title('Prawdopodobieństwa Akcji')
    axes[2].set_xticks(states)
    axes[2].set_xticklabels([f'Stan {s}' for s in states])
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    overview_path = os.path.join(output_dirs['main'], 'neuron_activations_overview.png')
    plt.savefig(overview_path, dpi=150)
    plt.close()
    print(f'\n✅ Przegląd aktywacji zapisany: {overview_path}')
