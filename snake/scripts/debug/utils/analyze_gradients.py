"""
🌊 GRADIENT ANALYSIS MODULE
- compute_layer_gradients: Oblicza gradienty dla wszystkich warstw
- analyze_bottlenecks: Analiza bottlenecków w sieci
- analyze_gradient_flow_detailed: Szczegółowa analiza przepływu gradientów
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def compute_layer_gradients(model, obs, obs_tensor, lstm_states, action_idx, features_extractor):
    """
    Oblicza gradienty dla wszystkich warstw
    🆕 UPDATED: Pure Bottleneck (NO Skip) + Fixed MLP naming
    """
    policy = model.policy
    
    features_extractor.eval()
    policy.lstm_actor.train()
    
    obs_grad = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            obs_grad[k] = torch.tensor(v, dtype=torch.float32, device=policy.device, requires_grad=True).unsqueeze(0)
        else:
            obs_grad[k] = torch.tensor([v], dtype=torch.float32, device=policy.device, requires_grad=True).unsqueeze(0)
    
    # Forward z intermediate activations
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    # 🆕 CNN layers - Multi-Scale Architecture
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
    x_cnn = features_extractor.dropout2(x_cnn)
    x_cnn = torch.nn.functional.gelu(x_cnn)
    x_cnn.retain_grad()
    cnn_intermediates.append(('cnn', 3, 'GELU-2', x_cnn))

    # Layer 3 - MULTI-SCALE (opcjonalna)
    layer_offset = 4
    if features_extractor.has_conv3:
        identity = features_extractor.residual_proj(x_cnn)
        identity.retain_grad()
        cnn_intermediates.append(('cnn', 4, 'Residual-Proj', identity))
        
        x_local = features_extractor.conv3_local(x_cnn)
        x_local.retain_grad()
        cnn_intermediates.append(('cnn', 5, 'Conv3-Local', x_local))
        x_local = features_extractor.bn3_local(x_local)
        
        x_global = features_extractor.conv3_global(x_cnn)
        x_global.retain_grad()
        cnn_intermediates.append(('cnn', 6, 'Conv3-Global', x_global))
        x_global = features_extractor.bn3_global(x_global)
        
        x_combined = torch.cat([x_local, x_global], dim=1)
        x_combined.retain_grad()
        cnn_intermediates.append(('cnn', 7, 'Conv3-Concat', x_combined))
        
        x_combined = features_extractor.dropout3(x_combined)
        x_combined = torch.nn.functional.gelu(x_combined)
        x_combined.retain_grad()
        cnn_intermediates.append(('cnn', 8, 'GELU-3', x_combined))
        
        x_cnn = x_combined + identity
        x_cnn.retain_grad()
        cnn_intermediates.append(('cnn', 9, 'Residual-Add', x_cnn))
        layer_offset = 10

    # Flatten
    cnn_raw = features_extractor.flatten(x_cnn)
    cnn_raw = cnn_raw.float()
    cnn_raw.retain_grad()
    cnn_intermediates.append(('cnn', layer_offset, 'Flatten', cnn_raw))

    # 🆕 BOTTLENECK (single projection)
    cnn_features = features_extractor.cnn_bottleneck(cnn_raw)
    cnn_features.retain_grad()
    cnn_intermediates.append(('cnn', layer_offset + 1, 'Bottleneck', cnn_features))
    
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
    linear_idx = 0
    for i, layer in enumerate(features_extractor.scalar_linear):
        x_scalar = layer(x_scalar)
        x_scalar.retain_grad()
        
        # ✅ Proper layer naming
        if isinstance(layer, torch.nn.Linear):
            layer_name = f'Linear-{linear_idx}'
            linear_idx += 1
        elif isinstance(layer, torch.nn.LayerNorm):
            layer_name = f'LayerNorm-{i}'
        elif isinstance(layer, torch.nn.SiLU):
            layer_name = f'SiLU-{i}'
        elif isinstance(layer, torch.nn.Dropout):
            layer_name = f'Dropout-{i}'
        else:
            layer_name = f'Unknown-{i}'
        
        scalar_intermediates.append(('scalar', i, layer_name, x_scalar))
    
    scalar_features = x_scalar
    
    # Combined features
    combined = torch.cat([cnn_features, scalar_features], dim=-1)
    combined.retain_grad()
    features_final = features_extractor.fusion_main(combined)
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
    
    # MLP layers - POLICY
    mlp_intermediates = []
    x_mlp = latent_pi_grad
    linear_idx_policy = 0
    for i, layer in enumerate(policy.mlp_extractor.policy_net):
        x_mlp = layer(x_mlp)
        x_mlp.retain_grad()
        
        # ✅ Proper naming for policy MLP
        if isinstance(layer, torch.nn.Linear):
            layer_name = f'Policy-Linear-{linear_idx_policy}'
            linear_idx_policy += 1
        elif isinstance(layer, torch.nn.Tanh):
            layer_name = f'Policy-Tanh-{i}'
        elif isinstance(layer, torch.nn.ReLU):
            layer_name = f'Policy-ReLU-{i}'
        else:
            layer_name = f'Policy-{type(layer).__name__}-{i}'
        
        mlp_intermediates.append(('mlp_policy', i, layer_name, x_mlp))

    # MLP layers - VALUE
    x_mlp_vf = latent_pi_grad
    linear_idx_value = 0
    for i, layer in enumerate(policy.mlp_extractor.value_net):
        x_mlp_vf = layer(x_mlp_vf)
        x_mlp_vf.retain_grad()
        
        # ✅ Proper naming for value MLP
        if isinstance(layer, torch.nn.Linear):
            layer_name = f'Value-Linear-{linear_idx_value}'
            linear_idx_value += 1
        elif isinstance(layer, torch.nn.Tanh):
            layer_name = f'Value-Tanh-{i}'
        elif isinstance(layer, torch.nn.ReLU):
            layer_name = f'Value-ReLU-{i}'
        else:
            layer_name = f'Value-{type(layer).__name__}-{i}'
        
        mlp_intermediates.append(('mlp_value', i, layer_name, x_mlp_vf))

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
            state_layer_grads['layers'].append({
                'type': layer_type,
                'index': layer_idx,
                'name': layer_name,
                'activation_mean': activation.mean().item(),
                'activation_std': activation.std().item(),
                'gradient_mean': activation.grad.mean().item(),
                'gradient_std': activation.grad.std().item(),
                'gradient_norm': activation.grad.norm().item()
            })
    
    # Scalar gradienty
    for layer_type, layer_idx, layer_name, activation in scalar_intermediates:
        if activation.grad is not None:
            state_layer_grads['layers'].append({
                'type': layer_type,
                'index': layer_idx,
                'name': layer_name,
                'activation_mean': activation.mean().item(),
                'activation_std': activation.std().item(),
                'gradient_mean': activation.grad.mean().item(),
                'gradient_std': activation.grad.std().item(),
                'gradient_norm': activation.grad.norm().item()
            })
    
    # Combined features
    if features_final.grad is not None:
        state_layer_grads['layers'].append({
            'type': 'fusion',
            'index': 0,
            'name': 'Fusion',
            'activation_mean': features_final.mean().item(),
            'activation_std': features_final.std().item(),
            'gradient_mean': features_final.grad.mean().item(),
            'gradient_std': features_final.grad.std().item(),
            'gradient_norm': features_final.grad.norm().item()
        })
    
    # LSTM
    if latent_pi_grad.grad is not None:
        state_layer_grads['layers'].append({
            'type': 'lstm',
            'index': 0,
            'name': 'LSTM',
            'activation_mean': latent_pi_grad.mean().item(),
            'activation_std': latent_pi_grad.std().item(),
            'gradient_mean': latent_pi_grad.grad.mean().item(),
            'gradient_std': latent_pi_grad.grad.std().item(),
            'gradient_norm': latent_pi_grad.grad.norm().item()
        })
    
    # MLP gradienty (POLICY + VALUE)
    for layer_type, layer_idx, layer_name, activation in mlp_intermediates:
        if activation.grad is not None:
            state_layer_grads['layers'].append({
                'type': layer_type,  # 'mlp_policy' lub 'mlp_value'
                'index': layer_idx,
                'name': layer_name,
                'activation_mean': activation.mean().item(),
                'activation_std': activation.std().item(),
                'gradient_mean': activation.grad.mean().item(),
                'gradient_std': activation.grad.std().item(),
                'gradient_norm': activation.grad.norm().item()
            })
    
    model.policy.eval()
    return state_layer_grads


def analyze_bottlenecks(layer_gradients, action_names, output_dir):
    """
    ENHANCED BOTTLENECK ANALYSIS
    - Split visualization by network section (CNN/Scalars/Fusion/MLP)
    - Cross-state gradient flow heatmap
    - Adaptive thresholds per layer type
    - Information flow metrics
    """
    print("\n" + "="*80)
    print("\n=== ENHANCED BOTTLENECK ANALYSIS ===")
    print("="*80)
    
    # ==================== ORGANIZE LAYERS BY SECTION ====================
    sections = {
        'cnn': [],
        'scalar': [],
        'fusion': [],
        'lstm': [],
        'mlp_policy': [],
        'mlp_value': []
    }
    
    # Collect all layers from all 3 states
    all_layers_by_state = []
    for state_idx in range(3):
        state_data = layer_gradients[state_idx]
        layers = state_data['layers']
        all_layers_by_state.append(layers)
        
        # Organize by section
        for layer in layers:
            layer_type = layer['type']
            if layer_type in sections:
                # Check if layer already exists in section
                layer_names = [l['name'] for l in sections[layer_type]]
                if layer['name'] not in layer_names:
                    sections[layer_type].append(layer)
    
    # ==================== FIGURE 1: SPLIT BY SECTION (4 subplots) ====================
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    section_axes = {
        'cnn': fig.add_subplot(gs[0, 0]),
        'scalar': fig.add_subplot(gs[0, 1]),
        'fusion': fig.add_subplot(gs[1, 0]),
        'mlp_policy': fig.add_subplot(gs[1, 1])
    }
    
    bottleneck_report = []
    
    # Plot each section for all 3 states
    for section_name, ax in section_axes.items():
        if section_name not in sections or len(sections[section_name]) == 0:
            ax.text(0.5, 0.5, f'No {section_name.upper()} layers', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_title(f'{section_name.upper()} Layers', fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        
        # Collect data for this section across all states
        layer_names_section = []
        for state_idx in range(3):
            state_layers = all_layers_by_state[state_idx]
            for layer in state_layers:
                if layer['type'] == section_name and layer['name'] not in layer_names_section:
                    layer_names_section.append(layer['name'])
        
        # Plot bars for each state
        x = np.arange(len(layer_names_section))
        width = 0.12  # Narrower bars for 6 series (3 states × 2 metrics)
        
        colors_activation = ['#3498db', '#2980b9', '#1f618d']  # Blue shades
        colors_gradient = ['#e74c3c', '#c0392b', '#922b21']     # Red shades
        
        for state_idx in range(3):
            state_layers = all_layers_by_state[state_idx]
            action_idx = layer_gradients[state_idx]['action']
            
            activation_means = []
            gradient_norms = []
            
            for layer_name in layer_names_section:
                # Find layer in this state
                layer_data = None
                for layer in state_layers:
                    if layer['name'] == layer_name and layer['type'] == section_name:
                        layer_data = layer
                        break
                
                if layer_data:
                    # USE RMS instead of mean
                    act_rms = np.sqrt(layer_data['activation_mean']**2 + layer_data['activation_std']**2)
                    activation_means.append(act_rms)
                    gradient_norms.append(layer_data['gradient_norm'])
                else:
                    activation_means.append(0)
                    gradient_norms.append(0)
            
            # Normalize gradients for visualization
            max_grad = max(gradient_norms) if max(gradient_norms) > 0 else 1.0
            gradient_norms_normalized = [g / max_grad for g in gradient_norms]
            
            # Plot bars
            offset_act = -3*width + state_idx*width
            offset_grad = state_idx*width
            
            ax.bar(x + offset_act, activation_means, width, 
                  label=f'S{state_idx} Act', color=colors_activation[state_idx], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.bar(x + offset_grad, gradient_norms_normalized, width, 
                  label=f'S{state_idx} Grad', color=colors_gradient[state_idx], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Detect bottlenecks with adaptive thresholds
            for i, layer_name in enumerate(layer_names_section):
                activation_rms = activation_means[i]
                gradient_mag = gradient_norms[i]
                
                # Adaptive thresholds based on layer type
                if section_name == 'cnn':
                    act_threshold = 0.05
                    grad_threshold_high = 0.001
                    grad_threshold_medium = 0.01
                elif section_name == 'scalar':
                    act_threshold = 0.03
                    grad_threshold_high = 0.0005
                    grad_threshold_medium = 0.005
                else:
                    act_threshold = 0.05
                    grad_threshold_high = 0.001
                    grad_threshold_medium = 0.01
                
                # Check for bottleneck
                if activation_rms > act_threshold:
                    if gradient_mag < grad_threshold_high:
                        severity = 'HIGH'
                        color = 'red'
                    elif gradient_mag < grad_threshold_medium:
                        severity = 'MEDIUM'
                        color = 'orange'
                    else:
                        continue
                    
                    bottleneck_report.append({
                        'state': state_idx,
                        'section': section_name,
                        'layer': layer_name,
                        'activation': activation_rms,
                        'gradient': gradient_mag,
                        'severity': severity
                    })
                    
                    # Mark on plot
                    y_pos = max(activation_rms, gradient_norms_normalized[i]) + 0.05
                    ax.text(i + offset_grad, y_pos, '!', 
                           ha='center', fontsize=12, color='white', weight='bold',
                           bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Magnitude (RMS for activations)', fontsize=11)
        ax.set_title(f'{section_name.upper()} Layers - RMS Activations vs Gradients', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names_section, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8, ncol=2, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add gradient max value as text
        ax.text(0.02, 0.98, f'Grad max: {max_grad:.4f}', 
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== FIGURE 1 BOTTOM: LSTM (single row) ====================
    ax_lstm = fig.add_subplot(gs[2, :])
    
    if 'lstm' in sections and len(sections['lstm']) > 0:
        lstm_data = []
        for state_idx in range(3):
            state_layers = all_layers_by_state[state_idx]
            for layer in state_layers:
                if layer['type'] == 'lstm':
                    # Use RMS
                    act_rms = np.sqrt(layer['activation_mean']**2 + layer['activation_std']**2)
                    lstm_data.append({
                        'state': state_idx,
                        'activation': act_rms,
                        'gradient': layer['gradient_norm']
                    })
        
        states = [d['state'] for d in lstm_data]
        activations = [d['activation'] for d in lstm_data]
        gradients = [d['gradient'] for d in lstm_data]
        
        x = np.arange(len(states))
        width = 0.35
        
        ax_lstm.bar(x - width/2, activations, width, label='Activation', 
                   color='#9b59b6', alpha=0.8, edgecolor='black')
        ax_lstm.bar(x + width/2, gradients, width, label='Gradient Norm', 
                   color='#8e44ad', alpha=0.8, edgecolor='black')
        
        ax_lstm.set_xlabel('State', fontsize=11)
        ax_lstm.set_ylabel('Magnitude', fontsize=11)
        ax_lstm.set_title('LSTM Layer - Activations vs Gradients', 
                         fontsize=13, fontweight='bold')
        ax_lstm.set_xticks(x)
        ax_lstm.set_xticklabels([f'State {s}' for s in states])
        ax_lstm.legend(fontsize=10)
        ax_lstm.grid(axis='y', alpha=0.3)
    else:
        ax_lstm.text(0.5, 0.5, 'No LSTM layer found', 
                    ha='center', va='center', fontsize=14, color='gray')
        ax_lstm.axis('off')
    
    plt.savefig(os.path.join(output_dir, 'bottleneck_analysis_split.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Split bottleneck analysis saved: bottleneck_analysis_split.png')
    
    # ==================== FIGURE 2: GRADIENT FLOW HEATMAP ====================
    print("\n[INFO] Generating gradient flow heatmap...")
    
    # Collect all unique layer names (in order)
    all_layer_names = []
    for state_layers in all_layers_by_state:
        for layer in state_layers:
            if layer['name'] not in all_layer_names:
                all_layer_names.append(layer['name'])
    
    # Build heatmap matrix: [layers × states]
    heatmap_data = np.zeros((len(all_layer_names), 3))
    
    # Track min/max ratios
    min_ratio = float('inf')
    max_ratio = float('-inf')
    
    for state_idx in range(3):
        state_layers = all_layers_by_state[state_idx]
        for layer in state_layers:
            layer_idx = all_layer_names.index(layer['name'])
            
            # Compute RMS activation
            act_mean = layer['activation_mean']
            act_std = layer['activation_std']
            activation_rms = np.sqrt(act_mean**2 + act_std**2) + 1e-8
            
            gradient = layer['gradient_norm'] + 1e-8
            
            # Gradient-to-activation ratio (log scale)
            ratio = gradient / activation_rms
            log_ratio = np.log10(ratio)
            heatmap_data[layer_idx, state_idx] = log_ratio
            
            # Track extremes
            min_ratio = min(min_ratio, log_ratio)
            max_ratio = max(max_ratio, log_ratio)
    
    print(f"   Heatmap range: log10(ratio) in [{min_ratio:.2f}, {max_ratio:.2f}]")
    print(f"   Mapping: <-3 = Critical, -3 to -2 = Vanishing, >-2 = Healthy")
    
    # Plot heatmap with optimized figure size
    num_layers = len(all_layer_names)
    fig_height = max(8, num_layers * 0.3)  # Tighter spacing
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Custom colormap: red (vanishing) → yellow (borderline) → green (healthy)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('gradient_flow', colors, N=n_bins)
    
    # Dynamic vmin/vmax based on actual data
    vmin = max(-4, min_ratio - 0.5)
    vmax = min(1, max_ratio + 0.5)
    
    print(f"   Colormap range: [{vmin:.2f}, {vmax:.2f}]")
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Add text annotations (Windows-compatible symbols)
    for i in range(len(all_layer_names)):
        for j in range(3):
            value = heatmap_data[i, j]
            
            # Determine severity (simplified symbols)
            if value < -3:  # ratio < 0.001
                symbol = 'X'
                color = 'white'
                bgcolor = '#d73027'
            elif value < -2:  # ratio < 0.01
                symbol = '!'
                color = 'black'
                bgcolor = '#fee090'
            else:
                symbol = 'OK'
                color = 'white'
                bgcolor = '#4575b4'
            
            ax.text(j, i, symbol, ha='center', va='center', fontsize=9, 
                   color=color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.25', facecolor=bgcolor, 
                            edgecolor='none', alpha=0.7))
    
    # Labels
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f'State {i}\n({action_names[layer_gradients[i]["action"]]})' 
                        for i in range(3)], fontsize=11)
    ax.set_yticks(np.arange(len(all_layer_names)))
    ax.set_yticklabels(all_layer_names, fontsize=9)
    
    ax.set_xlabel('State (Action)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Flow Heatmap (Gradient/Activation Ratio, log10)', 
                fontsize=14, fontweight='bold')
    
    # Colorbar with compact layout
    cbar = plt.colorbar(im, ax=ax, label='log10(Grad/Act)', pad=0.01, fraction=0.046)
    
    # Add threshold reference lines on colorbar
    cbar.ax.axhline(-3, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    cbar.ax.axhline(-2, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    
    # Add threshold labels INSIDE the colorbar
    cbar.ax.text(0.5, -3, ' Critical', fontsize=7, color='white', 
                va='center', ha='left', weight='bold')
    cbar.ax.text(0.5, -2, ' Vanish', fontsize=7, color='white', 
                va='center', ha='left', weight='bold')
    cbar.ax.text(0.5, -1, ' Healthy', fontsize=7, color='white', 
                va='center', ha='left', weight='bold')
    
    # Compact legend below the plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d73027', edgecolor='black', label='X: Critical (<0.001)'),
        Patch(facecolor='#fee090', edgecolor='black', label='!: Vanishing (0.001-0.01)'),
        Patch(facecolor='#4575b4', edgecolor='black', label='OK: Healthy (>0.01)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
             fontsize=8, framealpha=0.9, ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    plt.savefig(os.path.join(output_dir, 'bottleneck_gradient_heatmap.png'), 
               dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('[OK] Gradient flow heatmap saved: bottleneck_gradient_heatmap.png')
    
    # ==================== BOTTLENECK REPORT ====================
    print("\n" + "="*80)
    print("=== BOTTLENECK REPORT ===")
    print("="*80)
    
    high_severity = [b for b in bottleneck_report if b['severity'] == 'HIGH']
    medium_severity = [b for b in bottleneck_report if b['severity'] == 'MEDIUM']
    
    if high_severity:
        print(f"\n[HIGH] HIGH SEVERITY BOTTLENECKS ({len(high_severity)} cases):")
        for b in high_severity:
            print(f"   State {b['state']}, {b['section'].upper()}: {b['layer']}")
            print(f"   -> Activation: {b['activation']:.4f}, Gradient: {b['gradient']:.6f}")
    
    if medium_severity:
        print(f"\n[MED] MEDIUM SEVERITY BOTTLENECKS ({len(medium_severity)} cases):")
        for b in medium_severity:
            print(f"   State {b['state']}, {b['section'].upper()}: {b['layer']}")
            print(f"   -> Activation: {b['activation']:.4f}, Gradient: {b['gradient']:.6f}")
    
    if not high_severity and not medium_severity:
        print("\n[OK] No critical bottlenecks detected!")
    
    # ==================== CROSS-STATE CONSISTENCY ANALYSIS ====================
    print("\n" + "="*80)
    print("=== CROSS-STATE CONSISTENCY ===")
    print("="*80)
    
    # Find layers that are ALWAYS problematic (across all 3 states)
    layer_issue_count = {}
    for b in bottleneck_report:
        key = f"{b['section']}:{b['layer']}"
        if key not in layer_issue_count:
            layer_issue_count[key] = []
        layer_issue_count[key].append(b['state'])
    
    persistent_bottlenecks = {k: v for k, v in layer_issue_count.items() if len(v) == 3}
    inconsistent_bottlenecks = {k: v for k, v in layer_issue_count.items() if 0 < len(v) < 3}
    
    if persistent_bottlenecks:
        print(f"\n[!] PERSISTENT BOTTLENECKS (all 3 states):")
        for layer_key in persistent_bottlenecks:
            section, layer_name = layer_key.split(':')
            print(f"   {section.upper()}: {layer_name}")
        print("\n[TIP] These layers need architectural changes!")
    
    if inconsistent_bottlenecks:
        print(f"\n[~] INCONSISTENT BOTTLENECKS (1-2 states only):")
        for layer_key, states in inconsistent_bottlenecks.items():
            section, layer_name = layer_key.split(':')
            print(f"   {section.upper()}: {layer_name} (states: {states})")
        print("\n[TIP] These may be state-dependent issues (check input diversity)")
    
    # ==================== SAVE CSV REPORT ====================
    bottleneck_csv_path = os.path.join(output_dir, 'bottleneck_report.csv')
    if bottleneck_report:
        with open(bottleneck_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['state', 'section', 'layer', 
                                                   'activation', 'gradient', 'severity'])
            writer.writeheader()
            writer.writerows(bottleneck_report)
        print(f"\n[OK] Bottleneck report saved: {bottleneck_csv_path}")
    
    print("\n" + "="*80)
    
    return bottleneck_report


def analyze_gradient_flow_detailed(model, env, output_dir, num_samples=50):
    """
    🌊 GRADIENT FLOW DETAILED ANALYSIS - FIXED RMS VERSION
    - Per-layer gradient magnitude (RMS-based)
    - Gradient vanishing/explosion detection
    - Gradient-to-weight ratio analysis
    - Layer-wise gradient statistics
    
    ✅ CHANGES:
    - Uses RMS for activations (sqrt(mean^2 + std^2)) instead of just mean
    - More accurate for LayerNorm'ed layers
    """
    print("\n" + "="*80)
    print("🌊 GRADIENT FLOW DETAILED ANALYSIS (RMS)")
    print("="*80)
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # Collect gradient statistics
    layer_gradient_stats = {}
    
    for sample_idx in range(num_samples):
        obs, _ = env.reset()
        
        # Konwersja obs do tensora
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.tensor(value, dtype=torch.float32, device=policy.device).unsqueeze(0)
            else:
                obs_tensor[key] = torch.tensor([value], dtype=torch.float32, device=policy.device).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # CNN
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
            
            # LSTM - standardowa inicjalizacja stanów
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
            lstm_out, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
            latent_pi = lstm_out.squeeze(1)
            
            # MLP
            latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
            action_logits = policy.action_net(latent_pi_mlp)
            action = torch.argmax(action_logits, dim=-1)
        
        # Backward pass z gradientami
        policy.zero_grad()
        
        # Oblicz gradienty
        state_grads = compute_layer_gradients(
            model, obs, obs_tensor, lstm_states, action.item(), features_extractor
        )
        
        # ✅ Agreguj statystyki per layer z RMS
        for layer_data in state_grads['layers']:
            layer_name = layer_data['name']
            
            if layer_name not in layer_gradient_stats:
                layer_gradient_stats[layer_name] = {
                    'gradient_norms': [],
                    'gradient_means': [],
                    'gradient_stds': [],
                    'activation_means': [],
                    'activation_stds': [],
                    'type': layer_data['type']
                }
            
            layer_gradient_stats[layer_name]['gradient_norms'].append(layer_data['gradient_norm'])
            layer_gradient_stats[layer_name]['gradient_means'].append(layer_data['gradient_mean'])
            layer_gradient_stats[layer_name]['gradient_stds'].append(layer_data['gradient_std'])
            layer_gradient_stats[layer_name]['activation_means'].append(layer_data['activation_mean'])
            layer_gradient_stats[layer_name]['activation_stds'].append(layer_data['activation_std'])
        
        if (sample_idx + 1) % 10 == 0:
            print(f"   Processed {sample_idx + 1}/{num_samples} samples...")
    
    # ==================== ANALIZA STATYSTYK (RMS-BASED) ====================
    print("\n📊 Computing gradient flow statistics (RMS method)...")
    
    layer_names = list(layer_gradient_stats.keys())
    avg_gradient_norms = []
    std_gradient_norms = []
    avg_activations_rms = []  # ✅ RMS zamiast mean
    gradient_to_activation_ratios = []
    
    for layer_name in layer_names:
        stats = layer_gradient_stats[layer_name]
        
        # Gradient norm (average)
        avg_grad_norm = np.mean(stats['gradient_norms'])
        std_grad_norm = np.std(stats['gradient_norms'])
        
        # ✅ ACTIVATION RMS: sqrt(mean^2 + std^2)
        act_means = np.array(stats['activation_means'])
        act_stds = np.array(stats['activation_stds'])
        activation_rms_samples = np.sqrt(act_means**2 + act_stds**2)
        avg_activation_rms = np.mean(activation_rms_samples)
        
        avg_gradient_norms.append(avg_grad_norm)
        std_gradient_norms.append(std_grad_norm)
        avg_activations_rms.append(avg_activation_rms)
        
        # Gradient-to-activation ratio (using RMS)
        ratio = avg_grad_norm / (avg_activation_rms + 1e-8)
        gradient_to_activation_ratios.append(ratio)
    
    # ==================== WIZUALIZACJA ====================
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Gradient norms per layer
    ax = axes[0, 0]
    x = np.arange(len(layer_names))
    
    bars = ax.bar(x, avg_gradient_norms, yerr=std_gradient_norms, capsize=5,
                  color='#3498db', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Gradient Norm')
    ax.set_title('Gradient Flow: Per-Layer Gradient Magnitude')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Show max gradient value
    max_grad = max(avg_gradient_norms)
    ax.text(0.02, 0.98, f'Grad max: {max_grad:.4f}', 
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Highlight vanishing gradients (< 0.01)
    for i, (layer_name, avg_grad) in enumerate(zip(layer_names, avg_gradient_norms)):
        if avg_grad < 0.01:
            ax.text(i, avg_grad + max_grad*0.02, '⚠️', ha='center', fontsize=10, color='red')
    
    # Plot 2: Gradient-to-Activation ratio (RMS-based)
    ax = axes[0, 1]
    
    colors_ratio = ['red' if r < 0.001 else 'orange' if r < 0.01 else 'green' 
                   for r in gradient_to_activation_ratios]
    bars = ax.bar(x, gradient_to_activation_ratios, color=colors_ratio, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient / Activation (RMS) Ratio')
    ax.set_title('Gradient-to-Activation Ratio (Red = Vanishing)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(0.01, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.01)')
    ax.axhline(0.001, color='red', linestyle='--', alpha=0.5, label='Critical (0.001)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Add max ratio annotation
    max_ratio = max(gradient_to_activation_ratios)
    ax.text(0.02, 0.98, f'Grad max: {max_ratio:.4f}', 
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Activation magnitudes (RMS)
    ax = axes[1, 0]
    
    bars = ax.bar(x, avg_activations_rms, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Activation Magnitude (RMS)')
    ax.set_title('Per-Layer Activation Magnitudes (RMS)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Gradient distribution (boxplot for selected layers)
    ax = axes[1, 1]
    
    # Select key layers for boxplot
    key_layers = ['GELU-1', 'GELU-2', 'GELU-3', 'Bottleneck', 'Fusion', 'LSTM']
    
    boxplot_data = []
    boxplot_labels = []
    for layer_name in key_layers:
        if layer_name in layer_gradient_stats:
            boxplot_data.append(layer_gradient_stats[layer_name]['gradient_norms'])
            boxplot_labels.append(layer_name)
    
    if boxplot_data:
        bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#9b59b6')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Gradient Norm Distribution')
        ax.set_title('Gradient Distribution (Key Layers)')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    gradient_flow_path = os.path.join(output_dir, 'gradient_flow_detailed.png')
    plt.savefig(gradient_flow_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Gradient flow analysis saved: {gradient_flow_path}")
    
    # ==================== SAVE STATISTICS ====================
    stats_csv_path = os.path.join(output_dir, 'gradient_flow_stats.csv')
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Type', 'Avg_Gradient_Norm', 'Std_Gradient_Norm', 
                        'Avg_Activation_RMS', 'Gradient_to_Activation_Ratio', 'Status'])
        
        for i, layer_name in enumerate(layer_names):
            ratio = gradient_to_activation_ratios[i]
            
            if ratio < 0.001:
                status = 'CRITICAL_VANISHING'
            elif ratio < 0.01:
                status = 'VANISHING'
            elif ratio > 1.0:
                status = 'EXPLOSION'
            else:
                status = 'HEALTHY'
            
            writer.writerow([
                layer_name,
                layer_gradient_stats[layer_name]['type'],
                f"{avg_gradient_norms[i]:.6f}",
                f"{std_gradient_norms[i]:.6f}",
                f"{avg_activations_rms[i]:.6f}",  # ✅ RMS
                f"{ratio:.6f}",
                status
            ])
    
    print(f"✅ Gradient statistics saved: {stats_csv_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 GRADIENT FLOW SUMMARY (RMS-based)")
    print("="*80)
    
    vanishing_layers = [layer_names[i] for i, r in enumerate(gradient_to_activation_ratios) if r < 0.01]
    critical_vanishing = [layer_names[i] for i, r in enumerate(gradient_to_activation_ratios) if r < 0.001]
    exploding_layers = [layer_names[i] for i, r in enumerate(gradient_to_activation_ratios) if r > 1.0]
    
    if critical_vanishing:
        print(f"\n🔴 CRITICAL VANISHING GRADIENTS ({len(critical_vanishing)} layers):")
        for layer in critical_vanishing:
            idx = layer_names.index(layer)
            ratio = gradient_to_activation_ratios[idx]
            grad = avg_gradient_norms[idx]
            act = avg_activations_rms[idx]
            print(f"   - {layer}: grad={grad:.6f}, act_rms={act:.6f}, ratio={ratio:.6f}")
    
    if vanishing_layers:
        print(f"\n🟡 VANISHING GRADIENTS ({len(vanishing_layers)} layers):")
        for layer in vanishing_layers:
            if layer not in critical_vanishing:
                idx = layer_names.index(layer)
                ratio = gradient_to_activation_ratios[idx]
                grad = avg_gradient_norms[idx]
                act = avg_activations_rms[idx]
                print(f"   - {layer}: grad={grad:.6f}, act_rms={act:.6f}, ratio={ratio:.6f}")
    
    if exploding_layers:
        print(f"\n🔥 EXPLODING GRADIENTS ({len(exploding_layers)} layers):")
        for layer in exploding_layers:
            idx = layer_names.index(layer)
            ratio = gradient_to_activation_ratios[idx]
            grad = avg_gradient_norms[idx]
            act = avg_activations_rms[idx]
            print(f"   - {layer}: grad={grad:.6f}, act_rms={act:.6f}, ratio={ratio:.6f}")
    
    if not vanishing_layers and not exploding_layers:
        print("\n✅ Gradient flow is HEALTHY across all layers!")
    
    print(f"\nAverage gradient norm (all layers): {np.mean(avg_gradient_norms):.6f}")
    print(f"Average gradient-to-activation ratio: {np.mean(gradient_to_activation_ratios):.6f}")
    print(f"Max gradient norm: {max(avg_gradient_norms):.6f}")
    print(f"Max activation RMS: {max(avg_activations_rms):.6f}")
    
    print("="*80)