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
    🆕 UPDATED: Pure Bottleneck (NO Skip)
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
    for i, layer in enumerate(features_extractor.scalar_linear):
        x_scalar = layer(x_scalar)
        x_scalar.retain_grad()
        scalar_intermediates.append(('scalar', i, f'Linear-{i}', x_scalar))
    
    scalar_features = x_scalar
    scalar_features.retain_grad()
    scalar_intermediates.append(('scalar', len(scalar_intermediates), 'Output', scalar_features))
    
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
    
    # MLP layers
    mlp_intermediates = []
    x_mlp = latent_pi_grad
    for i, layer in enumerate(policy.mlp_extractor.policy_net):
        x_mlp = layer(x_mlp)
        x_mlp.retain_grad()
        mlp_intermediates.append(('mlp', i, f'MLP-{i}', x_mlp))
    
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
    
    # MLP gradienty
    for layer_type, layer_idx, layer_name, activation in mlp_intermediates:
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
    
    model.policy.eval()
    return state_layer_grads


def analyze_bottlenecks(layer_gradients, action_names, output_dir):
    """
    Analiza bottlenecków w sieci
    🆕 UPDATED: Dodano etykiety dla residual connections
    """
    print("\n=== Analiza bottlenecków (aktywacje vs gradienty) ===")
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    
    bottleneck_report = []
    
    for state_idx in range(3):
        state_data = layer_gradients[state_idx]
        layers = state_data['layers']
        
        # Przygotuj dane do wykresu
        layer_names = [l['name'] for l in layers]
        activation_means = [l['activation_mean'] for l in layers]
        gradient_norms = [l['gradient_norm'] for l in layers]
        
        # Normalizacja gradient norm (dla lepszej wizualizacji)
        max_grad = max(gradient_norms) if max(gradient_norms) > 0 else 1.0
        gradient_norms_normalized = [g / max_grad for g in gradient_norms]
        
        ax = axes[state_idx]
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        # Bars dla aktywacji i gradientów
        bars1 = ax.bar(x - width/2, activation_means, width, label='Activation Mean', color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, gradient_norms_normalized, width, label='Gradient Norm (normalized)', color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Stan {state_idx} - Akcja: {action_names[state_data["action"]]}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Wykryj bottlenecki (niskie gradienty przy wysokich aktywacjach)
        for i, layer in enumerate(layers):
            activation_mag = abs(layer['activation_mean'])
            gradient_mag = layer['gradient_norm']
            
            # Bottleneck jeśli:
            # 1. Aktywacja > 0.1 (layer jest aktywny)
            # 2. Gradient norm < 0.01 (gradient vanishing)
            if activation_mag > 0.1 and gradient_mag < 0.01:
                severity = 'HIGH' if gradient_mag < 0.001 else 'MEDIUM'
                bottleneck_report.append({
                    'state': state_idx,
                    'layer': layer['name'],
                    'activation': activation_mag,
                    'gradient': gradient_mag,
                    'severity': severity
                })
                
                # Oznacz na wykresie
                color = 'red' if severity == 'HIGH' else 'orange'
                ax.text(i, max(activation_mag, gradient_norms_normalized[i]) + 0.05, 
                       '⚠️', ha='center', fontsize=14, color=color)
    
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
        print(f"\n🔴 WYSOKIE RYZYKO ({len(high_severity)} przypadków):")
        for b in high_severity:
            print(f"   Stan {b['state']}, Layer: {b['layer']}")
            print(f"   → Activation: {b['activation']:.4f}, Gradient: {b['gradient']:.6f}")
    
    if medium_severity:
        print(f"\n🟡 ŚREDNIE RYZYKO ({len(medium_severity)} przypadków):")
        for b in medium_severity:
            print(f"   Stan {b['state']}, Layer: {b['layer']}")
            print(f"   → Activation: {b['activation']:.4f}, Gradient: {b['gradient']:.6f}")
    
    if not high_severity and not medium_severity:
        print("\n✅ Brak krytycznych bottlenecków!")
    
    bottleneck_csv_path = os.path.join(output_dir, 'bottleneck_report.csv')
    if bottleneck_report:
        with open(bottleneck_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['state', 'layer', 'activation', 'gradient', 'severity'])
            writer.writeheader()
            writer.writerows(bottleneck_report)
        print(f"\n✅ Raport zapisany: {bottleneck_csv_path}")
    
    return bottleneck_report


def analyze_gradient_flow_detailed(model, env, output_dir, num_samples=50):
    """
    🌊 GRADIENT FLOW DETAILED ANALYSIS
    - Per-layer gradient magnitude
    - Gradient vanishing/explosion detection
    - Gradient-to-weight ratio analysis
    - Layer-wise gradient statistics
    """
    print("\n" + "="*80)
    print("🌊 GRADIENT FLOW DETAILED ANALYSIS")
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
        
        # Agreguj statystyki per layer
        for layer_data in state_grads['layers']:
            layer_name = layer_data['name']
            
            if layer_name not in layer_gradient_stats:
                layer_gradient_stats[layer_name] = {
                    'gradient_norms': [],
                    'gradient_means': [],
                    'activation_means': [],
                    'type': layer_data['type']
                }
            
            layer_gradient_stats[layer_name]['gradient_norms'].append(layer_data['gradient_norm'])
            layer_gradient_stats[layer_name]['gradient_means'].append(abs(layer_data['gradient_mean']))
            layer_gradient_stats[layer_name]['activation_means'].append(abs(layer_data['activation_mean']))
        
        if (sample_idx + 1) % 10 == 0:
            print(f"   Processed {sample_idx + 1}/{num_samples} samples...")
    
    # ==================== ANALIZA STATYSTYK ====================
    print("\n📊 Computing gradient flow statistics...")
    
    layer_names = list(layer_gradient_stats.keys())
    avg_gradient_norms = []
    std_gradient_norms = []
    avg_activations = []
    gradient_to_activation_ratios = []
    
    for layer_name in layer_names:
        stats = layer_gradient_stats[layer_name]
        
        avg_grad_norm = np.mean(stats['gradient_norms'])
        std_grad_norm = np.std(stats['gradient_norms'])
        avg_activation = np.mean(stats['activation_means'])
        
        avg_gradient_norms.append(avg_grad_norm)
        std_gradient_norms.append(std_grad_norm)
        avg_activations.append(avg_activation)
        
        # Gradient-to-activation ratio
        ratio = avg_grad_norm / (avg_activation + 1e-8)
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
    
    # Highlight vanishing gradients (< 0.01)
    for i, (layer_name, avg_grad) in enumerate(zip(layer_names, avg_gradient_norms)):
        if avg_grad < 0.01:
            ax.text(i, avg_grad + 0.005, '⚠️', ha='center', fontsize=12, color='red')
    
    # Plot 2: Gradient-to-Activation ratio
    ax = axes[0, 1]
    
    colors_ratio = ['red' if r < 0.001 else 'orange' if r < 0.01 else 'green' for r in gradient_to_activation_ratios]
    bars = ax.bar(x, gradient_to_activation_ratios, color=colors_ratio, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient / Activation Ratio')
    ax.set_title('Gradient-to-Activation Ratio (Red = Vanishing)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(0.01, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.01)')
    ax.axhline(0.001, color='red', linestyle='--', alpha=0.5, label='Critical (0.001)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Activation magnitudes
    ax = axes[1, 0]
    
    bars = ax.bar(x, avg_activations, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Activation Magnitude')
    ax.set_title('Per-Layer Activation Magnitudes')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Gradient distribution (boxplot for selected layers)
    ax = axes[1, 1]
    
    # Select key layers for boxplot
    key_layers = ['GELU-1', 'GELU-2', 'Bottleneck', 'Fusion', 'LSTM']
    if features_extractor.has_conv3:
        key_layers.insert(2, 'GELU-3')
    
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
    plt.savefig(gradient_flow_path, dpi=150)
    plt.close()
    print(f"✅ Gradient flow analysis saved: {gradient_flow_path}")
    
    # ==================== SAVE STATISTICS ====================
    stats_csv_path = os.path.join(output_dir, 'gradient_flow_stats.csv')
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Type', 'Avg_Gradient_Norm', 'Std_Gradient_Norm', 
                        'Avg_Activation', 'Gradient_to_Activation_Ratio', 'Status'])
        
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
                f"{avg_activations[i]:.6f}",
                f"{ratio:.6f}",
                status
            ])
    
    print(f"✅ Gradient statistics saved: {stats_csv_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 GRADIENT FLOW SUMMARY")
    print("="*80)
    
    vanishing_layers = [layer_names[i] for i, r in enumerate(gradient_to_activation_ratios) if r < 0.01]
    critical_vanishing = [layer_names[i] for i, r in enumerate(gradient_to_activation_ratios) if r < 0.001]
    exploding_layers = [layer_names[i] for i, r in enumerate(gradient_to_activation_ratios) if r > 1.0]
    
    if critical_vanishing:
        print(f"\n🔴 CRITICAL VANISHING GRADIENTS ({len(critical_vanishing)} layers):")
        for layer in critical_vanishing:
            print(f"   - {layer}")
    
    if vanishing_layers:
        print(f"\n🟡 VANISHING GRADIENTS ({len(vanishing_layers)} layers):")
        for layer in vanishing_layers:
            if layer not in critical_vanishing:
                print(f"   - {layer}")
    
    if exploding_layers:
        print(f"\n🔥 EXPLODING GRADIENTS ({len(exploding_layers)} layers):")
        for layer in exploding_layers:
            print(f"   - {layer}")
    
    if not vanishing_layers and not exploding_layers:
        print("\n✅ Gradient flow is HEALTHY across all layers!")
    
    print(f"\nAverage gradient norm (all layers): {np.mean(avg_gradient_norms):.6f}")
    print(f"Average gradient-to-activation ratio: {np.mean(gradient_to_activation_ratios):.6f}")
