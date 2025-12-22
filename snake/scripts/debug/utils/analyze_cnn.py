"""
🔍 CNN COMPREHENSIVE ANALYSIS MODULE

Scalona analiza CNN:
- Channel specialization (dead channels, sparsity)
- Activation saturation (GELU saturacja)
- Layer visualizations (Conv1/2/3 filters)

Wszystkie wyniki w jednym katalogu: 02_cnn_layers/
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_cnn_layers(model, env, output_dir, num_samples=100):
    """
    🔍 KOMPLEKSOWA ANALIZA WARSTW CNN
    Łączy: specialization + saturation + visualizations
    """
    print("\n" + "="*80)
    print("🔍 CNN LAYERS COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # Utwórz podkatalogi
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # ==================== ZBIERANIE DANYCH ====================
    print(f"\n📊 Collecting data from {num_samples} samples...")
    print(f"   Analyzing {len(features_extractor.conv_layers)} CNN layers...")
    
    # Dynamic handling of 4 conv layers
    num_layers = len(features_extractor.conv_layers)
    all_conv_activations = [[] for _ in range(num_layers)]
    activation_data = {}
    for i in range(num_layers):
        activation_data[f'conv{i+1}_pre'] = []
        activation_data[f'conv{i+1}_post'] = []
    
    for sample_idx in range(num_samples):
        obs, _ = env.reset()
        
        # Random steps
        for _ in range(np.random.randint(0, 20)):
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
        
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
            
            # Process all 4 CNN layers
            x = image
            for layer_idx in range(num_layers):
                x_pre = features_extractor.conv_layers[layer_idx](x)
                x_pre = features_extractor.bn_layers[layer_idx](x_pre)
                
                pre_key = f'conv{layer_idx+1}_pre'
                post_key = f'conv{layer_idx+1}_post'
                
                activation_data[pre_key].extend(x_pre.flatten().cpu().numpy())
                
                x = torch.nn.functional.gelu(x_pre)
                activation_data[post_key].extend(x.flatten().cpu().numpy())
                all_conv_activations[layer_idx].append(x.detach().cpu().numpy()[0])
        
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{num_samples} samples")
    
    # Convert to numpy
    for i in range(num_layers):
        all_conv_activations[i] = np.array(all_conv_activations[i])
    
    for key in activation_data.keys():
        if activation_data[key]:
            activation_data[key] = np.array(activation_data[key])
    
    # ==================== [1/3] CHANNEL SPECIALIZATION ====================
    print("\n🎯 Analyzing channel specialization...")
    
    def analyze_channels(activations, layer_name):
        num_channels = activations.shape[1]
        channel_stats = []
        
        for ch in range(num_channels):
            ch_data = activations[:, ch, :, :].flatten()
            
            mean_activation = ch_data.mean()
            std_activation = ch_data.std()
            max_activation = ch_data.max()
            sparsity = (np.abs(ch_data) < 0.01).sum() / len(ch_data)
            
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
    
    # Analyze all 4 layers
    all_layer_stats = []
    for i in range(num_layers):
        stats = analyze_channels(all_conv_activations[i], f'Conv{i+1}')
        all_layer_stats.append(stats)
    
    # Plot specialization (2x2 grid dla każdej pary warstw)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    
    for layer_idx in range(num_layers):
        layer_stats = all_layer_stats[layer_idx]
        layer_name = f'Conv{layer_idx+1}'
        
        # Mean activation
        channels = [s['channel'] for s in layer_stats]
        means = [s['mean'] for s in layer_stats]
        is_dead = [s['is_dead'] for s in layer_stats]
        colors = ['red' if d else 'green' for d in is_dead]
        
        axes[layer_idx, 0].bar(channels, means, color=colors, alpha=0.7, edgecolor='black')
        axes[layer_idx, 0].set_xlabel('Channel')
        axes[layer_idx, 0].set_ylabel('Mean Activation')
        axes[layer_idx, 0].set_title(f'{layer_name} - Channel Activity (Red = Dead)')
        axes[layer_idx, 0].grid(axis='y', alpha=0.3)
        
        # Sparsity
        sparsity = [s['sparsity'] for s in layer_stats]
        axes[layer_idx, 1].bar(channels, sparsity, color='#3498db', alpha=0.7, edgecolor='black')
        axes[layer_idx, 1].set_xlabel('Channel')
        axes[layer_idx, 1].set_ylabel('Sparsity (% near-zero)')
        axes[layer_idx, 1].set_title(f'{layer_name} - Sparsity per Channel')
        axes[layer_idx, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_specialization.png'), dpi=150)
    plt.close()
    print(f"✅ Channel specialization saved")
    
    # ==================== [2/3] ACTIVATION SATURATION ====================
    print("\n🔥 Analyzing activation saturation...")
    
    saturation_stats = {}
    for i in range(num_layers):
        layer_name = f'conv{i+1}'
        pre_key = f'{layer_name}_pre'
        post_key = f'{layer_name}_post'
        
        if len(activation_data[pre_key]) > 0:
            pre_vals = activation_data[pre_key]
            post_vals = activation_data[post_key]
            
            saturated_pre = np.abs(pre_vals) > 3
            dead_post = np.abs(post_vals) < 0.01
            
            saturation_stats[layer_name] = {
                'pre_mean': pre_vals.mean(),
                'pre_std': pre_vals.std(),
                'post_mean': post_vals.mean(),
                'post_std': post_vals.std(),
                'saturation_rate_pre': saturated_pre.mean() * 100,
                'dead_rate': dead_post.mean() * 100,
                'pre_vals': pre_vals,
                'post_vals': post_vals
            }
    
    # Plot saturation (dynamic based on number of layers - use subplots for 4 layers)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    
    for i in range(num_layers):
        layer_name = f'conv{i+1}'
        if layer_name in saturation_stats:
            stats = saturation_stats[layer_name]
            
            # Pre-activation
            ax = axes[i, 0]
            pre_vals = stats['pre_vals']
            ax.hist(pre_vals, bins=100, alpha=0.7, edgecolor='black', color='#3498db', density=True)
            ax.axvline(-3, color='red', linestyle='--', alpha=0.7, label='Saturation')
            ax.axvline(3, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Pre-activation value')
            ax.set_ylabel('Density')
            ax.set_title(f'{layer_name.upper()} Pre-Activation Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
            
            sat_rate = stats['saturation_rate_pre']
            ax.text(0.02, 0.98, f'Saturation: {sat_rate:.1f}%\nMean: {pre_vals.mean():.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Post-activation
            ax = axes[i, 1]
            post_vals = stats['post_vals']
            ax.hist(post_vals, bins=100, alpha=0.7, edgecolor='black', color='#2ecc71', density=True)
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Dead threshold')
            ax.set_xlabel('Post-activation value')
            ax.set_ylabel('Density')
            ax.set_title(f'{layer_name.upper()} Post-Activation (GELU)')
            ax.legend()
            ax.grid(alpha=0.3)
            
            dead_rate = stats['dead_rate']
            ax.text(0.02, 0.98, f'Dead: {dead_rate:.1f}%\nMean: {post_vals.mean():.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_saturation.png'), dpi=150)
    plt.close()
    print(f"✅ Activation saturation saved")
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    layers = list(saturation_stats.keys())
    sat_rates = [saturation_stats[l]['saturation_rate_pre'] for l in layers]
    dead_rates = [saturation_stats[l]['dead_rate'] for l in layers]
    
    x_pos = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x_pos - width/2, sat_rates, width, label='Saturation (pre)', 
           color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, dead_rates, width, label='Dead neurons (post)', 
           color='#95a5a6', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('CNN Layer Health: Saturation & Dead Neurons (4-Layer Architecture)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='Warning (30%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'saturation_summary.png'), dpi=150)
    plt.close()
    print(f"✅ Saturation summary saved")
    
    # ==================== [3/3] SAMPLE VISUALIZATIONS ====================
    print("\n🖼️  Generating sample filter visualizations...")
    
    # Take first sample for visualization
    obs, _ = env.reset()
    
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
        
        # Process all 4 layers and visualize each one
        x = image
        for layer_idx in range(num_layers):
            x = features_extractor.conv_layers[layer_idx](x)
            x = features_extractor.bn_layers[layer_idx](x)
            x = torch.nn.functional.gelu(x)
            
            conv_out = x[0].cpu().numpy()
            
            # Visualize this layer (first 16 channels if available)
            layer_name = f'Conv{layer_idx+1}'
            visualize_conv_layer(conv_out, f'{layer_name} Output', 
                                os.path.join(viz_dir, f'{layer_name.lower()}_sample.png'), 
                                num_channels=min(16, conv_out.shape[0]))
        
        print(f"✅ Sample visualizations saved in {viz_dir}")
    
    # ==================== SAVE STATISTICS ====================
    stats_csv = os.path.join(output_dir, 'cnn_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'num_channels', 'dead_channels', 'dead_pct', 
                        'saturation_rate', 'dead_neuron_rate', 'status'])
        
        for i in range(num_layers):
            layer_name = f'conv{i+1}'
            stats_list = all_layer_stats[i]
            
            num_channels = len(stats_list)
            dead_count = sum(1 for s in stats_list if s['is_dead'])
            dead_pct = (dead_count / num_channels) * 100
            
            if layer_name in saturation_stats:
                sat_rate = saturation_stats[layer_name]['saturation_rate_pre']
                dead_rate = saturation_stats[layer_name]['dead_rate']
            else:
                sat_rate = 0
                dead_rate = 0
            
            if dead_pct > 30:
                status = 'UNDERUTILIZED'
            elif sat_rate > 30:
                status = 'HIGH_SATURATION'
            elif dead_rate > 30:
                status = 'TOO_MANY_DEAD'
            else:
                status = 'HEALTHY'
            
            writer.writerow([
                layer_name,
                num_channels,
                dead_count,
                f'{dead_pct:.1f}',
                f'{sat_rate:.1f}',
                f'{dead_rate:.1f}',
                status
            ])
    
    print(f"✅ Statistics saved: {stats_csv}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 CNN LAYERS SUMMARY (4-Layer Architecture)")
    print("="*80)
    
    for i in range(num_layers):
        layer_name = f'Conv{i+1}'
        stats_list = all_layer_stats[i]
        dead_count = sum(1 for s in stats_list if s['is_dead'])
        print(f"{layer_name}: {dead_count}/{len(stats_list)} dead channels ({dead_count/len(stats_list)*100:.1f}%)")
    
    layers = list(saturation_stats.keys())
    high_saturation = [name for name in layers if saturation_stats[name]['saturation_rate_pre'] > 30]
    high_dead = [name for name in layers if saturation_stats[name]['dead_rate'] > 30]
    
    if high_saturation:
        print(f"\n⚠️  High saturation (>30%): {', '.join(high_saturation)}")
    if high_dead:
        print(f"⚠️  High dead neurons (>30%): {', '.join(high_dead)}")
    
    if not high_saturation and not high_dead:
        print("\n✅ All CNN layers are healthy!")


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
    
    for i in range(num_channels, 16):
        axes[i // 4, i % 4].axis('off')
    
    plt.suptitle(f'{layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()