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
    
    conv1_activations = []
    conv2_activations = []
    conv3_activations = [] if features_extractor.has_conv3 else None
    
    activation_data = {
        'conv1_pre': [],
        'conv1_post': [],
        'conv2_pre': [],
        'conv2_post': [],
        'conv3_pre': [],
        'conv3_post': []
    }
    
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
            
            # Conv1
            x_pre = features_extractor.conv1(image)
            x_pre = features_extractor.bn1(x_pre)
            activation_data['conv1_pre'].extend(x_pre.flatten().cpu().numpy())
            x = torch.nn.functional.gelu(x_pre)
            activation_data['conv1_post'].extend(x.flatten().cpu().numpy())
            conv1_activations.append(x.detach().cpu().numpy()[0])
            
            # Conv2
            x_pre = features_extractor.conv2(x)
            x_pre = features_extractor.bn2(x_pre)
            x_pre = features_extractor.dropout2(x_pre)
            activation_data['conv2_pre'].extend(x_pre.flatten().cpu().numpy())
            x = torch.nn.functional.gelu(x_pre)
            activation_data['conv2_post'].extend(x.flatten().cpu().numpy())
            conv2_activations.append(x.detach().cpu().numpy()[0])
            
            # Conv3
            if features_extractor.has_conv3:
                identity = features_extractor.residual_proj(x)
                
                x_local = features_extractor.conv3_local(x)
                x_local = features_extractor.bn3_local(x_local)
                
                x_global = features_extractor.conv3_global(x)
                x_global = features_extractor.bn3_global(x_global)
                
                x_pre = torch.cat([x_local, x_global], dim=1)
                x_pre = features_extractor.dropout3(x_pre)
                activation_data['conv3_pre'].extend(x_pre.flatten().cpu().numpy())
                x_combined = torch.nn.functional.gelu(x_pre)
                activation_data['conv3_post'].extend(x_combined.flatten().cpu().numpy())
                x = x_combined + identity
                conv3_activations.append(x.detach().cpu().numpy()[0])
        
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{num_samples} samples")
    
    # Convert to numpy
    conv1_activations = np.array(conv1_activations)
    conv2_activations = np.array(conv2_activations)
    if conv3_activations is not None:
        conv3_activations = np.array(conv3_activations)
    
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
    
    conv1_stats = analyze_channels(conv1_activations, 'Conv1')
    conv2_stats = analyze_channels(conv2_activations, 'Conv2')
    
    if conv3_activations is not None:
        conv3_stats = analyze_channels(conv3_activations, 'Conv3')
    
    # Plot specialization
    if conv3_activations is not None:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Conv1 - Mean activation
    channels_conv1 = [s['channel'] for s in conv1_stats]
    means_conv1 = [s['mean'] for s in conv1_stats]
    dead_conv1 = [s['is_dead'] for s in conv1_stats]
    colors_conv1 = ['red' if d else 'green' for d in dead_conv1]
    
    axes[0, 0].bar(channels_conv1, means_conv1, color=colors_conv1, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean Activation')
    axes[0, 0].set_title('Conv1 - Channel Activity (Red = Dead)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Conv2 - Mean activation
    channels_conv2 = [s['channel'] for s in conv2_stats]
    means_conv2 = [s['mean'] for s in conv2_stats]
    dead_conv2 = [s['is_dead'] for s in conv2_stats]
    colors_conv2 = ['red' if d else 'green' for d in dead_conv2]
    
    axes[0, 1].bar(channels_conv2, means_conv2, color=colors_conv2, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Mean Activation')
    axes[0, 1].set_title('Conv2 - Channel Activity (Red = Dead)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Conv3 (if exists)
    if conv3_activations is not None:
        channels_conv3 = [s['channel'] for s in conv3_stats]
        means_conv3 = [s['mean'] for s in conv3_stats]
        dead_conv3 = [s['is_dead'] for s in conv3_stats]
        colors_conv3 = ['red' if d else 'green' for d in dead_conv3]
        
        axes[0, 2].bar(channels_conv3, means_conv3, color=colors_conv3, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Channel')
        axes[0, 2].set_ylabel('Mean Activation')
        axes[0, 2].set_title('Conv3 - Channel Activity (Red = Dead)')
        axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Sparsity plots
    sparsity_conv1 = [s['sparsity'] for s in conv1_stats]
    axes[1, 0].bar(channels_conv1, sparsity_conv1, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Sparsity (% near-zero)')
    axes[1, 0].set_title('Conv1 - Sparsity per Channel')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    sparsity_conv2 = [s['sparsity'] for s in conv2_stats]
    axes[1, 1].bar(channels_conv2, sparsity_conv2, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Sparsity (% near-zero)')
    axes[1, 1].set_title('Conv2 - Sparsity per Channel')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    if conv3_activations is not None:
        sparsity_conv3 = [s['sparsity'] for s in conv3_stats]
        axes[1, 2].bar(channels_conv3, sparsity_conv3, color='#3498db', alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Channel')
        axes[1, 2].set_ylabel('Sparsity (% near-zero)')
        axes[1, 2].set_title('Conv3 - Sparsity per Channel')
        axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_specialization.png'), dpi=150)
    plt.close()
    print(f"✅ Channel specialization saved")
    
    # ==================== [2/3] ACTIVATION SATURATION ====================
    print("\n🔥 Analyzing activation saturation...")
    
    saturation_stats = {}
    for layer_name in ['conv1', 'conv2', 'conv3']:
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
    
    # Plot saturation
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    for idx, layer_name in enumerate(['conv1', 'conv2', 'conv3']):
        if layer_name in saturation_stats:
            stats = saturation_stats[layer_name]
            
            # Pre-activation
            ax = axes[idx, 0]
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
            ax = axes[idx, 1]
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
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
    ax.set_title('CNN Layer Health: Saturation & Dead Neurons')
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
        
        # Conv1
        x = features_extractor.conv1(image)
        x = features_extractor.bn1(x)
        x = torch.nn.functional.gelu(x)
        conv1_out = x[0].cpu().numpy()
        
        # Conv2
        x = features_extractor.conv2(x)
        x = features_extractor.bn2(x)
        x = features_extractor.dropout2(x)
        x = torch.nn.functional.gelu(x)
        conv2_out = x[0].cpu().numpy()
        
        # Visualize Conv1 (first 16 channels)
        visualize_conv_layer(conv1_out, 'Conv1 Output', 
                            os.path.join(viz_dir, 'conv1_sample.png'), num_channels=16)
        
        # Visualize Conv2 (first 16 channels)
        visualize_conv_layer(conv2_out, 'Conv2 Output',
                            os.path.join(viz_dir, 'conv2_sample.png'), num_channels=16)
        
        print(f"✅ Sample visualizations saved in {viz_dir}")
    
    # ==================== SAVE STATISTICS ====================
    stats_csv = os.path.join(output_dir, 'cnn_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'num_channels', 'dead_channels', 'dead_pct', 
                        'saturation_rate', 'dead_neuron_rate', 'status'])
        
        for layer_name in ['conv1', 'conv2', 'conv3']:
            if layer_name == 'conv1':
                stats_list = conv1_stats
            elif layer_name == 'conv2':
                stats_list = conv2_stats
            elif layer_name == 'conv3' and conv3_activations is not None:
                stats_list = conv3_stats
            else:
                continue
            
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
    print("📋 CNN LAYERS SUMMARY")
    print("="*80)
    
    dead_conv1_count = sum(dead_conv1)
    dead_conv2_count = sum(dead_conv2)
    
    print(f"Conv1: {dead_conv1_count}/{len(conv1_stats)} dead channels ({dead_conv1_count/len(conv1_stats)*100:.1f}%)")
    print(f"Conv2: {dead_conv2_count}/{len(conv2_stats)} dead channels ({dead_conv2_count/len(conv2_stats)*100:.1f}%)")
    
    if conv3_activations is not None:
        dead_conv3_count = sum(dead_conv3)
        print(f"Conv3: {dead_conv3_count}/{len(conv3_stats)} dead channels ({dead_conv3_count/len(conv3_stats)*100:.1f}%)")
    
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