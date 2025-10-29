"""
📊 CHANNEL ANALYSIS MODULE
- analyze_channel_specialization: Analiza specjalizacji kanałów CNN
- analyze_activation_saturation: Analiza saturacji aktywacji (GELU)
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt


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
    conv3_activations = [] if features_extractor.has_conv3 else None
    
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
            conv1_activations.append(x.detach().cpu().numpy()[0])

            # Conv2
            x = features_extractor.conv2(x)
            x = features_extractor.bn2(x)
            x = features_extractor.dropout2(x)
            x = torch.nn.functional.gelu(x)
            conv2_activations.append(x.detach().cpu().numpy()[0])

            # Conv3 (opcjonalna)
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
                conv3_activations.append(x.detach().cpu().numpy()[0])
    
    # Stack: [num_samples, channels, height, width]
    conv1_activations = np.array(conv1_activations)
    conv2_activations = np.array(conv2_activations)
    if conv3_activations is not None:
        conv3_activations = np.array(conv3_activations)
    
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
    
    if conv3_activations is not None:
        conv3_stats = analyze_channels(conv3_activations, 'Conv3')
    
    # Wizualizacja - 2x3 grid jeśli mamy Conv3, 2x2 jeśli nie
    if conv3_activations is not None:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
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
    
    # Plot 3: Conv3 - Mean activation per channel (if exists)
    if conv3_activations is not None:
        channels_conv3 = [s['channel'] for s in conv3_stats]
        means_conv3 = [s['mean'] for s in conv3_stats]
        dead_conv3 = [s['is_dead'] for s in conv3_stats]
        
        colors_conv3 = ['red' if d else 'green' for d in dead_conv3]
        axes[0, 2].bar(channels_conv3, means_conv3, color=colors_conv3, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Channel')
        axes[0, 2].set_ylabel('Mean Activation')
        axes[0, 2].set_title(f'Conv3 - Channel Activity (Red = Dead)')
        axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Sparsity (ile neuronów ~0) - Conv1
    sparsity_conv1 = [s['sparsity'] for s in conv1_stats]
    axes[1, 0].bar(channels_conv1, sparsity_conv1, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Sparsity (% near-zero)')
    axes[1, 0].set_title('Conv1 - Sparsity per Channel')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 5: Sparsity Conv2
    sparsity_conv2 = [s['sparsity'] for s in conv2_stats]
    axes[1, 1].bar(channels_conv2, sparsity_conv2, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Sparsity (% near-zero)')
    axes[1, 1].set_title('Conv2 - Sparsity per Channel')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Plot 6: Sparsity Conv3 (if exists)
    if conv3_activations is not None:
        sparsity_conv3 = [s['sparsity'] for s in conv3_stats]
        axes[1, 2].bar(channels_conv3, sparsity_conv3, color='#3498db', alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Channel')
        axes[1, 2].set_ylabel('Sparsity (% near-zero)')
        axes[1, 2].set_title('Conv3 - Sparsity per Channel')
        axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    specialization_path = os.path.join(output_dir, 'channel_specialization.png')
    plt.savefig(specialization_path, dpi=150)
    plt.close()
    
    # Raport
    dead_conv1_count = sum(dead_conv1)
    dead_conv2_count = sum(dead_conv2)
    
    print(f"\n📊 Channel Specialization Report:")
    print(f"   Conv1: {dead_conv1_count}/{len(conv1_stats)} dead channels ({dead_conv1_count/len(conv1_stats)*100:.1f}%)")
    print(f"   Conv2: {dead_conv2_count}/{len(conv2_stats)} dead channels ({dead_conv2_count/len(conv2_stats)*100:.1f}%)")
    
    if conv3_activations is not None:
        dead_conv3_count = sum(dead_conv3)
        print(f"   Conv3: {dead_conv3_count}/{len(conv3_stats)} dead channels ({dead_conv3_count/len(conv3_stats)*100:.1f}%)")
    
    print(f"\n   ⚠️ Jeśli >30% kanałów jest dead, sieć jest UNDERUTILIZED!")
    
    print(f"\n✅ Channel specialization zapisana: {specialization_path}")


def analyze_activation_saturation(model, env, output_dir, num_samples=100):
    """
    🔥 ACTIVATION SATURATION ANALYSIS
    
    Sprawdza:
    - % neuronów w saturated region (GELU: |x| > 3)
    - Dead ReLU regions
    - Mean/std of pre-activation vs post-activation
    - Histogram of activation values
    """
    print("\n" + "="*80)
    print("🔥 ACTIVATION SATURATION ANALYSIS")
    print("="*80)
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # Collect pre/post activations
    activation_data = {
        'conv1_pre': [],
        'conv1_post': [],
        'conv2_pre': [],
        'conv2_post': [],
        'conv3_pre': [],
        'conv3_post': []
    }
    
    print(f"\n📊 Collecting activation data from {num_samples} samples...")
    
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
            
            # Conv2
            x_pre = features_extractor.conv2(x)
            x_pre = features_extractor.bn2(x_pre)
            x_pre = features_extractor.dropout2(x_pre)
            activation_data['conv2_pre'].extend(x_pre.flatten().cpu().numpy())
            x = torch.nn.functional.gelu(x_pre)
            activation_data['conv2_post'].extend(x.flatten().cpu().numpy())
            
            # Conv3 (if exists)
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
                x = x_combined + identity  # Residual
        
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{num_samples} samples")
    
    # Convert to numpy arrays
    for key in activation_data.keys():
        if activation_data[key]:
            activation_data[key] = np.array(activation_data[key])
    
    # Analysis
    print("\n📈 Analyzing saturation...")
    
    saturation_stats = {}
    for layer_name in ['conv1', 'conv2', 'conv3']:
        pre_key = f'{layer_name}_pre'
        post_key = f'{layer_name}_post'
        
        if len(activation_data[pre_key]) > 0:
            pre_vals = activation_data[pre_key]
            post_vals = activation_data[post_key] if len(activation_data[post_key]) > 0 else pre_vals
            
            # GELU saturation: |x| > 3
            saturated_pre = np.abs(pre_vals) > 3
            saturated_post = np.abs(post_vals) > 3
            
            # Dead neurons: output ≈ 0
            dead_threshold = 0.01
            dead_post = np.abs(post_vals) < dead_threshold
            
            saturation_stats[layer_name] = {
                'pre_mean': pre_vals.mean(),
                'pre_std': pre_vals.std(),
                'post_mean': post_vals.mean(),
                'post_std': post_vals.std(),
                'saturation_rate_pre': saturated_pre.mean() * 100,
                'saturation_rate_post': saturated_post.mean() * 100,
                'dead_rate': dead_post.mean() * 100,
                'pre_vals': pre_vals,
                'post_vals': post_vals
            }
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot 1-3: Pre-activation histograms
    for idx, layer_name in enumerate(['conv1', 'conv2', 'conv3']):
        if layer_name in saturation_stats:
            ax = axes[idx, 0]
            pre_vals = saturation_stats[layer_name]['pre_vals']
            
            ax.hist(pre_vals, bins=100, alpha=0.7, edgecolor='black', color='#3498db', density=True)
            ax.axvline(-3, color='red', linestyle='--', alpha=0.7, label='Saturation threshold')
            ax.axvline(3, color='red', linestyle='--', alpha=0.7)
            ax.axvline(0, color='green', linestyle='--', alpha=0.7, label='Zero')
            
            ax.set_xlabel('Pre-activation value')
            ax.set_ylabel('Density')
            ax.set_title(f'{layer_name.upper()} Pre-Activation Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Add stats text
            sat_rate = saturation_stats[layer_name]['saturation_rate_pre']
            ax.text(0.02, 0.98, f'Saturation: {sat_rate:.1f}%\nMean: {pre_vals.mean():.3f}\nStd: {pre_vals.std():.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4-6: Post-activation histograms
    for idx, layer_name in enumerate(['conv1', 'conv2', 'conv3']):
        if layer_name in saturation_stats:
            ax = axes[idx, 1]
            post_vals = saturation_stats[layer_name]['post_vals']
            
            ax.hist(post_vals, bins=100, alpha=0.7, edgecolor='black', color='#2ecc71', density=True)
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Dead threshold')
            
            ax.set_xlabel('Post-activation value')
            ax.set_ylabel('Density')
            ax.set_title(f'{layer_name.upper()} Post-Activation Distribution (after GELU)')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Add stats text
            dead_rate = saturation_stats[layer_name]['dead_rate']
            ax.text(0.02, 0.98, f'Dead: {dead_rate:.1f}%\nMean: {post_vals.mean():.3f}\nStd: {post_vals.std():.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    saturation_path = os.path.join(output_dir, 'activation_saturation.png')
    plt.savefig(saturation_path, dpi=150)
    plt.close()
    print(f"✅ Activation saturation analysis saved: {saturation_path}")
    
    # Summary plot: Saturation rates
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = list(saturation_stats.keys())
    sat_rates_pre = [saturation_stats[l]['saturation_rate_pre'] for l in layers]
    dead_rates = [saturation_stats[l]['dead_rate'] for l in layers]
    
    x_pos = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x_pos - width/2, sat_rates_pre, width, label='Saturation (pre-act)', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, dead_rates, width, label='Dead neurons (post-act)', color='#95a5a6', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Activation Saturation & Dead Neurons per Layer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='Warning threshold (30%)')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'activation_saturation_summary.png')
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"✅ Saturation summary saved: {summary_path}")
    
    # Save statistics to CSV
    stats_csv = os.path.join(output_dir, 'activation_saturation_stats.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'pre_mean', 'pre_std', 'post_mean', 'post_std', 
                        'saturation_rate_pre_%', 'saturation_rate_post_%', 'dead_rate_%', 'status'])
        
        for layer_name in layers:
            stats = saturation_stats[layer_name]
            
            # Determine status
            if stats['dead_rate'] > 30:
                status = 'TOO_MANY_DEAD'
            elif stats['saturation_rate_pre'] > 30:
                status = 'HIGH_SATURATION'
            elif stats['dead_rate'] < 10 and stats['saturation_rate_pre'] < 10:
                status = 'HEALTHY'
            else:
                status = 'MODERATE'
            
            writer.writerow([
                layer_name,
                stats['pre_mean'],
                stats['pre_std'],
                stats['post_mean'],
                stats['post_std'],
                stats['saturation_rate_pre'],
                stats['saturation_rate_post'],
                stats['dead_rate'],
                status
            ])
    
    print(f"✅ Saturation statistics saved: {stats_csv}")
    
    # Summary
    print("\n" + "="*80)
    print("📋 ACTIVATION SATURATION SUMMARY")
    print("="*80)
    
    high_saturation = [name for name in layers if saturation_stats[name]['saturation_rate_pre'] > 30]
    high_dead = [name for name in layers if saturation_stats[name]['dead_rate'] > 30]
    
    if high_saturation:
        print(f"⚠️  High saturation (>30%) in: {', '.join(high_saturation)}")
    else:
        print(f"✅ No high saturation detected!")
    
    if high_dead:
        print(f"⚠️  High dead neuron rate (>30%) in: {', '.join(high_dead)}")
    else:
        print(f"✅ No excessive dead neurons!")
    
    for layer_name in layers:
        stats = saturation_stats[layer_name]
        print(f"\n{layer_name}:")
        print(f"  Pre-activation:  mean={stats['pre_mean']:.3f}, std={stats['pre_std']:.3f}, saturation={stats['saturation_rate_pre']:.1f}%")
        print(f"  Post-activation: mean={stats['post_mean']:.3f}, std={stats['post_std']:.3f}, dead={stats['dead_rate']:.1f}%")
