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
    ✅ FIXED: Obsługuje tryb z i bez scalarów
    """
    print("\n" + "="*80)
    print("🔍 CNN LAYERS COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # 🎯 DETECT SCALARS MODE
    scalars_enabled = features_extractor.scalars_enabled
    
    # Utwórz podkatalogi
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # ==================== ZBIERANIE DANYCH ====================
    print(f"\n📊 Collecting data from {num_samples} samples...")
    
    conv1_activations = []
    conv2_activations = []
    conv3_activations = []  # 🔥 NEW!
    
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
            
            # 🔥 HYBRID CNN: 5×5 → 3×3 → 3×3 (use SiLU like in forward pass!)
            
            # Conv1 (5×5)
            x_pre = features_extractor.conv1(image)
            x_pre = features_extractor.norm1(x_pre)
            activation_data['conv1_pre'].extend(x_pre.flatten().cpu().numpy())
            x = torch.nn.functional.silu(x_pre)  # ✅ Use SiLU like in model!
            activation_data['conv1_post'].extend(x.flatten().cpu().numpy())
            conv1_activations.append(x.detach().cpu().numpy()[0])
            
            # Conv2 (3×3)
            x_pre = features_extractor.conv2(x)
            x_pre = features_extractor.norm2(x_pre)
            activation_data['conv2_pre'].extend(x_pre.flatten().cpu().numpy())
            x = torch.nn.functional.silu(x_pre)  # ✅ Use SiLU
            activation_data['conv2_post'].extend(x.flatten().cpu().numpy())
            conv2_activations.append(x.detach().cpu().numpy()[0])
            
            # Conv3 (3×3 strided, NO MAXPOOL!) 🔥 NEW!
            x_pre = features_extractor.conv3(x)
            x_pre = features_extractor.norm3(x_pre)
            activation_data['conv3_pre'].extend(x_pre.flatten().cpu().numpy())
            x = torch.nn.functional.silu(x_pre)  # ✅ Use SiLU
            x = features_extractor.dropout3(x)
            activation_data['conv3_post'].extend(x.flatten().cpu().numpy())
            conv3_activations.append(x.detach().cpu().numpy()[0])
        
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{num_samples} samples")
    
    # Convert to numpy
    conv1_activations = np.array(conv1_activations)
    conv2_activations = np.array(conv2_activations)
    conv3_activations = np.array(conv3_activations)  # 🔥 NEW!
    
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
    conv3_stats = analyze_channels(conv3_activations, 'Conv3')  # 🔥 NEW!
    
    # Plot specialization (2x3 grid dla Conv1, Conv2, Conv3) - teraz 3 warstwy!
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
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
    
    # Conv3 - Mean activation 🔥 NEW!
    channels_conv3 = [s['channel'] for s in conv3_stats]
    means_conv3 = [s['mean'] for s in conv3_stats]
    dead_conv3 = [s['is_dead'] for s in conv3_stats]
    colors_conv3 = ['red' if d else 'green' for d in dead_conv3]
    
    axes[0, 2].bar(channels_conv3, means_conv3, color=colors_conv3, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Channel')
    axes[0, 2].set_ylabel('Mean Activation')
    axes[0, 2].set_title('Conv3 (Strided) - Channel Activity (Red = Dead)')
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
    
    # Conv3 sparsity 🔥 NEW!
    sparsity_conv3 = [s['sparsity'] for s in conv3_stats]
    axes[1, 2].bar(channels_conv3, sparsity_conv3, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Channel')
    axes[1, 2].set_ylabel('Sparsity (% near-zero)')
    axes[1, 2].set_title('Conv3 (Strided) - Sparsity per Channel')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_specialization.png'), dpi=150)
    plt.close()
    print(f"✅ Channel specialization saved")
    
    # ==================== [2/3] ACTIVATION SATURATION ====================
    print("\n🔥 Analyzing activation saturation...")
    
    saturation_stats = {}
    for layer_name in ['conv1', 'conv2', 'conv3']:  # 🔥 Include conv3!
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, layer_name in enumerate(['conv1', 'conv2']):
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
    
    # ==================== ✨ [NEW] TWO CNN PATHS ANALYSIS ====================
    print("\n✨ Analyzing CNN paths...")
    
    attended_cnn_activations = []
    direct_cnn_activations = []
    
    for sample_idx in range(min(50, num_samples)):
        obs, _ = env.reset()
        
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
            
            # Forward pass to get CNN paths
            x = image
            x = features_extractor.conv1(x)
            x = features_extractor.norm1(x)
            x = torch.nn.functional.silu(x)
            
            x = features_extractor.conv2(x)
            x = features_extractor.norm2(x)
            x = torch.nn.functional.silu(x)
            
            x = features_extractor.conv3(x)
            x = features_extractor.norm3(x)
            x = torch.nn.functional.silu(x)
            x = features_extractor.dropout3(x)
            
            # 🔥 NO MAXPOOL! Downsampling done by strided conv
            
            x = features_extractor.pos_encoding(x)
            
            spatial_features = x.flatten(2).transpose(1, 2)
            spatial_features_flat = spatial_features.flatten(1)
            
            # PATH 1: Attention (only if scalars enabled)
            if scalars_enabled and features_extractor.scalar_network is not None:
                normalized = features_extractor.cnn_prenorm(spatial_features)
                
                scalars = torch.cat([
                    obs_tensor['direction'],
                    obs_tensor['dx_head'],
                    obs_tensor['dy_head'],
                    obs_tensor['front_coll'],
                    obs_tensor['left_coll'],
                    obs_tensor['right_coll'],
                    obs_tensor['snake_length']
                ], dim=-1)
                scalars = features_extractor.scalar_input_dropout(scalars)
                scalar_features = features_extractor.scalar_network(scalars)
                
                attended = features_extractor.cross_attention(normalized, scalar_features)
                attended_cnn_activations.append(attended.detach().cpu().numpy()[0])
            
            # PATH 2: Direct CNN (Skip Connection) - only if scalars enabled
            if scalars_enabled and features_extractor.cnn_direct is not None:
                direct = features_extractor.cnn_direct(spatial_features_flat)
                direct_cnn_activations.append(direct.detach().cpu().numpy()[0])
            elif not scalars_enabled and hasattr(features_extractor, 'cnn_only_direct') and features_extractor.cnn_only_direct is not None:
                # CNN-only: use cnn_only_direct compression
                direct = features_extractor.cnn_only_direct(spatial_features_flat)
                direct_cnn_activations.append(direct.detach().cpu().numpy()[0])
            else:
                # Fallback for old models: detect expected fusion input size
                # We need to compute attended_cnn first to know its size
                attended_size = attended_cnn_activations[-1].shape[0] if attended_cnn_activations else 0
                scalar_size = 0  # No scalars in CNN-only mode
                
                expected_fusion_size = features_extractor.fusion[0].in_features
                current_size = attended_size + spatial_features_flat.shape[-1] + scalar_size
                
                if current_size > expected_fusion_size:
                    # Need compression
                    direct_size = expected_fusion_size - attended_size - scalar_size
                    if not hasattr(features_extractor, '_temp_direct_projection'):
                        features_extractor._temp_direct_projection = torch.nn.Linear(
                            spatial_features_flat.shape[-1], 
                            direct_size,
                            device=spatial_features_flat.device
                        )
                    direct = features_extractor._temp_direct_projection(spatial_features_flat)
                    direct_cnn_activations.append(direct.detach().cpu().numpy()[0])
                else:
                    direct_cnn_activations.append(spatial_features_flat.detach().cpu().numpy()[0])
    
    attended_cnn_activations = np.array(attended_cnn_activations) if attended_cnn_activations else np.array([])
    direct_cnn_activations = np.array(direct_cnn_activations) if direct_cnn_activations else np.array([])
    
    # ✅ CONDITIONAL: Only visualize with proper labels
    if len(attended_cnn_activations) > 0 or len(direct_cnn_activations) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        attended_mean = np.abs(attended_cnn_activations).mean(axis=0) if len(attended_cnn_activations) > 0 else np.array([])
        direct_mean = np.abs(direct_cnn_activations).mean(axis=0) if len(direct_cnn_activations) > 0 else np.array([])
        
        # 1. Mean activation magnitude
        ax = axes[0, 0]
        
        if len(attended_mean) > 0 and len(direct_mean) > 0:
            num_features_viz = min(100, len(attended_mean), len(direct_mean))
            x_pos = np.arange(num_features_viz)
            width = 0.35
            ax.bar(x_pos - width/2, attended_mean[:num_features_viz], width, label='Attended CNN (via Attention)', alpha=0.8, color='#3498db')
            ax.bar(x_pos + width/2, direct_mean[:num_features_viz], width, label='Direct CNN (Skip Connection)', alpha=0.8, color='#e74c3c')
            ax.set_xlabel(f'Feature Index (first {num_features_viz})')
            ax.set_title('Activation Magnitude: Attention vs Skip')
        elif len(direct_mean) > 0:
            num_features_viz = min(100, len(direct_mean))
            x_pos = np.arange(num_features_viz)
            ax.bar(x_pos, direct_mean[:num_features_viz], alpha=0.8, color='#e74c3c', label='Raw CNN Features')
            ax.set_xlabel(f'Feature Index (first {num_features_viz})')
            ax.set_title('CNN Features (Raw Spatial)')
        
        ax.set_ylabel('Mean Abs Activation')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Sparsity comparison
        ax = axes[0, 1]
        if len(attended_cnn_activations) > 0 and len(direct_cnn_activations) > 0:
            sparsity_attended = (np.abs(attended_cnn_activations) < 0.01).sum(axis=0) / len(attended_cnn_activations)
            sparsity_direct = (np.abs(direct_cnn_activations) < 0.01).sum(axis=0) / len(direct_cnn_activations)
            
            num_sparsity_viz = min(256, len(sparsity_attended), len(sparsity_direct))
            ax.scatter(sparsity_attended[:num_sparsity_viz], sparsity_direct[:num_sparsity_viz], alpha=0.6, s=50)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal sparsity')
            ax.set_xlabel('Sparsity (Attended CNN)')
            ax.set_ylabel('Sparsity (Direct CNN)')
            ax.set_title('Feature Sparsity Comparison')
            ax.legend()
        elif len(direct_cnn_activations) > 0:
            sparsity_direct = (np.abs(direct_cnn_activations) < 0.01).sum(axis=0) / len(direct_cnn_activations)
            ax.hist(sparsity_direct, bins=50, alpha=0.8, color='#e74c3c', edgecolor='black')
            ax.set_xlabel('Sparsity (% near-zero)')
            ax.set_ylabel('Frequency')
            ax.set_title('CNN Features Sparsity Distribution')
        
        ax.grid(alpha=0.3)
        
        # 3. Standard deviation comparison
        ax = axes[1, 0]
        if len(attended_cnn_activations) > 0 and len(direct_cnn_activations) > 0:
            std_attended = attended_cnn_activations.std(axis=0)
            std_direct = direct_cnn_activations.std(axis=0)
            
            num_std_viz = min(100, len(std_attended), len(std_direct))
            ax.plot(std_attended[:num_std_viz], label='Attended CNN', color='#3498db', linewidth=2)
            ax.plot(std_direct[:num_std_viz], label='Direct CNN', color='#e74c3c', linewidth=2)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Std Dev (across samples)')
            ax.set_title('Feature Variability Comparison')
            ax.legend()
        elif len(direct_cnn_activations) > 0:
            std_direct = direct_cnn_activations.std(axis=0)
            num_std_viz = min(100, len(std_direct))
            ax.plot(std_direct[:num_std_viz], color='#e74c3c', linewidth=2, label='Raw CNN')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Std Dev (across samples)')
            ax.set_title('CNN Features Variability')
            ax.legend()
        
        ax.grid(alpha=0.3)
        
        # 4. Distribution comparison
        ax = axes[1, 1]
        if len(attended_cnn_activations) > 0 and len(direct_cnn_activations) > 0:
            ax.hist(attended_cnn_activations.flatten(), bins=50, alpha=0.6, label='Attended CNN', color='#3498db', density=True)
            ax.hist(direct_cnn_activations.flatten(), bins=50, alpha=0.6, label='Direct CNN', color='#e74c3c', density=True)
            ax.set_title('Overall Distribution Comparison')
        elif len(direct_cnn_activations) > 0:
            ax.hist(direct_cnn_activations.flatten(), bins=50, alpha=0.8, color='#e74c3c', density=True, edgecolor='black')
            ax.set_title('Raw CNN Features Distribution')
        
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cnn_paths_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ CNN paths comparison saved: cnn_paths_comparison.png")
        
        # Summary statistics
        print(f"\n📊 CNN PATHS STATISTICS:")
        if len(attended_cnn_activations) > 0:
            print(f"  Attended CNN (via Attention):")
            print(f"    - Mean activation: {attended_cnn_activations.mean():.4f}")
            print(f"    - Std activation:  {attended_cnn_activations.std():.4f}")
            print(f"    - Sparsity: {(np.abs(attended_cnn_activations) < 0.01).sum() / attended_cnn_activations.size * 100:.1f}%")
        
        if len(direct_cnn_activations) > 0:
            path_label = "Direct CNN (Skip Connection)" if scalars_enabled else "Direct CNN (Raw Spatial)"
            print(f"  {path_label}:")
            print(f"    - Mean activation: {direct_cnn_activations.mean():.4f}")
            print(f"    - Std activation:  {direct_cnn_activations.std():.4f}")
            print(f"    - Sparsity: {(np.abs(direct_cnn_activations) < 0.01).sum() / direct_cnn_activations.size * 100:.1f}%")
    else:
        # CNN-ONLY or no data: skip comparison
        print(f"\n⚠️  CNN PATHS: Skipping comparison (no scalars enabled)")
        
        # Summary statistics for Direct CNN only
        if len(direct_cnn_activations) > 0:
            print(f"\n📊 RAW CNN FEATURES STATISTICS:")
            print(f"    - Mean activation: {direct_cnn_activations.mean():.4f}")
            print(f"    - Std activation:  {direct_cnn_activations.std():.4f}")
            print(f"    - Sparsity: {(np.abs(direct_cnn_activations) < 0.01).sum() / direct_cnn_activations.size * 100:.1f}%")
    
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
        
        # 🔥 HYBRID CNN: 5×5 → 3×3 → 3×3
        
        # Conv1 (5×5)
        x = features_extractor.conv1(image)
        x = features_extractor.norm1(x)
        x = torch.nn.functional.silu(x)
        conv1_out = x[0].cpu().numpy()
        
        # Conv2 (3×3)
        x = features_extractor.conv2(x)
        x = features_extractor.norm2(x)
        x = torch.nn.functional.silu(x)
        conv2_out = x[0].cpu().numpy()
        
        # Conv3 (3×3 strided, NO MAXPOOL!) 🔥 NEW!
        x = features_extractor.conv3(x)
        x = features_extractor.norm3(x)
        x = torch.nn.functional.silu(x)
        x = features_extractor.dropout3(x)
        conv3_out = x[0].cpu().numpy()
        
        # Visualize Conv1 (first 16 channels)
        visualize_conv_layer(conv1_out, 'Conv1 Output (5×5)', 
                            os.path.join(viz_dir, 'conv1_sample.png'), num_channels=16)
        
        # Visualize Conv2 (first 16 channels)
        visualize_conv_layer(conv2_out, 'Conv2 Output (3×3)',
                            os.path.join(viz_dir, 'conv2_sample.png'), num_channels=16)
        
        # Visualize Conv3 (first 16 channels) 🔥 NEW!
        visualize_conv_layer(conv3_out, 'Conv3 Output (3×3 Strided)',
                            os.path.join(viz_dir, 'conv3_sample.png'), num_channels=16)
        
        print(f"✅ Sample visualizations saved in {viz_dir}")
    
    # ==================== SAVE STATISTICS ====================
    stats_csv = os.path.join(output_dir, 'cnn_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'num_channels', 'dead_channels', 'dead_pct', 
                        'saturation_rate', 'dead_neuron_rate', 'status'])
        
        for layer_name in ['conv1', 'conv2', 'conv3']:  # 🔥 Include conv3!
            if layer_name == 'conv1':
                stats_list = conv1_stats
            elif layer_name == 'conv2':
                stats_list = conv2_stats
            elif layer_name == 'conv3':  # 🔥 NEW!
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
    dead_conv3_count = sum(dead_conv3)  # 🔥 NEW!
    
    print(f"Conv1: {dead_conv1_count}/{len(conv1_stats)} dead channels ({dead_conv1_count/len(conv1_stats)*100:.1f}%)")
    print(f"Conv2: {dead_conv2_count}/{len(conv2_stats)} dead channels ({dead_conv2_count/len(conv2_stats)*100:.1f}%)")
    print(f"Conv3: {dead_conv3_count}/{len(conv3_stats)} dead channels ({dead_conv3_count/len(conv3_stats)*100:.1f}%)  [Strided Conv]")
    
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