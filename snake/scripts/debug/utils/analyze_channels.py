"""
🔬 CNN CHANNEL ACTIVATION ANALYSIS
Analizuje aktywację każdego kanału Conv1/Conv2 na różnych obserwacjach
✅ Zintegrowane z analyze_model.py
✅ Dostosowane do 2-layer CNN (Conv1 + Conv2)
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_conv_channels_detailed(model, env, output_dir, num_samples=100):
    """
    Analizuje aktywację kanałów Conv1, Conv2 na różnych obserwacjach
    ✅ UPDATED: 2-layer CNN with BatchNorm and GELU
    
    Args:
        model: RecurrentPPO model
        env: Snake environment
        output_dir: Directory to save results (logs/Analyze_model/02_cnn_layers/)
        num_samples: Number of samples to collect
    """
    print("\n" + "="*70)
    print("🔬 ANALYZING ALL CNN CHANNELS (Conv1/Conv2)")
    print("="*70 + "\n")
    
    device = next(model.policy.parameters()).device
    print(f"📍 Model device: {device}\n")
    
    features_extractor = model.policy.features_extractor
    
    # Detect number of channels for each layer
    num_ch1 = features_extractor.conv1.out_channels
    num_ch2 = features_extractor.conv2.out_channels
    
    print(f"📐 Architecture: Conv1({num_ch1}) → Conv2({num_ch2})\n")
    
    # Storage for channel statistics (per layer)
    layers_stats = {}
    for layer_name, num_channels in [('conv1', num_ch1), ('conv2', num_ch2)]:
        layers_stats[layer_name] = {
            'mean': np.zeros(num_channels),
            'max': np.zeros(num_channels),
            'std': np.zeros(num_channels),
            'active_ratio': np.zeros(num_channels),  # % pixels > 0.1
        }
    
    print(f"📊 Collecting {num_samples} samples...\n")
    
    for sample_idx in range(num_samples):
        obs, _ = env.reset()
        
        # Random steps to get varied observations
        for _ in range(np.random.randint(0, 20)):
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
                break
        
        # Prepare image tensor and move to correct device
        image = torch.FloatTensor(obs['image']).unsqueeze(0)
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        image = image.to(device)  # ✅ MOVE TO GPU!
        
        with torch.no_grad():
            # 🔥 2-LAYER CNN: 3×3 → 3×3 (stride 2)
            
            # Conv1 (3×3)
            x = features_extractor.conv1(image)
            x = features_extractor.bn1(x)
            conv1_out = F.gelu(x)
            conv1_out_cpu = conv1_out.cpu().numpy()[0]  # Shape: (num_ch1, H, W)
            
            # Conv2 (3×3, strided)
            x = features_extractor.conv2(conv1_out)
            x = features_extractor.bn2(x)
            x = features_extractor.dropout2(x)
            conv2_out = F.gelu(x)
            conv2_out_cpu = conv2_out.cpu().numpy()[0]  # Shape: (num_ch2, H/2, W/2)
            
            # Accumulate statistics for each layer
            for ch in range(num_ch1):
                ch_data = conv1_out_cpu[ch]
                layers_stats['conv1']['mean'][ch] += ch_data.mean()
                layers_stats['conv1']['max'][ch] += ch_data.max()
                layers_stats['conv1']['std'][ch] += ch_data.std()
                layers_stats['conv1']['active_ratio'][ch] += (ch_data > 0.1).mean()
            
            for ch in range(num_ch2):
                ch_data = conv2_out_cpu[ch]
                layers_stats['conv2']['mean'][ch] += ch_data.mean()
                layers_stats['conv2']['max'][ch] += ch_data.max()
                layers_stats['conv2']['std'][ch] += ch_data.std()
                layers_stats['conv2']['active_ratio'][ch] += (ch_data > 0.1).mean()
        
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{num_samples} samples...")
    
    # Average statistics
    for layer_name in layers_stats:
        for key in layers_stats[layer_name]:
            layers_stats[layer_name][key] /= num_samples
    
    # ==================== PRINT RESULTS ====================
    for layer_name in ['conv1', 'conv2']:
        stats = layers_stats[layer_name]
        num_channels = len(stats['mean'])
        
        print("\n" + "="*70)
        print(f"📊 {layer_name.upper()} CHANNEL STATISTICS (averaged over {num_samples} samples)")
        print("="*70)
        print(f"{'Ch':<4} {'Mean':<8} {'Max':<8} {'Std':<8} {'Active%':<10} {'Status'}")
        print("-"*70)
        
        dead_channels = []
        weak_channels = []
        
        for ch in range(num_channels):
            mean = stats['mean'][ch]
            max_val = stats['max'][ch]
            std = stats['std'][ch]
            active = stats['active_ratio'][ch] * 100
            
            # Classify channel health
            if active < 10:
                status = "💀 DEAD"
                dead_channels.append(ch)
            elif active < 30:
                status = "⚠️  WEAK"
                weak_channels.append(ch)
            else:
                status = "✅ OK"
            
            print(f"{ch:<4} {mean:<8.3f} {max_val:<8.3f} {std:<8.3f} {active:<10.1f} {status}")
        
        print("="*70)
        
        # Summary for this layer
        print(f"\n📋 {layer_name.upper()} SUMMARY:")
        print(f"  Dead channels (active < 10%):  {dead_channels if dead_channels else 'None ✅'}")
        print(f"  Weak channels (active < 30%):  {weak_channels if weak_channels else 'None ✅'}")
        print(f"  Healthy channels:              {num_channels - len(dead_channels) - len(weak_channels)}/{num_channels}")
        
        # Overall statistics
        avg_active = stats['active_ratio'].mean() * 100
        print(f"\n  Average active ratio:          {avg_active:.1f}%")
        print(f"  Most active channel:           Ch{stats['active_ratio'].argmax()} ({stats['active_ratio'].max()*100:.1f}%)")
        print(f"  Least active channel:          Ch{stats['active_ratio'].argmin()} ({stats['active_ratio'].min()*100:.1f}%)")
        
        # Detailed analysis of worst channel
        if dead_channels or weak_channels:
            worst_ch = stats['active_ratio'].argmin()
            print(f"\n🔍 WORST CHANNEL (Ch{worst_ch}):")
            print(f"  Mean:         {stats['mean'][worst_ch]:.4f}")
            print(f"  Max:          {stats['max'][worst_ch]:.4f}")
            print(f"  Std:          {stats['std'][worst_ch]:.4f}")
            print(f"  Active ratio: {stats['active_ratio'][worst_ch]*100:.2f}%")
            
            if stats['max'][worst_ch] < 0.5:
                print(f"  ❌ Channel NEVER activates strongly (max < 0.5)")
                print(f"  💡 Likely cause: Bad initialization or gradient vanishing")
    
    # ==================== VISUALIZATION ====================
    fig = plt.figure(figsize=(20, 8))
    
    for idx, layer_name in enumerate(['conv1', 'conv2']):
        stats = layers_stats[layer_name]
        num_channels = len(stats['mean'])
        
        # Determine dead/weak channels for coloring
        dead_channels = [ch for ch in range(num_channels) if stats['active_ratio'][ch] < 0.1]
        weak_channels = [ch for ch in range(num_channels) if 0.1 <= stats['active_ratio'][ch] < 0.3]
        
        colors = ['red' if ch in dead_channels else 'orange' if ch in weak_channels else 'green' 
                  for ch in range(num_channels)]
        
        # 4 subplots per layer
        base_pos = idx * 4
        
        # Plot 1: Mean activation
        ax1 = plt.subplot(2, 4, base_pos + 1)
        ax1.bar(range(num_channels), stats['mean'], color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Mean Activation')
        ax1.set_title(f'{layer_name.upper()} - Mean Activation')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Active ratio
        ax2 = plt.subplot(2, 4, base_pos + 2)
        ax2.bar(range(num_channels), stats['active_ratio'] * 100, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Dead (10%)')
        ax2.axhline(y=30, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Weak (30%)')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Active Pixels (%)')
        ax2.set_title(f'{layer_name.upper()} - Active Ratio')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Max activation
        ax3 = plt.subplot(2, 4, base_pos + 3)
        ax3.bar(range(num_channels), stats['max'], color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Max Activation')
        ax3.set_title(f'{layer_name.upper()} - Max Activation')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Std deviation
        ax4 = plt.subplot(2, 4, base_pos + 4)
        ax4.bar(range(num_channels), stats['std'], color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('Std Dev')
        ax4.set_title(f'{layer_name.upper()} - Variability')
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to output directory
    output_path = os.path.join(output_dir, 'all_conv_channels_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {output_path}")
    plt.close()
    
    # ==================== OVERALL SUMMARY ====================
    print("\n" + "="*70)
    print("🎯 OVERALL CNN HEALTH SUMMARY")
    print("="*70)
    
    for layer_name in ['conv1', 'conv2']:
        stats = layers_stats[layer_name]
        num_channels = len(stats['mean'])
        dead_count = sum(1 for ch in range(num_channels) if stats['active_ratio'][ch] < 0.1)
        weak_count = sum(1 for ch in range(num_channels) if 0.1 <= stats['active_ratio'][ch] < 0.3)
        healthy_count = num_channels - dead_count - weak_count
        
        health_pct = (healthy_count / num_channels) * 100
        
        if health_pct >= 80:
            status = "✅ EXCELLENT"
        elif health_pct >= 60:
            status = "🟢 GOOD"
        elif health_pct >= 40:
            status = "🟡 FAIR"
        else:
            status = "🔴 POOR"
        
        print(f"{layer_name.upper():<8} {healthy_count:>2}/{num_channels} healthy ({health_pct:>5.1f}%)  {status}")
    
    print("="*70 + "\n")
    
    return layers_stats
