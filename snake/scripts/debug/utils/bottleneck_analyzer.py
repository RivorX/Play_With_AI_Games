"""
Zaawansowana analiza architektury bottleneck w modelu Snake RL.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def run_bottleneck_analysis(model, env, output_dir, config):
    """
    Główna funkcja analizy bottleneck:
    1. EXTRACT ALPHA VALUE
    2. GRADIENT FLOW ANALYSIS
    3. OUTPUT CORRELATION
    4. INFORMATION LOSS
    5. RECOMMENDATIONS
    """
    print("\n" + "="*80)
    print("🔬 ZAAWANSOWANA ANALIZA BOTTLENECK")
    print("="*80)
    
    extractor = model.policy.features_extractor
    device = next(extractor.parameters()).device
    
    # ===========================
    # 1. EXTRACT ALPHA VALUE
    # ===========================
    print("\n[1/5] Ekstrakcja wartości alpha...")
    
    if hasattr(extractor, 'alpha'):
        alpha_raw = extractor.alpha.item()
        alpha_sigmoid = torch.sigmoid(extractor.alpha).item()
        
        print(f"  Raw alpha:        {alpha_raw:.4f}")
        print(f"  Sigmoid(alpha):   {alpha_sigmoid:.4f}")
        print(f"  Main path weight: {alpha_sigmoid:.1%}")
        print(f"  Skip path weight: {1-alpha_sigmoid:.1%}")
        
        # Interpretacja
        if alpha_sigmoid < 0.3:
            print("  ⚠️  Model IGNORUJE bottleneck (preferuje skip path)")
            alpha_interpretation = "skip_dominant"
        elif alpha_sigmoid > 0.7:
            print("  ✅ Model używa głównie bottleneck (main path)")
            alpha_interpretation = "main_dominant"
        else:
            print("  🤝 Model balansuje obie ścieżki")
            alpha_interpretation = "balanced"
    else:
        print("  ⚠️ Brak learnable alpha w modelu!")
        alpha_sigmoid = 0.5
        alpha_interpretation = "no_alpha"
    
    # ===========================
    # 2. GRADIENT FLOW ANALYSIS
    # ===========================
    print("\n[2/5] Analiza przepływu gradientów...")
    
    obs, _ = env.reset()
    
    # Przygotuj batch (1 sample) - PRZENIEŚ NA DEVICE!
    obs_tensor = {
        'image': torch.from_numpy(obs['image']).unsqueeze(0).to(device),
        'direction': torch.from_numpy(obs['direction']).unsqueeze(0).to(device),
        'dx_head': torch.from_numpy(obs['dx_head']).unsqueeze(0).to(device),
        'dy_head': torch.from_numpy(obs['dy_head']).unsqueeze(0).to(device),
        'front_coll': torch.from_numpy(obs['front_coll']).unsqueeze(0).to(device),
        'left_coll': torch.from_numpy(obs['left_coll']).unsqueeze(0).to(device),
        'right_coll': torch.from_numpy(obs['right_coll']).unsqueeze(0).to(device)
    }
    
    # Forward pass z gradient tracking
    extractor.eval()
    with torch.enable_grad():
        # CNN forward
        image = obs_tensor['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        x = image
        x.requires_grad_(True)
        
        # Conv layers
        x = extractor.conv1(x)
        x = extractor.bn1(x)
        x = torch.nn.functional.gelu(x)
        
        x = extractor.conv2(x)
        x = extractor.bn2(x)
        x = torch.nn.functional.gelu(x)
        x = extractor.dropout2(x)
        
        cnn_raw = extractor.flatten(x)
        cnn_raw = cnn_raw.float()
        cnn_raw.requires_grad_(True)
        
        # Main path (bottleneck)
        main_path = extractor.cnn_compress(cnn_raw)
        main_path.retain_grad()
        
        # Skip path
        skip_path = extractor.cnn_residual(cnn_raw)
        skip_path.retain_grad()
        
        # Fusion
        alpha = torch.sigmoid(extractor.alpha)
        cnn_features = alpha * main_path + (1 - alpha) * skip_path
        
        # Dummy loss
        loss = cnn_features.sum()
        loss.backward()
        
        # Gradient magnitudes
        main_grad = main_path.grad.abs().mean().item() if main_path.grad is not None else 0.0
        skip_grad = skip_path.grad.abs().mean().item() if skip_path.grad is not None else 0.0
        
        print(f"  Main path gradient: {main_grad:.6f}")
        print(f"  Skip path gradient: {skip_grad:.6f}")
        print(f"  Gradient ratio:     {main_grad / (skip_grad + 1e-8):.2f}x")
        
        gradient_interpretation = "balanced"
        if main_grad < skip_grad * 0.5:
            print("  ⚠️  Main path ma SŁABSZE gradienty (może być ignorowany)")
            gradient_interpretation = "skip_dominant"
        else:
            print("  ✅ Gradienty wydają się zbalansowane")
            gradient_interpretation = "balanced"
    
    # ===========================
    # 3. OUTPUT CORRELATION
    # ===========================
    print("\n[3/5] Analiza korelacji między ścieżkami...")
    
    n_samples = 100
    main_outputs = []
    skip_outputs = []
    
    for _ in range(n_samples):
        obs, _ = env.reset()
        obs_tensor = {
            'image': torch.from_numpy(obs['image']).unsqueeze(0).to(device),
            'direction': torch.from_numpy(obs['direction']).unsqueeze(0).to(device),
            'dx_head': torch.from_numpy(obs['dx_head']).unsqueeze(0).to(device),
            'dy_head': torch.from_numpy(obs['dy_head']).unsqueeze(0).to(device),
            'front_coll': torch.from_numpy(obs['front_coll']).unsqueeze(0).to(device),
            'left_coll': torch.from_numpy(obs['left_coll']).unsqueeze(0).to(device),
            'right_coll': torch.from_numpy(obs['right_coll']).unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            x = image
            x = extractor.conv1(x)
            x = extractor.bn1(x)
            x = torch.nn.functional.gelu(x)
            x = extractor.conv2(x)
            x = extractor.bn2(x)
            x = torch.nn.functional.gelu(x)
            x = extractor.dropout2(x)
            
            cnn_raw = extractor.flatten(x).float()
            main_path = extractor.cnn_compress(cnn_raw)
            skip_path = extractor.cnn_residual(cnn_raw)
            
            main_outputs.append(main_path.cpu().numpy())
            skip_outputs.append(skip_path.cpu().numpy())
    
    main_outputs = np.concatenate(main_outputs, axis=0)  # (n_samples, 640)
    skip_outputs = np.concatenate(skip_outputs, axis=0)
    
    # Korelacja (per feature)
    correlations = []
    for i in range(main_outputs.shape[1]):
        corr = np.corrcoef(main_outputs[:, i], skip_outputs[:, i])[0, 1]
        correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    print(f"  Mean correlation:   {mean_corr:.3f}")
    print(f"  Std correlation:    {np.std(correlations):.3f}")
    
    correlation_interpretation = "moderate"
    if mean_corr > 0.9:
        print("  ⚠️  BARDZO WYSOKA korelacja - obie ścieżki robią to samo!")
        print("     → Bottleneck może być zbędny")
        correlation_interpretation = "redundant"
    elif mean_corr > 0.7:
        print("  ⚠️  Wysoka korelacja - ścieżki są podobne")
        correlation_interpretation = "high"
    elif mean_corr < 0.3:
        print("  ✅ Niska korelacja - ścieżki uczą się różnych reprezentacji")
        correlation_interpretation = "diverse"
    else:
        print("  🤝 Umiarkowana korelacja - ścieżki się uzupełniają")
        correlation_interpretation = "moderate"
    
    # ===========================
    # 4. INFORMATION LOSS
    # ===========================
    print("\n[4/5] Analiza utraty informacji...")
    
    # Variance explained
    cnn_raw_var = np.var(cnn_raw.detach().cpu().numpy())
    main_var = np.var(main_outputs)
    skip_var = np.var(skip_outputs)
    
    # Per-neuron variance (znormalizowana przez liczbę cech)
    cnn_raw_dim = cnn_raw.shape[1]  # 4096
    main_dim = main_outputs.shape[1]  # 640
    skip_dim = skip_outputs.shape[1]  # 640
    
    cnn_per_neuron = cnn_raw_var / cnn_raw_dim
    main_per_neuron = main_var / main_dim
    skip_per_neuron = skip_var / skip_dim
    
    print(f"  CNN raw variance:       {cnn_raw_var:.4f} (per-neuron: {cnn_per_neuron:.6f})")
    print(f"  Main path variance:     {main_var:.4f} (per-neuron: {main_per_neuron:.6f})")
    print(f"  Skip path variance:     {skip_var:.4f} (per-neuron: {skip_per_neuron:.6f})")
    print(f"")
    print(f"  📊 Variance ratio:")
    print(f"     Main/CNN: {main_var/cnn_raw_var:.2f}x (raw) | {main_per_neuron/cnn_per_neuron:.2f}x (per-neuron)")
    print(f"     Skip/CNN: {skip_var/cnn_raw_var:.2f}x (raw) | {skip_per_neuron/cnn_per_neuron:.2f}x (per-neuron)")
    
    information_interpretation = "neutral"
    if main_per_neuron > cnn_per_neuron * 2:
        print(f"")
        print("  ✅ Bottleneck AMPLIFIKUJE cechy (zwiększa separację)")
        print("     → To jest DOBRE! Model uczy się użytecznych transformacji")
        information_interpretation = "amplifying"
    elif main_per_neuron < cnn_per_neuron * 0.5:
        print(f"")
        print("  ⚠️  Bottleneck TRACI informację (compresses too much)")
        information_interpretation = "lossy"
    else:
        print(f"")
        print("  🤝 Bottleneck zachowuje podobny poziom informacji")
        information_interpretation = "neutral"
    
    # ===========================
    # 5. RECOMMENDATIONS
    # ===========================
    print("\n[5/5] Generowanie rekomendacji...")
    
    recommendations = []
    
    if alpha_interpretation == "skip_dominant" and correlation_interpretation == "redundant":
        recommendations.append("❌ Bottleneck NIE DZIAŁA - usuń go!")
        recommendations.append("   - Model preferuje skip path")
        recommendations.append("   - Obie ścieżki są zbyt podobne")
        recommendations.append("   💡 SUGESTIE:")
        recommendations.append("      1. USUŃ bottleneck (zostaw tylko skip)")
        recommendations.append("      2. LUB zwiększ cnn_bottleneck_dims (np. [2048, 1024])")
        verdict = "remove_bottleneck"
    
    elif alpha_interpretation == "main_dominant" and correlation_interpretation == "diverse":
        recommendations.append("✅ Bottleneck DZIAŁA DOBRZE!")
        recommendations.append("   - Model preferuje main path")
        recommendations.append("   - Ścieżki uczą się różnych rzeczy")
        recommendations.append("   💡 SUGESTIE:")
        recommendations.append("      - Architektura wygląda OK, kontynuuj trening")
        verdict = "keep_current"
    
    else:
        recommendations.append("🤔 Bottleneck jest AMBIWALENTNY")
        recommendations.append("   💡 SUGESTIE:")
        recommendations.append("      1. Zaloguj alpha przez czas (czy się zmienia?)")
        recommendations.append("      2. Spróbuj fixed alpha=0.0 (tylko skip) vs alpha=1.0 (tylko main)")
        recommendations.append("      3. Porównaj performance na walidacji")
        verdict = "unclear"
    
    # Wyświetl rekomendacje
    print("\n" + "="*80)
    print("[REKOMENDACJE]")
    print("="*80)
    for rec in recommendations:
        print(rec)
    print("="*80)
    
    # ===========================
    # WIZUALIZACJA
    # ===========================
    print("\n[6/6] Generowanie wizualizacji...")
    
    # Plot 1: Alpha value & gradient flow
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Alpha visualization
    ax = axes[0, 0]
    alpha_data = [alpha_sigmoid, 1 - alpha_sigmoid]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(['Main Path', 'Skip Path'], alpha_data, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Weight (alpha)')
    ax.set_title(f'Alpha Distribution (learned = {alpha_sigmoid:.3f})')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, alpha_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Gradient flow
    ax = axes[0, 1]
    grad_data = [main_grad, skip_grad]
    bars = ax.bar(['Main Path', 'Skip Path'], grad_data, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Through Paths')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, grad_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10)
    
    # Correlation histogram
    ax = axes[1, 0]
    ax.hist(correlations, bins=30, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.axvline(mean_corr, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_corr:.3f}')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Path Output Correlation Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Variance comparison
    ax = axes[1, 1]
    variance_data = [cnn_per_neuron, main_per_neuron, skip_per_neuron]
    variance_labels = ['CNN Raw', 'Main Path', 'Skip Path']
    colors_var = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(variance_labels, variance_data, color=colors_var, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Variance per Neuron')
    ax.set_title('Information Density (per-neuron variance)')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, variance_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'bottleneck_advanced_analysis.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✅ Wykres zapisany: {plot_path}")
    
    # Zapisz wyniki do pliku tekstowego
    report_path = os.path.join(output_dir, 'bottleneck_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZAAWANSOWANA ANALIZA BOTTLENECK - RAPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("[1] ALPHA VALUE\n")
        f.write(f"  Alpha (sigmoid): {alpha_sigmoid:.4f}\n")
        f.write(f"  Interpretation: {alpha_interpretation}\n\n")
        
        f.write("[2] GRADIENT FLOW\n")
        f.write(f"  Main path gradient: {main_grad:.6f}\n")
        f.write(f"  Skip path gradient: {skip_grad:.6f}\n")
        f.write(f"  Interpretation: {gradient_interpretation}\n\n")
        
        f.write("[3] PATH CORRELATION\n")
        f.write(f"  Mean correlation: {mean_corr:.3f}\n")
        f.write(f"  Interpretation: {correlation_interpretation}\n\n")
        
        f.write("[4] INFORMATION LOSS\n")
        f.write(f"  CNN per-neuron variance: {cnn_per_neuron:.6f}\n")
        f.write(f"  Main per-neuron variance: {main_per_neuron:.6f}\n")
        f.write(f"  Skip per-neuron variance: {skip_per_neuron:.6f}\n")
        f.write(f"  Interpretation: {information_interpretation}\n\n")
        
        f.write("[5] VERDICT\n")
        f.write(f"  {verdict}\n\n")
        
        f.write("[6] RECOMMENDATIONS\n")
        for rec in recommendations:
            f.write(f"  {rec}\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Raport zapisany: {report_path}")
    
    print("\n✅ Zaawansowana analiza bottleneck zakończona!")
    
    return {
        'alpha_sigmoid': alpha_sigmoid,
        'alpha_interpretation': alpha_interpretation,
        'main_grad': main_grad,
        'skip_grad': skip_grad,
        'gradient_interpretation': gradient_interpretation,
        'mean_correlation': mean_corr,
        'correlation_interpretation': correlation_interpretation,
        'information_interpretation': information_interpretation,
        'verdict': verdict,
        'recommendations': recommendations
    }