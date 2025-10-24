"""
Skrypt do analizy efektywności architektury bottleneck w modelu Snake RL.
Sprawdza:
1. Wartość learnable alpha (czy model preferuje main czy skip path)
2. Gradient flow przez obie ścieżki
3. Korelację między main_path i skip_path
4. Information loss w bottleneck
"""

import torch
import numpy as np
import yaml
import os
from sb3_contrib import RecurrentPPO
from model import make_env
import matplotlib.pyplot as plt

# ===========================
# 1. LOAD MODEL & CONFIG
# ===========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_path = os.path.join(base_dir, config['paths']['model_path'])

if not os.path.exists(model_path):
    print(f"❌ Model nie znaleziony: {model_path}")
    exit(1)

print(f"📂 Ładowanie modelu z: {model_path}")
model = RecurrentPPO.load(model_path)

# ===========================
# 2. EXTRACT ALPHA VALUE
# ===========================
extractor = model.policy.features_extractor

if hasattr(extractor, 'alpha'):
    alpha_raw = extractor.alpha.item()
    alpha_sigmoid = torch.sigmoid(extractor.alpha).item()
    
    print(f"\n{'='*70}")
    print(f"[ALPHA VALUE]")
    print(f"{'='*70}")
    print(f"  Raw alpha:        {alpha_raw:.4f}")
    print(f"  Sigmoid(alpha):   {alpha_sigmoid:.4f}")
    print(f"  Main path weight: {alpha_sigmoid:.1%}")
    print(f"  Skip path weight: {1-alpha_sigmoid:.1%}")
    print(f"{'='*70}\n")
    
    # Interpretacja
    if alpha_sigmoid < 0.3:
        print("⚠️  UWAGA: Model IGNORUJE bottleneck (preferuje skip path)")
    elif alpha_sigmoid > 0.7:
        print("✅ Model używa głównie bottleneck (main path)")
    else:
        print("🤝 Model balansuje obie ścieżki")
else:
    print("❌ Brak learnable alpha w modelu!")
    alpha_sigmoid = 0.5

# ===========================
# 3. GRADIENT FLOW ANALYSIS
# ===========================
print(f"\n{'='*70}")
print(f"[GRADIENT FLOW TEST]")
print(f"{'='*70}")

# Detect device
device = next(extractor.parameters()).device
print(f"  Model device: {device}")

# Stwórz dummy environment
env = make_env(render_mode=None, grid_size=16)()
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
    main_path.retain_grad()  # ✅ ZACHOWAJ GRADIENT!
    
    # Skip path
    skip_path = extractor.cnn_residual(cnn_raw)
    skip_path.retain_grad()  # ✅ ZACHOWAJ GRADIENT!
    
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
    
    if main_grad < skip_grad * 0.5:
        print("  ⚠️  Main path ma SŁABSZE gradienty (może być ignorowany)")
    else:
        print("  ✅ Gradienty wydają się zbalansowane")

print(f"{'='*70}\n")

# ===========================
# 4. OUTPUT CORRELATION
# ===========================
print(f"\n{'='*70}")
print(f"[PATH CORRELATION]")
print(f"{'='*70}")

# Zbierz wiele sampli
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

if mean_corr > 0.9:
    print("  ⚠️  BARDZO WYSOKA korelacja - obie ścieżki robią to samo!")
    print("     → Bottleneck może być zbędny")
elif mean_corr > 0.7:
    print("  ⚠️  Wysoka korelacja - ścieżki są podobne")
elif mean_corr < 0.3:
    print("  ✅ Niska korelacja - ścieżki uczą się różnych reprezentacji")
else:
    print("  🤝 Umiarkowana korelacja - ścieżki się uzupełniają")

print(f"{'='*70}\n")

# ===========================
# 5. INFORMATION LOSS
# ===========================
print(f"\n{'='*70}")
print(f"[INFORMATION LOSS]")
print(f"{'='*70}")

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

if main_per_neuron > cnn_per_neuron * 2:
    print(f"")
    print("  ✅ Bottleneck AMPLIFIKUJE cechy (zwiększa separację)")
    print("     → To jest DOBRE! Model uczy się użytecznych transformacji")
elif main_per_neuron < cnn_per_neuron * 0.5:
    print(f"")
    print("  ⚠️  Bottleneck TRACI informację (compresses too much)")
else:
    print(f"")
    print("  🤝 Bottleneck zachowuje podobny poziom informacji")

print(f"{'='*70}\n")

# ===========================
# 6. RECOMMENDATIONS
# ===========================
print(f"\n{'='*70}")
print(f"[REKOMENDACJE]")
print(f"{'='*70}")

if alpha_sigmoid < 0.3 and mean_corr > 0.8:
    print("❌ Bottleneck NIE DZIAŁA:")
    print("   - Model preferuje skip path")
    print("   - Obie ścieżki są zbyt podobne")
    print("\n💡 SUGESTIE:")
    print("   1. USUŃ bottleneck (zostaw tylko skip)")
    print("   2. LUB zwiększ cnn_bottleneck_dims (np. [2048, 1024])")
    print("   3. LUB dodaj więcej regularizacji do skip path")
    
elif alpha_sigmoid > 0.7 and mean_corr < 0.5:
    print("✅ Bottleneck DZIAŁA DOBRZE:")
    print("   - Model preferuje main path")
    print("   - Ścieżki uczą się różnych rzeczy")
    print("\n💡 SUGESTIE:")
    print("   - Architektura wygląda OK, kontynuuj trening")
    
else:
    print("🤔 Bottleneck jest AMBIWALENTNY:")
    print("\n💡 SUGESTIE:")
    print("   1. Zaloguj alpha przez czas (czy się zmienia?)")
    print("   2. Spróbuj fixed alpha=0.0 (tylko skip) vs alpha=1.0 (tylko main)")
    print("   3. Porównaj performance na walidacji")

print(f"{'='*70}\n")

env.close()
print("✅ Analiza zakończona!")