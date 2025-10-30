import os
import sys
import yaml
import torch
import numpy as np
import shutil
from sb3_contrib import RecurrentPPO

# Dodaj scripts do path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import make_env

# Import modułów analizy
from utils.analyze_basic import analyze_basic_states, plot_activation_overview
from utils.analyze_gradients import analyze_bottlenecks, analyze_gradient_flow_detailed
from utils.analyze_cnn import analyze_cnn_layers
from utils.analyze_lstm import analyze_lstm_comprehensive
from utils.analyze_performance import analyze_performance_metrics

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Ścieżka do najlepszego modelu
model_path = os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip')

# Utwórz katalogi na wyniki
output_dir = os.path.join(base_dir, 'logs', 'Analyze_model')

# ⚠️ WAŻNE: Wyczyść poprzednie wyniki przed rozpoczęciem
if os.path.exists(output_dir):
    print(f"🗑️  Czyszczenie poprzednich wyników z: {output_dir}")
    shutil.rmtree(output_dir)
    print("   ✓ Katalog wyczyszczony")

# Utwórz strukturę katalogów
subdirs = {
    'main': os.path.join(output_dir, '01_basic_analysis'),
    'conv_viz': os.path.join(output_dir, '01_basic_analysis', 'conv_visualizations'),
    'viewport': os.path.join(output_dir, '01_basic_analysis', 'viewports'),
    'heatmap': os.path.join(output_dir, '01_basic_analysis', 'attention_heatmaps'),
    'cnn': os.path.join(output_dir, '02_cnn_layers'),
    'gradients': os.path.join(output_dir, '03_gradient_flow'),
    'lstm': os.path.join(output_dir, '04_lstm_memory'),
    'performance': os.path.join(output_dir, '05_performance')
}

for dir_path in subdirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Załaduj model RecurrentPPO
print("="*80)
print("🚀 MODEL ANALYSIS")
print("="*80)
print("\nŁadowanie modelu...")
model = RecurrentPPO.load(model_path)
policy = model.policy
features_extractor = policy.features_extractor

print(f"\n=== Informacje o modelu ===")
print(f"CNN channels: {config['model']['convlstm']['cnn_channels']}")
print(f"Bottleneck dims: {config['model']['convlstm'].get('cnn_bottleneck_dims', 'N/A')}")
print(f"CNN output dim: {config['model']['convlstm'].get('cnn_output_dim', 'N/A')}")
print(f"Scalar hidden dims: {config['model']['convlstm']['scalar_hidden_dims']}")
print(f"Features dim: {config['model']['policy_kwargs']['features_extractor_kwargs']['features_dim']}")
print(f"LSTM hidden size: {config['model']['policy_kwargs']['lstm_hidden_size']}")
print(f"LSTM layers: {config['model']['policy_kwargs']['n_lstm_layers']}")
print(f"Actor network: {config['model']['policy_kwargs']['net_arch']['pi']}")
print(f"Critic network: {config['model']['policy_kwargs']['net_arch']['vf']}")

# Przygotuj środowisko
env = make_env(render_mode=None, grid_size=16)()

# Nazwy akcji
action_names = ['lewo', 'prosto', 'prawo']

print("\n" + "="*80)
print("=== ROZPOCZĘCIE ANALIZY ===")
print("="*80)

# ===================================================
# CZĘŚĆ 1: ANALIZA PODSTAWOWA (viewport, activations, attention)
# ===================================================
print("\n[1/5] 📊 Analiza podstawowa: aktywacje, viewport, attention...")
action_probs_list, detailed_activations, layer_gradients, attention_heatmaps = analyze_basic_states(
    model=model,
    env=env,
    output_dirs=subdirs,  # ✅ POPRAWKA: output_dirs zamiast output_dir
    action_names=action_names,
    config=config
)

plot_activation_overview(
    detailed_activations=detailed_activations,
    action_probs_list=action_probs_list,
    action_names=action_names,
    output_dirs=subdirs  # ✅ POPRAWKA: output_dirs zamiast output_dir
)

# ===================================================
# CZĘŚĆ 2: ANALIZA CNN (channels, saturation, specialization)
# ===================================================
print("\n[2/5] 🔍 Analiza warstw CNN (channels, saturation, specialization)...")
analyze_cnn_layers(
    model=model,
    env=env,
    output_dir=subdirs['cnn'],
    num_samples=100
)

# ===================================================
# CZĘŚĆ 3: ANALIZA GRADIENTÓW (bottlenecks, gradient flow)
# ===================================================
print("\n[3/5] 🌊 Analiza przepływu gradientów...")
bottleneck_report = analyze_bottlenecks(
    layer_gradients=layer_gradients,
    action_names=action_names,
    output_dir=subdirs['gradients']
)

analyze_gradient_flow_detailed(
    model=model,
    env=env,
    output_dir=subdirs['gradients'],
    num_samples=50
)

# ===================================================
# CZĘŚĆ 4: ANALIZA LSTM (memory, temporal patterns, forgetting)
# ===================================================
print("\n[4/5] 🧠 Kompleksowa analiza LSTM...")
analyze_lstm_comprehensive(
    model=model,
    env=env,
    output_dir=subdirs['lstm'],
    action_names=action_names,
    config=config,
    num_episodes=20
)

# ===================================================
# CZĘŚĆ 5: ANALIZA WYDAJNOŚCI (critical moments, feature importance, uncertainty)
# ===================================================
print("\n[5/5] 🎯 Analiza wydajności i zachowań modelu...")
analyze_performance_metrics(
    model=model,
    env=env,
    output_dir=subdirs['performance'],
    action_names=action_names,
    num_episodes=30,
    num_samples=100
)

env.close()

# ===================================================
# PODSUMOWANIE KOŃCOWE
# ===================================================
print("\n" + "="*80)
print("=== ANALIZA ZAKOŃCZONA ===")
print("="*80)
print(f"\n📂 Wyniki analizy zapisane w:")
print(f"   {output_dir}/")
print(f"   ├── 01_basic_analysis/         📊 Podstawowe aktywacje i viewport")
print(f"   ├── 02_cnn_layers/             🔍 Analiza warstw CNN")
print(f"   ├── 03_gradient_flow/          🌊 Przepływ gradientów")
print(f"   ├── 04_lstm_memory/            🧠 Pamięć i wzorce temporalne")
print(f"   └── 05_performance/            🎯 Wydajność i zachowania")

print("\n" + "="*80)
print("=== KLUCZOWE WYNIKI ===")
print("="*80)

print("\n📊 BASIC ANALYSIS:")
print("   - neuron_activations_overview.png: RMS aktywacji CNN vs Scalars")
print("   - attention_heatmaps/: gdzie model skupia uwagę")
print("   - viewports/: wizualizacja stanów gry")

print("\n🔍 CNN LAYERS:")
print("   - channel_specialization.png: aktywne vs martwe kanały")
print("   - activation_saturation.png: saturacja GELU")
print("   - conv_visualizations/: filtry CNN dla każdej warstwy")

print("\n🌊 GRADIENT FLOW:")
print("   - bottleneck_analysis_split.png: bottlenecki per sekcja")
print("   - bottleneck_gradient_heatmap.png: flow przez warstwy")
print("   - gradient_flow_detailed.png: vanishing/explosion")

print("\n🧠 LSTM MEMORY:")
print("   - lstm_memory_evolution.png: ewolucja pamięci")
print("   - lstm_neurons_heatmap.png: aktywacja neuronów")
print("   - temporal_forgetting_curve.png: jak szybko zapomina")
print("   - temporal_ngrams.png: sekwencje akcji")

print("\n🎯 PERFORMANCE:")
print("   - critical_near_death.png: zachowanie przed kolizją")
print("   - critical_food_acquisition.png: efektywność zbierania")
print("   - feature_ablation_study.png: wpływ CNN vs scalars")
print("   - confusion_matrix.png: porównanie z heurystyką")
print("   - uncertainty_analysis.png: pewność decyzji")

print("\n⚠️ BOTTLENECKS:")
if bottleneck_report:
    high_severity = [b for b in bottleneck_report if b['severity'] == 'HIGH']
    medium_severity = [b for b in bottleneck_report if b['severity'] == 'MEDIUM']
    if high_severity:
        print(f"   - 🔴 WYSOKIE RYZYKO: {len(high_severity)} przypadków")
    if medium_severity:
        print(f"   - 🟡 ŚREDNIE RYZYKO: {len(medium_severity)} przypadków")
    if not high_severity and not medium_severity:
        print("   - ✅ Brak krytycznych bottlenecków")

print("\n" + "="*80)
print("✅ ANALIZA ZAKOŃCZONA!")
print("="*80)