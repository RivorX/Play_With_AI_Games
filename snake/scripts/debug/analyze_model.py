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

# Import z utils/
from utils.analyze_activations import (
    analyze_basic_states,
    analyze_bottlenecks,
    plot_activation_overview,
    analyze_channel_specialization
)
from utils.analyze_lstm import (
    analyze_lstm_memory,
    analyze_confusion_matrix,
    analyze_uncertainty
)
from utils.analyze_advanced import (
    analyze_temporal_patterns,
    analyze_critical_moments,
    analyze_feature_importance
)

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
conv_viz_dir = os.path.join(output_dir, 'conv_visualizations')
viewport_dir = os.path.join(output_dir, 'viewports')
action_probs_dir = os.path.join(output_dir, 'action_probs')
heatmap_dir = os.path.join(output_dir, 'attention_heatmaps')
lstm_dir = os.path.join(output_dir, 'lstm_analysis')
uncertainty_dir = os.path.join(output_dir, 'uncertainty_analysis')
confusion_dir = os.path.join(output_dir, 'confusion_matrix')
temporal_dir = os.path.join(output_dir, 'temporal_patterns')
critical_dir = os.path.join(output_dir, 'critical_moments')
feature_dir = os.path.join(output_dir, 'feature_importance')

for dir_path in [output_dir, conv_viz_dir, viewport_dir, action_probs_dir, 
                 heatmap_dir, lstm_dir, uncertainty_dir, confusion_dir,
                 temporal_dir, critical_dir, feature_dir]:
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
print("=== ROZPOCZĘCIE ROZSZERZONEJ ANALIZY ===")
print("="*80)

# ===================================================
# CZĘŚĆ 1: ANALIZA PODSTAWOWA
# ===================================================
print("\n[1/10] Analiza podstawowych stanów, aktywacji i attention...")
action_probs_list, detailed_activations, layer_gradients, attention_heatmaps = analyze_basic_states(
    model=model,
    env=env,
    output_dirs={
        'conv_viz': conv_viz_dir,
        'viewport': viewport_dir,
        'action_probs': action_probs_dir,
        'heatmap': heatmap_dir
    },
    action_names=action_names,
    config=config
)

# ===================================================
# CZĘŚĆ 2: ANALIZA BOTTLENECKÓW
# ===================================================
print("\n[2/10] Analiza bottlenecków...")
bottleneck_report = analyze_bottlenecks(
    layer_gradients=layer_gradients,
    action_names=action_names,
    output_dir=output_dir
)

# ===================================================
# CZĘŚĆ 3: PRZEGLĄD AKTYWACJI
# ===================================================
print("\n[3/10] Generowanie wykresów przeglądu aktywacji...")
plot_activation_overview(
    detailed_activations=detailed_activations,
    action_probs_list=action_probs_list,
    action_names=action_names,
    output_dirs={
        'main': output_dir,
        'action_probs': action_probs_dir
    }
)

# ===================================================
# CZĘŚĆ 4: ANALIZA LSTM MEMORY
# ===================================================
print("\n[4/10] Analiza LSTM memory...")
analyze_lstm_memory(
    model=model,
    env=env,
    output_dir=lstm_dir,
    action_names=action_names,
    config=config
)

# ===================================================
# CZĘŚĆ 5: CONFUSION MATRIX
# ===================================================
print("\n[5/10] Analiza Confusion Matrix...")
analyze_confusion_matrix(
    model=model,
    env=env,
    output_dir=confusion_dir,
    action_names=action_names,
    num_episodes=20
)

# ===================================================
# CZĘŚĆ 6: UNCERTAINTY ANALYSIS
# ===================================================
print("\n[6/10] Analiza Uncertainty...")
analyze_uncertainty(
    model=model,
    env=env,
    output_dir=uncertainty_dir,
    action_names=action_names,
    num_episodes=10
)

# ===================================================
# CZĘŚĆ 7: ANALIZA SPECJALIZACJI KANAŁÓW
# ===================================================
print("\n[7/10] Analiza specjalizacji kanałów CNN...")
analyze_channel_specialization(
    model=model,
    env=env,
    output_dir=conv_viz_dir,
    num_samples=50
)

# ===================================================
# CZĘŚĆ 8: TEMPORAL PATTERNS ANALYSIS
# ===================================================
print("\n[8/10] Analiza wzorców temporalnych (LSTM memory patterns)...")
analyze_temporal_patterns(
    model=model,
    env=env,
    output_dir=temporal_dir,
    action_names=action_names,
    num_episodes=20
)

# ===================================================
# CZĘŚĆ 9: CRITICAL MOMENTS ANALYSIS
# ===================================================
print("\n[9/10] Analiza krytycznych momentów (near-death, food acquisition)...")
analyze_critical_moments(
    model=model,
    env=env,
    output_dir=critical_dir,
    action_names=action_names,
    num_episodes=30
)

# ===================================================
# CZĘŚĆ 10: FEATURE IMPORTANCE ANALYSIS
# ===================================================
print("\n[10/10] Analiza ważności cech (ablation study)...")
analyze_feature_importance(
    model=model,
    env=env,
    output_dir=feature_dir,
    action_names=action_names,
    num_samples=100
)

env.close()

# ===================================================
# PODSUMOWANIE KOŃCOWE
# ===================================================
print("\n" + "="*80)
print("=== ANALIZA ZAKOŃCZONA ===")
print("="*80)
print(f"\n📂 Ważne pliki analizy:")
print(f"   {output_dir}/")
print(f"   ├── bottleneck_analysis.png                ⚠️ Analiza bottlenecków")
print(f"   ├── bottleneck_report.csv                  📊 Raport bottlenecków")
print(f"   ├── neuron_activations_overview.png        🧠 Przegląd aktywacji")
print(f"   ├── attention_heatmaps/                    🔥 Attention heatmaps")
print(f"   ├── lstm_analysis/                         🧠 Analiza LSTM memory")
print(f"   ├── confusion_matrix/                      📊 Confusion matrix")
print(f"   ├── uncertainty_analysis/                  🎲 Uncertainty metrics")
print(f"   ├── temporal_patterns/                     🕐 Wzorce temporalne")
print(f"   ├── critical_moments/                      ⚠️ Krytyczne momenty")
print(f"   └── feature_importance/                    🎯 Ważność cech")

print("\n" + "="*80)
print("=== KLUCZOWE WYNIKI ===")
print("="*80)

print("\n🔥 ATTENTION HEATMAPS:")
print("   - Pokazują które regiony viewport są najważniejsze dla decyzji")
print("   - Czerwone obszary = wysoka uwaga modelu")
print("   - Sprawdź czy model patrzy na jedzenie, ściany, czy własne ciało")

print("\n🧠 LSTM MEMORY ANALYSIS:")
print("   - lstm_memory_evolution.png: jak zmienia się pamięć w czasie")
print("   - lstm_neurons_heatmap.png: aktywacja wszystkich neuronów LSTM")
print("   - Sprawdź czy LSTM faktycznie wykorzystuje pamięć długoterminową")

print("\n🕐 TEMPORAL PATTERNS:")
print("   - temporal_ngrams.png: najczęstsze sekwencje akcji (bigrams/trigrams)")
print("   - temporal_forgetting_curve.png: jak szybko LSTM zapomina")
print("   - temporal_entropy_evolution.png: niepewność decyzji w czasie")

print("\n⚠️ CRITICAL MOMENTS:")
print("   - critical_near_death.png: zachowanie modelu przed kolizją")
print("   - critical_food_acquisition.png: efektywność zbierania jedzenia")
print("   - critical_tight_spaces.png: decyzje w ciasnych przestrzeniach")

print("\n🎯 FEATURE IMPORTANCE:")
print("   - feature_ablation_study.png: wpływ CNN vs scalars")
print("   - feature_gradient_importance.png: gradient-based importance")
print("   - feature_importance_results.csv: szczegółowe wyniki")

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