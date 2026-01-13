import os
import sys
import yaml
import torch
import numpy as np
import shutil
from pathlib import Path
from stable_baselines3 import PPO

# Dodaj scripts do path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import make_env
from cnn import CustomFeaturesExtractor

# Import modułów analizy
from utils.analyze_basic import analyze_basic_states, plot_activation_overview
from utils.analyze_gradients import analyze_bottlenecks, analyze_gradient_flow_detailed
from utils.analyze_cnn import analyze_cnn_layers
from utils.analyze_channels import analyze_conv_channels_detailed
from utils.analyze_performance import analyze_performance_metrics

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def load_model_interactive():
    """
    🎯 Interaktywny wybór źródła modelu dla analizy
    
    Returns:
        tuple: (model, source_name)
    """
    models_dir = os.path.join(base_dir, config['paths']['models_dir'])
    best_model_path = os.path.join(models_dir, 'best_model.zip')
    latest_model_path = os.path.join(models_dir, 'snake_ppo_model.zip')
    policy_path = os.path.join(base_dir, 'models', 'policy.pth')
    
    has_best = os.path.exists(best_model_path)
    has_latest = os.path.exists(latest_model_path)
    has_policy = os.path.exists(policy_path)
    
    print(f"\n{'='*70}")
    print(f"[MODEL SOURCE SELECTION FOR ANALYSIS]")
    print(f"{'='*70}")
    
    options = []
    
    if has_best:
        options.append(('1', 'best_model.zip', best_model_path))
        print(f"  [1] 🏆 best_model.zip (najlepszy model z treningu)")
    
    if has_latest and latest_model_path != best_model_path:
        options.append(('2', 'snake_ppo_model.zip', latest_model_path))
        print(f"  [2] 📦 snake_ppo_model.zip (ostatni checkpoint)")
    
    if has_policy:
        key = str(len(options) + 1)
        options.append((key, 'policy.pth', policy_path))
        print(f"  [{key}] 🎯 policy.pth (tylko wagi sieci)")
    
    print(f"{'='*70}")
    
    if not options:
        raise FileNotFoundError("Nie znaleziono żadnego modelu! Sprawdź folder models/")
    
    if len(options) == 1:
        choice = options[0][0]
        print(f"\n✅ Automatycznie wybrany: {options[0][1]}\n")
    else:
        while True:
            choice = input(f"\nWybierz źródło modelu [1-{len(options)}]: ").strip()
            if any(choice == opt[0] for opt in options):
                break
            print(f"❌ Nieprawidłowy wybór. Wybierz 1-{len(options)}.")
    
    selected = next(opt for opt in options if opt[0] == choice)
    source_name = selected[1]
    source_path = selected[2]
    
    print(f"\n🎬 Ładowanie: {source_name}...")
    
    # Załaduj model
    if source_name == 'policy.pth':
        # Stwórz env do sprawdzenia observation_space
        temp_env = make_env(render_mode=None, grid_size=8)()
        
        # Stwórz pusty model
        policy_kwargs = config['model']['policy_kwargs'].copy()
        policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor
        
        model = PPO(
            config['model']['policy'],
            temp_env,
            learning_rate=0.0001,
            n_steps=config['model']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=config['model']['device']
        )
        
        # Załaduj wagi
        state_dict = torch.load(source_path, map_location=config['model']['device'])
        model.policy.load_state_dict(state_dict)
        
        temp_env.close()
        print(f"✅ Załadowano policy.pth\n")
    else:
        model = PPO.load(source_path)
        print(f"✅ Załadowano {source_name}\n")
    
    return model, source_name


# Ścieżka do najlepszego modelu (dla backward compatibility)
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
    'performance': os.path.join(output_dir, '04_performance')
}

for dir_path in subdirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Załaduj model PPO
print("="*80)
print("🚀 MODEL ANALYSIS")
print("="*80)

# 🎯 INTERAKTYWNY WYBÓR MODELU
model, source_name = load_model_interactive()

print(f"\n📌 Analyzing model from: {source_name}")
print("\nŁadowanie modelu...")
policy = model.policy
features_extractor = policy.features_extractor

print(f"\n=== Informacje o modelu ===")
print(f"CNN channels: {config['model']['convlstm']['cnn_channels']}")
print(f"Bottleneck dims: {config['model']['convlstm'].get('cnn_bottleneck_dims', 'N/A')}")
print(f"CNN output dim: {config['model']['convlstm'].get('cnn_output_dim', 'N/A')}")
print(f"Scalar hidden dims: {config['model']['convlstm']['scalar_hidden_dims']}")
print(f"Features dim: {config['model']['policy_kwargs']['features_extractor_kwargs']['features_dim']}")
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
print("\n[1/6] 📊 Analiza podstawowa: aktywacje, viewport, attention...")
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
print("\n[2/6] 🔍 Analiza warstw CNN (channels, saturation, specialization)...")
analyze_cnn_layers(
    model=model,
    env=env,
    output_dir=subdirs['cnn'],
    num_samples=100
)

# ===================================================
# CZĘŚĆ 2.5: SZCZEGÓŁOWA ANALIZA KANAŁÓW CNN
# ===================================================
print("\n[2.5/6] 🔬 Szczegółowa analiza kanałów Conv1/Conv2...")
analyze_conv_channels_detailed(
    model=model,
    env=env,
    output_dir=subdirs['cnn'],
    num_samples=100
)

# ===================================================
# CZĘŚĆ 3: ANALIZA GRADIENTÓW (bottlenecks, gradient flow)
# ===================================================
print("\n[3/6] 🌊 Analiza przepływu gradientów...")
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
# CZĘŚĆ 4: ANALIZA WYDAJNOŚCI (critical moments, feature importance)
# ===================================================
print("\n[4/5] 🎯 Analiza wydajności i zachowań modelu...")
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
print("\n📂 Wyniki analizy zapisane w:")
print(f"   {output_dir}/")
print(f"   ├── 01_basic_analysis/         📊 Podstawowe aktywacje i viewport")
print(f"   ├── 02_cnn_layers/             🔍 Analiza warstw CNN + szczegółowa analiza kanałów")
print(f"   ├── 03_gradient_flow/          🌊 Przepływ gradientów")
print(f"   └── 04_performance/            🎯 Wydajność i zachowania")

print("\n" + "="*80)
print("=== KLUCZOWE WYNIKI ===")
print("="*80)

print(f"\n📌 Analyzed model: {source_name}")

print("\n📊 BASIC ANALYSIS:")
print("   - neuron_activations_overview.png: RMS aktywacji CNN vs Scalars")
print("   - attention_heatmaps/: gdzie model skupia uwagę")
print("   - viewports/: wizualizacja stanów gry")

print("\n🔍 CNN LAYERS:")
print("   - channel_specialization.png: aktywne vs martwe kanały")
print("   - activation_saturation.png: saturacja GELU")
print("   - conv_visualizations/: filtry CNN dla każdej warstwy")
print("   - all_conv_channels_analysis.png: szczegółowa analiza każdego kanału")

print("\n🌊 GRADIENT FLOW:")
print("   - bottleneck_analysis_split.png: bottlenecki per sekcja")
print("   - bottleneck_gradient_heatmap.png: flow przez warstwy")
print("   - gradient_flow_detailed.png: vanishing/explosion")

print("\n🎯 PERFORMANCE:")
print("   - critical_near_death.png: zachowanie przed kolizją")
print("   - critical_food_acquisition.png: efektywność zbierania")
print("   - feature_ablation_study.png: wpływ CNN vs scalars")

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