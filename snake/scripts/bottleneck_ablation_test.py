"""
Ablation study: Porównanie 3 architektur bottleneck
1. CURRENT: alpha-weighted (main + skip)
2. SKIP ONLY: alpha=0 (tylko skip path)
3. MAIN ONLY: alpha=1 (tylko main path)

Testuje na 100 epizodach i porównuje performance.
"""

import torch
import numpy as np
import yaml
import os
from sb3_contrib import RecurrentPPO
from model import make_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===========================
# LOAD CONFIG & MODEL
# ===========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_path = os.path.join(base_dir, config['paths']['model_path'])
print(f"📂 Ładowanie modelu z: {model_path}")
model = RecurrentPPO.load(model_path)

device = next(model.policy.features_extractor.parameters()).device
print(f"🖥️  Device: {device}")

# ===========================
# EVALUATION FUNCTION
# ===========================
def evaluate_with_alpha(model, alpha_value, n_episodes=50, grid_size=16):
    """
    Evaluates model with forced alpha value.
    
    Args:
        alpha_value: 0.0 (skip only), 1.0 (main only), or None (use learned)
    """
    extractor = model.policy.features_extractor
    original_alpha = extractor.alpha.clone()
    
    # Force alpha if specified
    if alpha_value is not None:
        # Convert to raw alpha (inverse sigmoid)
        if alpha_value <= 0.01:
            raw_alpha = -10.0  # sigmoid(-10) ≈ 0
        elif alpha_value >= 0.99:
            raw_alpha = 10.0   # sigmoid(10) ≈ 1
        else:
            raw_alpha = np.log(alpha_value / (1 - alpha_value))
        
        with torch.no_grad():
            extractor.alpha.fill_(raw_alpha)
    
    # Create eval environment
    env = DummyVecEnv([make_env(render_mode=None, grid_size=grid_size)])
    
    scores = []
    lengths = []
    rewards = []
    
    for _ in tqdm(range(n_episodes), desc=f"Alpha={alpha_value}"):
        obs = env.reset()
        lstm_states = None
        episode_rewards = []
        done = False
        
        while not done:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                deterministic=True
            )
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward[0])
            
            if done[0]:
                scores.append(info[0]['score'])
                lengths.append(info[0]['snake_length'])
                rewards.append(sum(episode_rewards))
                break
    
    env.close()
    
    # Restore original alpha
    with torch.no_grad():
        extractor.alpha.copy_(original_alpha)
    
    return {
        'scores': scores,
        'lengths': lengths,
        'rewards': rewards,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_length': np.mean(lengths),
        'mean_reward': np.mean(rewards)
    }

# ===========================
# RUN ABLATION STUDY
# ===========================
print(f"\n{'='*70}")
print(f"[ABLATION STUDY]")
print(f"{'='*70}")
print(f"Testing 3 configurations on {50} episodes each (grid_size=16)")
print(f"")

# Current learned alpha
extractor = model.policy.features_extractor
learned_alpha = torch.sigmoid(extractor.alpha).item()

results = {}

# 1. Current (learned alpha)
print(f"1️⃣  Testing CURRENT (alpha={learned_alpha:.3f})...")
results['current'] = evaluate_with_alpha(model, alpha_value=None, n_episodes=50)

# 2. Skip only (alpha=0)
print(f"\n2️⃣  Testing SKIP ONLY (alpha=0.0)...")
results['skip_only'] = evaluate_with_alpha(model, alpha_value=0.0, n_episodes=50)

# 3. Main only (alpha=1)
print(f"\n3️⃣  Testing MAIN ONLY (alpha=1.0)...")
results['main_only'] = evaluate_with_alpha(model, alpha_value=1.0, n_episodes=50)

# ===========================
# RESULTS
# ===========================
print(f"\n{'='*70}")
print(f"[RESULTS]")
print(f"{'='*70}")

configs = [
    ('CURRENT (learned)', 'current', learned_alpha),
    ('SKIP ONLY', 'skip_only', 0.0),
    ('MAIN ONLY', 'main_only', 1.0)
]

for name, key, alpha in configs:
    r = results[key]
    print(f"\n{name} (alpha={alpha:.3f}):")
    print(f"  Mean score:  {r['mean_score']:.2f} ± {r['std_score']:.2f}")
    print(f"  Mean length: {r['mean_length']:.2f}")
    print(f"  Mean reward: {r['mean_reward']:.2f}")
    print(f"  Max score:   {max(r['scores'])}")

# ===========================
# COMPARISON
# ===========================
print(f"\n{'='*70}")
print(f"[COMPARISON]")
print(f"{'='*70}")

current_score = results['current']['mean_score']
skip_score = results['skip_only']['mean_score']
main_score = results['main_only']['mean_score']

best_config = max(configs, key=lambda x: results[x[1]]['mean_score'])

print(f"\n🏆 Best configuration: {best_config[0]}")
print(f"   Score: {results[best_config[1]]['mean_score']:.2f}")

# Calculate improvements
skip_diff = ((current_score - skip_score) / skip_score * 100) if skip_score > 0 else 0
main_diff = ((current_score - main_score) / main_score * 100) if main_score > 0 else 0

print(f"\n📊 Current vs Skip only:  {skip_diff:+.1f}%")
print(f"📊 Current vs Main only:  {main_diff:+.1f}%")

# ===========================
# VERDICT
# ===========================
print(f"\n{'='*70}")
print(f"[VERDICT]")
print(f"{'='*70}")

if abs(skip_diff) < 5 and abs(main_diff) < 5:
    print("⚠️  Wszystkie konfiguracje mają podobne wyniki (<5% różnicy)")
    print("   → Bottleneck może być zbędny - rozważ prostszą architekturę")
    print("   → REKOMENDACJA: Usuń bottleneck, zostaw skip only (mniej parametrów)")
    
elif current_score > skip_score and current_score > main_score:
    improvement = max(skip_diff, main_diff)
    print(f"✅ Current (learned alpha) jest NAJLEPSZY (+{improvement:.1f}%)")
    print("   → Bottleneck MA SENS - obie ścieżki się uzupełniają")
    print("   → REKOMENDACJA: Zostaw obecną architekturę")
    
elif skip_score > current_score and skip_score > main_score:
    improvement = ((skip_score - current_score) / current_score * 100)
    print(f"🔥 SKIP ONLY jest LEPSZY (+{improvement:.1f}%)")
    print("   → Bottleneck jest ZBĘDNY i spowalnia model")
    print("   → REKOMENDACJA: Usuń bottleneck (cnn_compress), zostaw skip")
    print(f"   → Zaoszczędzisz ~7M parametrów!")
    
elif main_score > current_score and main_score > skip_score:
    improvement = ((main_score - current_score) / current_score * 100)
    print(f"⚡ MAIN ONLY jest LEPSZY (+{improvement:.1f}%)")
    print("   → Skip connection PRZESZKADZA")
    print("   → REKOMENDACJA: Usuń skip path, zostaw tylko bottleneck")
    
print(f"{'='*70}\n")

# ===========================
# VISUALIZATION
# ===========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, key, alpha) in enumerate(configs):
    r = results[key]
    
    # Score distribution
    axes[idx].hist(r['scores'], bins=20, alpha=0.7, edgecolor='black')
    axes[idx].axvline(r['mean_score'], color='red', linestyle='--', linewidth=2, label=f'Mean: {r["mean_score"]:.1f}')
    axes[idx].set_title(f'{name}\n(alpha={alpha:.2f})')
    axes[idx].set_xlabel('Score')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(base_dir, 'logs', 'bottleneck_ablation.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📊 Wykres zapisany: {plot_path}")

print("\n✅ Analiza zakończona!")
print("\n💡 FINAL RECOMMENDATION:")

if skip_score > current_score * 1.02:  # >2% lepszy
    print("   🔥 USUŃ bottleneck! Skip only jest lepszy.")
    print("   📝 W cnn.py zmień:")
    print("      self.cnn_compress = nn.Identity()  # Wyłącz bottleneck")
    print("      self.alpha = 0.0  # Force skip only")
elif current_score > max(skip_score, main_score) * 1.02:
    print("   ✅ ZOSTAW obecną architekturę - działa najlepiej!")
else:
    print("   🤔 Różnice są minimalne - wybierz prostszą architekturę (skip only)")
    print("      Zaoszczędzisz parametry bez utraty performance.")