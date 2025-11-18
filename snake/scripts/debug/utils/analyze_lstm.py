"""
🧠 LSTM COMPREHENSIVE ANALYSIS MODULE

Scalona analiza LSTM:
- Memory evolution (hidden/cell states)
- Temporal patterns (n-grams, forgetting curve)
- Confusion matrix
- Uncertainty analysis

Wszystkie wyniki w jednym katalogu: 04_lstm_memory/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import Counter
from scipy.spatial.distance import cosine


def analyze_lstm_comprehensive(model, env, output_dir, action_names, config, num_episodes=20):
    """
    🧠 KOMPLEKSOWA ANALIZA LSTM
    Łączy: memory evolution + temporal patterns + confusion matrix + uncertainty
    """
    print("\n" + "="*80)
    print("🧠 LSTM COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    policy = model.policy
    
    # ==================== ZBIERANIE DANYCH ====================
    all_episodes_data = []
    confusion_matrix = np.zeros((3, 3))
    uncertainty_data = []
    
    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        step = 0
        
        episode_data = {
            'actions': [],
            'rewards': [],
            'hidden_states': [],
            'cell_states': [],
            'action_probs': [],
            'food_distances': []
        }
        
        while not done and step < 512 * 2:
            with torch.no_grad():
                # Predict
                action, new_lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=False
                )
                
                if torch.is_tensor(action):
                    action_idx = int(action.item())
                elif isinstance(action, np.ndarray):
                    action_idx = int(action.item())
                else:
                    action_idx = int(action)
                action_idx = int(np.clip(action_idx, 0, len(action_names) - 1))
                
                # Get action probabilities
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
                
                features = policy.features_extractor(obs_tensor)
                
                if new_lstm_states is not None:
                    lstm_states_tensor = (
                        torch.tensor(new_lstm_states[0], dtype=torch.float32, device=policy.device),
                        torch.tensor(new_lstm_states[1], dtype=torch.float32, device=policy.device)
                    )
                else:
                    batch_size = 1
                    n_layers = policy.lstm_actor.num_layers
                    hidden_size = policy.lstm_actor.hidden_size
                    lstm_states_tensor = (
                        torch.zeros(n_layers, batch_size, hidden_size, device=policy.device),
                        torch.zeros(n_layers, batch_size, hidden_size, device=policy.device)
                    )
                
                features_seq = features.unsqueeze(1)
                lstm_output, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
                latent_pi = lstm_output.squeeze(1)
                latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
                logits = policy.action_net(latent_pi_mlp)
                action_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Store data
                episode_data['actions'].append(action_idx)
                episode_data['action_probs'].append(action_probs.copy())
                
                if new_lstm_states is not None:
                    episode_data['hidden_states'].append(new_lstm_states[0].copy())
                    episode_data['cell_states'].append(new_lstm_states[1].copy())
                
                food_dist = np.sqrt(obs['dx_head'][0]**2 + obs['dy_head'][0]**2)
                episode_data['food_distances'].append(food_dist)
                
                # Uncertainty metrics
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
                max_prob = np.max(action_probs)
                
                uncertainty_data.append({
                    'episode': episode_idx,
                    'step': step,
                    'entropy': entropy,
                    'max_prob': max_prob,
                    'action': action_names[action_idx]
                })
                
                # Confusion matrix (heurystyka)
                dx = obs['dx_head'][0]
                dy = obs['dy_head'][0]
                front_coll = obs['front_coll'][0]
                left_coll = obs['left_coll'][0]
                right_coll = obs['right_coll'][0]
                
                expected_action = 1  # default: prosto
                if front_coll > 0.5:
                    if left_coll < 0.5:
                        expected_action = 0
                    elif right_coll < 0.5:
                        expected_action = 2
                else:
                    food_angle = np.arctan2(dy, dx) * 180 / np.pi
                    if abs(food_angle) < 30:
                        expected_action = 1
                    elif food_angle < -30:
                        expected_action = 0
                    elif food_angle > 30:
                        expected_action = 2
                
                confusion_matrix[expected_action, action_idx] += 1
                
                lstm_states = new_lstm_states
            
            # Step
            obs, reward, done, truncated, info = env.step(action_idx)
            episode_data['rewards'].append(reward)
            episode_starts = np.array([done or truncated], dtype=bool)
            done = done or truncated
            step += 1
        
        all_episodes_data.append(episode_data)
        
        if (episode_idx + 1) % 5 == 0:
            print(f"  Processed {episode_idx + 1}/{num_episodes} episodes")
    
    # ==================== [1/4] MEMORY EVOLUTION ====================
    print("\n📈 Generating memory evolution plots...")
    
    longest_episode = max(all_episodes_data, key=lambda x: len(x['hidden_states']))
    hidden_states = longest_episode['hidden_states']
    cell_states = longest_episode['cell_states']
    
    if len(hidden_states) > 0:
        steps = list(range(len(hidden_states)))
        hidden_means = [np.abs(h[0, 0, :]).mean() for h in hidden_states]
        cell_means = [np.abs(c[0, 0, :]).mean() for c in cell_states]
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Hidden state
        axes[0].plot(steps, hidden_means, color='#9b59b6', linewidth=2)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Mean |Activation|')
        axes[0].set_title('LSTM Hidden State Evolution')
        axes[0].grid(alpha=0.3)
        
        # Cell state
        axes[1].plot(steps, cell_means, color='#8e44ad', linewidth=2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Mean |Activation|')
        axes[1].set_title('LSTM Cell State Evolution')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lstm_memory_evolution.png'), dpi=150)
        plt.close()
        
        # Heatmap
        hidden_states_matrix = np.array([h[0, 0, :] for h in hidden_states])
        
        plt.figure(figsize=(16, 8))
        plt.imshow(hidden_states_matrix.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Activation Value')
        plt.xlabel('Time Step')
        plt.ylabel('LSTM Neuron')
        plt.title('LSTM Hidden State Heatmap Over Time')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lstm_neurons_heatmap.png'), dpi=150)
        plt.close()
        
        print(f"✅ Memory evolution saved")
    
    # ==================== [2/4] TEMPORAL PATTERNS ====================
    print("\n🕐 Analyzing temporal patterns...")
    
    # N-grams
    bigrams = Counter()
    trigrams = Counter()
    
    for ep_data in all_episodes_data:
        actions = ep_data['actions']
        
        for i in range(len(actions) - 1):
            bigram = (action_names[actions[i]], action_names[actions[i+1]])
            bigrams[bigram] += 1
        
        for i in range(len(actions) - 2):
            trigram = (action_names[actions[i]], action_names[actions[i+1]], action_names[actions[i+2]])
            trigrams[trigram] += 1
    
    # Plot n-grams
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    top_bigrams = bigrams.most_common(10)
    bigram_labels = [f"{b[0]}→{b[1]}" for b, _ in top_bigrams]
    bigram_counts = [c for _, c in top_bigrams]
    
    axes[0].barh(bigram_labels, bigram_counts, color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Frequency')
    axes[0].set_title('Top 10 Action Bigrams')
    axes[0].grid(axis='x', alpha=0.3)
    
    top_trigrams = trigrams.most_common(10)
    trigram_labels = [f"{t[0]}→{t[1]}→{t[2]}" for t, _ in top_trigrams]
    trigram_counts = [c for _, c in top_trigrams]
    
    axes[1].barh(trigram_labels, trigram_counts, color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Frequency')
    axes[1].set_title('Top 10 Action Trigrams')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_ngrams.png'), dpi=150)
    plt.close()
    print(f"✅ N-grams saved")
    
    # Forgetting curve
    if len(hidden_states) > 10:
        max_lag = min(50, len(hidden_states) - 1)
        similarities = []
        
        for lag in range(1, max_lag + 1):
            sim_values = []
            for i in range(len(hidden_states) - lag):
                h1 = hidden_states[i][0, 0, :].flatten()
                h2 = hidden_states[i + lag][0, 0, :].flatten()
                sim = 1 - cosine(h1, h2)
                sim_values.append(sim)
            similarities.append(np.mean(sim_values))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        lags = list(range(1, max_lag + 1))
        ax.plot(lags, similarities, marker='o', linewidth=2, markersize=4, color='#9b59b6')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random threshold')
        ax.set_xlabel('Time Lag (steps)')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('LSTM Memory Forgetting Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_forgetting_curve.png'), dpi=150)
        plt.close()
        print(f"✅ Forgetting curve saved")
        
        # Half-life
        half_life = None
        for i, sim in enumerate(similarities):
            if sim < 0.7:
                half_life = i + 1
                break
    
    # ==================== [3/4] CONFUSION MATRIX ====================
    print("\n🎯 Generating confusion matrix...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')
    
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, int(confusion_matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(action_names)
    ax.set_yticklabels(action_names)
    ax.set_xlabel('Model Action (Actual)', fontsize=12)
    ax.set_ylabel('Expected Action (Heuristic)', fontsize=12)
    ax.set_title('Confusion Matrix vs Simple Heuristic', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum() if confusion_matrix.sum() > 0 else 0
    print(f"✅ Confusion matrix saved (accuracy vs heuristic: {accuracy*100:.1f}%)")
    
    # ==================== [4/4] UNCERTAINTY ====================
    print("\n🎲 Analyzing uncertainty...")
    
    entropies = [d['entropy'] for d in uncertainty_data]
    max_probs = [d['max_prob'] for d in uncertainty_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Entropy histogram
    axes[0, 0].hist(entropies, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(entropies):.3f}')
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Decision Entropy Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Max probability histogram
    axes[0, 1].hist(max_probs, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(max_probs):.3f}')
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Entropy over time (first episode)
    first_ep_entropy = [d['entropy'] for d in uncertainty_data if d['episode'] == 0]
    axes[1, 0].plot(first_ep_entropy, linewidth=2, color='#9b59b6')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Decision Uncertainty Over Time (Episode 0)')
    axes[1, 0].grid(alpha=0.3)
    
    # Confidence categories
    high_cert = [d for d in uncertainty_data if d['max_prob'] > 0.8]
    medium_cert = [d for d in uncertainty_data if 0.5 <= d['max_prob'] <= 0.8]
    low_cert = [d for d in uncertainty_data if d['max_prob'] < 0.5]
    
    categories = ['High\n(>0.8)', 'Medium\n(0.5-0.8)', 'Low\n(<0.5)']
    counts = [len(high_cert), len(medium_cert), len(low_cert)]
    colors_cert = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = axes[1, 1].bar(categories, counts, color=colors_cert, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Number of Decisions')
    axes[1, 1].set_title('Confidence Categories')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/len(uncertainty_data)*100:.1f}%)',
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=150)
    plt.close()
    print(f"✅ Uncertainty analysis saved")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 LSTM ANALYSIS SUMMARY")
    print("="*80)
    if len(hidden_states) > 0:
        print(f"Average hidden state magnitude: {np.mean(hidden_means):.4f}")
        print(f"Average cell state magnitude: {np.mean(cell_means):.4f}")
    if half_life:
        print(f"Memory half-life: ~{half_life} steps")
    print(f"Most common bigram: {top_bigrams[0][0]} ({top_bigrams[0][1]} times)")
    print(f"Most common trigram: {top_trigrams[0][0]} ({top_trigrams[0][1]} times)")
    print(f"Confusion matrix accuracy: {accuracy*100:.1f}%")
    print(f"Average entropy: {np.mean(entropies):.3f}")
    print(f"Average confidence: {np.mean(max_probs):.3f}")
    print(f"High confidence decisions: {len(high_cert)}/{len(uncertainty_data)} ({len(high_cert)/len(uncertainty_data)*100:.1f}%)")