import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import csv
import os
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from matplotlib.patches import Rectangle


def analyze_temporal_patterns(model, env, output_dir, action_names, num_episodes=20):
    """
    🕐 TEMPORAL PATTERN ANALYSIS
    - Sequence dependency (jak długa historia wpływa na decyzję)
    - Forgetting curve (jak szybko LSTM "zapomina")
    - Action n-grams (typowe sekwencje akcji)
    - LSTM hidden state similarity over time
    """
    print("\n" + "="*80)
    print("🕐 TEMPORAL PATTERN ANALYSIS")
    print("="*80)
    
    policy = model.policy
    
    # Zbieranie danych
    all_episodes_data = []
    
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
            'positions': [],
            'food_distances': []
        }
        
        while not done and step < 200:
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
                
                # Position and food distance
                head_pos = obs.get('head_position', None)
                if head_pos is None:
                    # Extract from environment
                    head_pos = (0, 0)  # Placeholder
                
                food_dist = np.sqrt(obs['dx_head'][0]**2 + obs['dy_head'][0]**2)
                episode_data['positions'].append(head_pos)
                episode_data['food_distances'].append(food_dist)
                
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
    
    # ==================== ANALIZA 1: N-GRAMS (Action Sequences) ====================
    print("\n📊 Analyzing action n-grams...")
    
    bigrams = Counter()
    trigrams = Counter()
    
    for ep_data in all_episodes_data:
        actions = ep_data['actions']
        
        # Bigrams
        for i in range(len(actions) - 1):
            bigram = (action_names[actions[i]], action_names[actions[i+1]])
            bigrams[bigram] += 1
        
        # Trigrams
        for i in range(len(actions) - 2):
            trigram = (action_names[actions[i]], action_names[actions[i+1]], action_names[actions[i+2]])
            trigrams[trigram] += 1
    
    # Plot bigrams
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    top_bigrams = bigrams.most_common(10)
    bigram_labels = [f"{b[0]}→{b[1]}" for b, _ in top_bigrams]
    bigram_counts = [c for _, c in top_bigrams]
    
    axes[0].barh(bigram_labels, bigram_counts, color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Frequency')
    axes[0].set_title('Top 10 Action Bigrams (2-step sequences)')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot trigrams
    top_trigrams = trigrams.most_common(10)
    trigram_labels = [f"{t[0]}→{t[1]}→{t[2]}" for t, _ in top_trigrams]
    trigram_counts = [c for _, c in top_trigrams]
    
    axes[1].barh(trigram_labels, trigram_counts, color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Frequency')
    axes[1].set_title('Top 10 Action Trigrams (3-step sequences)')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    ngrams_path = os.path.join(output_dir, 'temporal_ngrams.png')
    plt.savefig(ngrams_path, dpi=150)
    plt.close()
    print(f"✅ N-grams saved: {ngrams_path}")
    
    # ==================== ANALIZA 2: HIDDEN STATE SIMILARITY (Forgetting Curve) ====================
    print("\n🧠 Analyzing LSTM memory persistence...")
    
    # Wybierz najdłuższy epizod
    longest_episode = max(all_episodes_data, key=lambda x: len(x['hidden_states']))
    hidden_states = longest_episode['hidden_states']
    
    if len(hidden_states) > 10:
        # Compute cosine similarity between hidden states at different time lags
        max_lag = min(50, len(hidden_states) - 1)
        similarities = []
        
        for lag in range(1, max_lag + 1):
            sim_values = []
            for i in range(len(hidden_states) - lag):
                h1 = hidden_states[i][0, 0, :].flatten()
                h2 = hidden_states[i + lag][0, 0, :].flatten()
                
                # Cosine similarity
                sim = 1 - cosine(h1, h2)
                sim_values.append(sim)
            
            similarities.append(np.mean(sim_values))
        
        # Plot forgetting curve
        fig, ax = plt.subplots(figsize=(12, 6))
        
        lags = list(range(1, max_lag + 1))
        ax.plot(lags, similarities, marker='o', linewidth=2, markersize=4, color='#9b59b6')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random similarity threshold')
        ax.set_xlabel('Time Lag (steps)')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('LSTM Memory Persistence (Forgetting Curve)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        forgetting_path = os.path.join(output_dir, 'temporal_forgetting_curve.png')
        plt.savefig(forgetting_path, dpi=150)
        plt.close()
        print(f"✅ Forgetting curve saved: {forgetting_path}")
        
        # Find memory half-life
        half_life = None
        for i, sim in enumerate(similarities):
            if sim < 0.7:  # Threshold for "significant forgetting"
                half_life = i + 1
                break
        
        if half_life:
            print(f"📉 Memory half-life: ~{half_life} steps (similarity drops below 0.7)")
        else:
            print(f"📈 Memory persists strongly for >{max_lag} steps!")
    
    # ==================== ANALIZA 3: SEQUENCE DEPENDENCY ====================
    print("\n🔗 Analyzing sequence dependency...")
    
    # Test: prediction consistency when given different history lengths
    # We'll simulate by comparing predictions with truncated history
    
    dependency_data = []
    
    for ep_idx, ep_data in enumerate(all_episodes_data[:5]):  # First 5 episodes
        if len(ep_data['actions']) < 20:
            continue
        
        # Pick a random point in the middle
        test_point = len(ep_data['actions']) // 2
        
        # Get observation at that point (we need to replay)
        # For simplicity, we'll analyze action probability variance over time
        probs = np.array(ep_data['action_probs'])
        
        # Entropy over time (measure of uncertainty)
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        
        dependency_data.append({
            'episode': ep_idx,
            'entropies': entropies
        })
    
    # Plot entropy over time (indicator of dependency)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for data in dependency_data:
        ax.plot(data['entropies'], alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Action Entropy')
    ax.set_title('Decision Uncertainty Over Time (Higher = More Uncertain)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    entropy_path = os.path.join(output_dir, 'temporal_entropy_evolution.png')
    plt.savefig(entropy_path, dpi=150)
    plt.close()
    print(f"✅ Entropy evolution saved: {entropy_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 TEMPORAL PATTERN SUMMARY")
    print("="*80)
    print(f"Most common bigram: {top_bigrams[0][0]} ({top_bigrams[0][1]} times)")
    print(f"Most common trigram: {top_trigrams[0][0]} ({top_trigrams[0][1]} times)")
    if half_life:
        print(f"LSTM memory half-life: ~{half_life} steps")
    print(f"Average decision entropy: {np.mean([np.mean(d['entropies']) for d in dependency_data]):.3f}")


def analyze_critical_moments(model, env, output_dir, action_names, num_episodes=30):
    """
    ⚠️ CRITICAL MOMENTS ANALYSIS
    - Near-death decisions (what happens 1-3 steps before collision)
    - Food acquisition patterns (optimal vs suboptimal paths)
    - Tight spaces behavior (when snake is long)
    - Heatmap of "death positions"
    """
    print("\n" + "="*80)
    print("⚠️ CRITICAL MOMENTS ANALYSIS")
    print("="*80)
    
    policy = model.policy
    
    # Data collection
    near_death_moments = []
    food_acquisitions = []
    death_positions = []
    tight_space_moments = []
    
    for episode_idx in range(num_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        step = 0
        
        episode_history = []
        
        while not done and step < 300:
            with torch.no_grad():
                # Get current state info
                snake_length = info.get('snake_length', 3)
                map_occupancy = info.get('map_occupancy', 0.0)
                
                # Collision detection
                front_coll = obs['front_coll'][0]
                left_coll = obs['left_coll'][0]
                right_coll = obs['right_coll'][0]
                
                # Predict
                action, lstm_states = model.predict(
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
                
                # Store moment
                moment = {
                    'step': step,
                    'action': action_idx,
                    'snake_length': snake_length,
                    'map_occupancy': map_occupancy,
                    'front_coll': front_coll,
                    'left_coll': left_coll,
                    'right_coll': right_coll,
                    'food_dist': np.sqrt(obs['dx_head'][0]**2 + obs['dy_head'][0]**2)
                }
                
                episode_history.append(moment)
            
            # Step
            prev_obs = obs
            prev_info = info
            obs, reward, done, truncated, info = env.step(action_idx)
            episode_starts = np.array([done or truncated], dtype=bool)
            
            # Check if food was eaten
            if reward > 5.0:  # Assuming food reward is positive and significant
                food_acquisitions.append({
                    'episode': episode_idx,
                    'step': step,
                    'snake_length': prev_info.get('snake_length', 3),
                    'steps_to_acquire': len(episode_history),
                    'efficiency': step / max(prev_info.get('score', 1), 1)
                })
            
            # Check for tight spaces
            if map_occupancy > 0.3:  # More than 30% of map occupied
                tight_space_moments.append(moment.copy())
            
            # Check if died
            if done or truncated:
                # Record death position (we don't have exact position, but we have occupancy)
                death_positions.append({
                    'episode': episode_idx,
                    'step': step,
                    'snake_length': prev_info.get('snake_length', 3),
                    'map_occupancy': map_occupancy,
                    'reason': info.get('termination_reason', 'unknown')
                })
                
                # Analyze last 5 steps before death (near-death moments)
                lookback = min(5, len(episode_history))
                for i in range(lookback):
                    moment_data = episode_history[-(lookback - i)]
                    moment_data['steps_before_death'] = lookback - i
                    moment_data['death_reason'] = info.get('termination_reason', 'unknown')
                    near_death_moments.append(moment_data)
            
            done = done or truncated
            step += 1
        
        if (episode_idx + 1) % 10 == 0:
            print(f"  Processed {episode_idx + 1}/{num_episodes} episodes")
    
    # ==================== ANALIZA 1: NEAR-DEATH DECISIONS ====================
    print("\n💀 Analyzing near-death decisions...")
    
    if len(near_death_moments) > 0:
        # Group by steps_before_death
        steps_before = [m['steps_before_death'] for m in near_death_moments]
        actions_taken = [m['action'] for m in near_death_moments]
        
        # Collision awareness
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Actions taken in last 5 steps before death
        steps_range = range(1, 6)
        action_counts = {step: {0: 0, 1: 0, 2: 0} for step in steps_range}
        
        for moment in near_death_moments:
            step = moment['steps_before_death']
            action = moment['action']
            action_counts[step][action] += 1
        
        x = np.arange(len(steps_range))
        width = 0.25
        
        left_counts = [action_counts[s][0] for s in steps_range]
        straight_counts = [action_counts[s][1] for s in steps_range]
        right_counts = [action_counts[s][2] for s in steps_range]
        
        axes[0, 0].bar(x - width, left_counts, width, label='Left', color='#e74c3c', alpha=0.8)
        axes[0, 0].bar(x, straight_counts, width, label='Straight', color='#3498db', alpha=0.8)
        axes[0, 0].bar(x + width, right_counts, width, label='Right', color='#2ecc71', alpha=0.8)
        axes[0, 0].set_xlabel('Steps Before Death')
        axes[0, 0].set_ylabel('Action Frequency')
        axes[0, 0].set_title('Actions Taken Before Death')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(steps_range)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Collision detection in near-death moments
        front_colls = [m['front_coll'] for m in near_death_moments]
        left_colls = [m['left_coll'] for m in near_death_moments]
        right_colls = [m['right_coll'] for m in near_death_moments]
        
        axes[0, 1].hist([front_colls, left_colls, right_colls], bins=2, 
                       label=['Front', 'Left', 'Right'], 
                       color=['#e74c3c', '#f39c12', '#2ecc71'], 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Collision Detected')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Collision Awareness Before Death')
        axes[0, 1].legend()
        axes[0, 1].set_xticks([0.25, 0.75])
        axes[0, 1].set_xticklabels(['No', 'Yes'])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Death reasons
        death_reasons = [m['death_reason'] for m in near_death_moments]
        reason_counts = Counter(death_reasons)
        
        axes[1, 0].bar(reason_counts.keys(), reason_counts.values(), 
                      color='#9b59b6', alpha=0.8, edgecolor='black')
        axes[1, 0].set_xlabel('Death Reason')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Death Reasons')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Snake length at death
        death_lengths = [d['snake_length'] for d in death_positions]
        
        axes[1, 1].hist(death_lengths, bins=20, color='#e67e22', alpha=0.8, edgecolor='black')
        axes[1, 1].axvline(np.mean(death_lengths), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(death_lengths):.1f}')
        axes[1, 1].set_xlabel('Snake Length at Death')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Snake Length at Death')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        near_death_path = os.path.join(output_dir, 'critical_near_death.png')
        plt.savefig(near_death_path, dpi=150)
        plt.close()
        print(f"✅ Near-death analysis saved: {near_death_path}")
    
    # ==================== ANALIZA 2: FOOD ACQUISITION PATTERNS ====================
    print("\n🍎 Analyzing food acquisition patterns...")
    
    if len(food_acquisitions) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Efficiency over snake length
        lengths = [f['snake_length'] for f in food_acquisitions]
        efficiencies = [f['efficiency'] for f in food_acquisitions]
        
        axes[0].scatter(lengths, efficiencies, alpha=0.6, s=50, color='#e74c3c', edgecolor='black')
        axes[0].set_xlabel('Snake Length')
        axes[0].set_ylabel('Steps per Apple (lower = better)')
        axes[0].set_title('Food Acquisition Efficiency vs Snake Length')
        axes[0].grid(alpha=0.3)
        
        # Add trendline
        z = np.polyfit(lengths, efficiencies, 1)
        p = np.poly1d(z)
        axes[0].plot(sorted(lengths), p(sorted(lengths)), 
                    color='blue', linestyle='--', linewidth=2, label='Trend')
        axes[0].legend()
        
        # Plot 2: Distribution of acquisition times
        acquisition_steps = [f['step'] for f in food_acquisitions]
        
        axes[1].hist(acquisition_steps, bins=30, color='#2ecc71', alpha=0.8, edgecolor='black')
        axes[1].axvline(np.mean(acquisition_steps), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(acquisition_steps):.1f}')
        axes[1].set_xlabel('Step in Episode')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('When Does Model Acquire Food?')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        food_path = os.path.join(output_dir, 'critical_food_acquisition.png')
        plt.savefig(food_path, dpi=150)
        plt.close()
        print(f"✅ Food acquisition analysis saved: {food_path}")
    
    # ==================== ANALIZA 3: TIGHT SPACES BEHAVIOR ====================
    print("\n🔒 Analyzing tight spaces behavior...")
    
    if len(tight_space_moments) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Actions in tight spaces
        tight_actions = [m['action'] for m in tight_space_moments]
        action_dist = Counter(tight_actions)
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        axes[0].bar([action_names[i] for i in range(3)], 
                   [action_dist[i] for i in range(3)],
                   color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Action')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Action Distribution in Tight Spaces (>30% occupancy)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Collision awareness in tight spaces
        tight_front = [m['front_coll'] for m in tight_space_moments]
        tight_left = [m['left_coll'] for m in tight_space_moments]
        tight_right = [m['right_coll'] for m in tight_space_moments]
        
        collision_data = [
            sum(tight_front) / len(tight_front) if tight_front else 0,
            sum(tight_left) / len(tight_left) if tight_left else 0,
            sum(tight_right) / len(tight_right) if tight_right else 0
        ]
        
        axes[1].bar(['Front', 'Left', 'Right'], collision_data,
                   color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Average Collision Rate')
        axes[1].set_title('Collision Frequency in Tight Spaces')
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        tight_path = os.path.join(output_dir, 'critical_tight_spaces.png')
        plt.savefig(tight_path, dpi=150)
        plt.close()
        print(f"✅ Tight spaces analysis saved: {tight_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 CRITICAL MOMENTS SUMMARY")
    print("="*80)
    if near_death_moments:
        most_common_action_before_death = Counter([m['action'] for m in near_death_moments]).most_common(1)[0]
        print(f"Most common action before death: {action_names[most_common_action_before_death[0]]} ({most_common_action_before_death[1]} times)")
    if death_positions:
        print(f"Average snake length at death: {np.mean([d['snake_length'] for d in death_positions]):.1f}")
        print(f"Average map occupancy at death: {np.mean([d['map_occupancy'] for d in death_positions]):.1f}%")
    if food_acquisitions:
        print(f"Average food acquisition efficiency: {np.mean([f['efficiency'] for f in food_acquisitions]):.2f} steps/apple")


def analyze_feature_importance(model, env, output_dir, action_names, num_samples=100):
    """
    🎯 FEATURE IMPORTANCE ANALYSIS
    - Ablation study (disable CNN vs scalars)
    - Gradient-based importance
    - Feature correlation with rewards
    """
    print("\n" + "="*80)
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # ==================== ABLATION STUDY ====================
    print("\n🔬 Running ablation study...")
    
    def run_episode_with_ablation(ablation_type='none'):
        """
        ablation_type: 'none', 'cnn_only', 'scalar_only', 'no_cnn', 'no_scalar'
        """
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 200:
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
                
                # Apply ablation
                if ablation_type == 'no_cnn':
                    # Zero out image
                    obs_tensor['image'] = torch.zeros_like(obs_tensor['image'])
                elif ablation_type == 'no_scalar':
                    # Zero out scalars
                    obs_tensor['direction'] = torch.zeros_like(obs_tensor['direction'])
                    obs_tensor['dx_head'] = torch.zeros_like(obs_tensor['dx_head'])
                    obs_tensor['dy_head'] = torch.zeros_like(obs_tensor['dy_head'])
                    obs_tensor['front_coll'] = torch.zeros_like(obs_tensor['front_coll'])
                    obs_tensor['left_coll'] = torch.zeros_like(obs_tensor['left_coll'])
                    obs_tensor['right_coll'] = torch.zeros_like(obs_tensor['right_coll'])
                elif ablation_type == 'cnn_only':
                    # Keep only CNN
                    obs_tensor['direction'] = torch.zeros_like(obs_tensor['direction'])
                    obs_tensor['dx_head'] = torch.zeros_like(obs_tensor['dx_head'])
                    obs_tensor['dy_head'] = torch.zeros_like(obs_tensor['dy_head'])
                    obs_tensor['front_coll'] = torch.zeros_like(obs_tensor['front_coll'])
                    obs_tensor['left_coll'] = torch.zeros_like(obs_tensor['left_coll'])
                    obs_tensor['right_coll'] = torch.zeros_like(obs_tensor['right_coll'])
                elif ablation_type == 'scalar_only':
                    # Keep only scalars
                    obs_tensor['image'] = torch.zeros_like(obs_tensor['image'])
                
                # Convert tensors back to numpy for model.predict()
                obs_numpy = {}
                for k, v in obs_tensor.items():
                    obs_numpy[k] = v.cpu().numpy()
                
                action, lstm_states = model.predict(
                    obs_numpy, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=True
                )
                
                if torch.is_tensor(action):
                    action_idx = int(action.item())
                elif isinstance(action, np.ndarray):
                    action_idx = int(action.item())
                else:
                    action_idx = int(action)
                action_idx = int(np.clip(action_idx, 0, len(action_names) - 1))
            
            obs, reward, done, truncated, info = env.step(action_idx)
            total_reward += reward
            episode_starts = np.array([done or truncated], dtype=bool)
            done = done or truncated
            step += 1
        
        return total_reward, step, info.get('score', 0)
    
    # Run ablation experiments
    ablation_types = ['none', 'no_cnn', 'no_scalar', 'cnn_only', 'scalar_only']
    ablation_results = {abl: {'rewards': [], 'steps': [], 'scores': []} for abl in ablation_types}
    
    episodes_per_ablation = max(10, num_samples // 5)
    
    for abl_type in ablation_types:
        print(f"  Testing ablation: {abl_type}...")
        for _ in range(episodes_per_ablation):
            reward, steps, score = run_episode_with_ablation(abl_type)
            ablation_results[abl_type]['rewards'].append(reward)
            ablation_results[abl_type]['steps'].append(steps)
            ablation_results[abl_type]['scores'].append(score)
    
    # Plot ablation results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    labels = ['Full Model', 'No CNN', 'No Scalars', 'CNN Only', 'Scalars Only']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    # Plot 1: Average rewards
    avg_rewards = [np.mean(ablation_results[abl]['rewards']) for abl in ablation_types]
    std_rewards = [np.std(ablation_results[abl]['rewards']) for abl in ablation_types]
    
    bars = axes[0].bar(labels, avg_rewards, yerr=std_rewards, capsize=5, 
                       color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Average Total Reward')
    axes[0].set_title('Ablation Study: Rewards')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Highlight baseline
    axes[0].axhline(avg_rewards[0], color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].legend()
    
    # Plot 2: Average scores
    avg_scores = [np.mean(ablation_results[abl]['scores']) for abl in ablation_types]
    std_scores = [np.std(ablation_results[abl]['scores']) for abl in ablation_types]
    
    axes[1].bar(labels, avg_scores, yerr=std_scores, capsize=5, 
               color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Average Score (food collected)')
    axes[1].set_title('Ablation Study: Scores')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(avg_scores[0], color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[1].legend()
    
    # Plot 3: Performance drop
    baseline_reward = avg_rewards[0]
    performance_drops = [(baseline_reward - r) / baseline_reward * 100 for r in avg_rewards]
    
    axes[2].bar(labels, performance_drops, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Performance Drop (%)')
    axes[2].set_title('Ablation Study: Performance Impact')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    ablation_path = os.path.join(output_dir, 'feature_ablation_study.png')
    plt.savefig(ablation_path, dpi=150)
    plt.close()
    print(f"✅ Ablation study saved: {ablation_path}")
    
    # ==================== GRADIENT-BASED IMPORTANCE ====================
    print("\n🔍 Computing gradient-based feature importance...")
    
    gradient_importances = {
        'cnn': [],
        'direction': [],
        'dx_head': [],
        'dy_head': [],
        'front_coll': [],
        'left_coll': [],
        'right_coll': []
    }
    
    for sample_idx in range(min(50, num_samples)):
        obs, _ = env.reset()
        
        # Random steps
        for _ in range(np.random.randint(0, 20)):
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
        
        # Prepare observation with gradients
        obs_grad = {}
        for k, v in obs.items():
            v_np = v if isinstance(v, np.ndarray) else np.array(v)
            v_tensor = torch.tensor(v_np, dtype=torch.float32, device=policy.device, requires_grad=True)
            
            if k == 'image':
                if v_tensor.ndim == 3:
                    v_tensor = v_tensor.unsqueeze(0)
            else:
                if v_tensor.ndim == 1:
                    v_tensor = v_tensor.unsqueeze(0)
            
            v_tensor.retain_grad()
            obs_grad[k] = v_tensor
        
        # Set model to training mode temporarily for backward pass
        policy.train()
        
        # Forward pass
        features = features_extractor(obs_grad)
        
        lstm_states_init = (
            torch.zeros(policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size, device=policy.device),
            torch.zeros(policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size, device=policy.device)
        )
        
        features_seq = features.unsqueeze(1)
        lstm_output, _ = policy.lstm_actor(features_seq, lstm_states_init)
        latent_pi = lstm_output.squeeze(1)
        latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
        logits = policy.action_net(latent_pi_mlp)
        
        # Get max action
        action_idx = torch.argmax(logits, dim=1)
        selected_logit = logits[0, action_idx]
        
        # Backward
        policy.zero_grad()
        selected_logit.backward()
        
        # Return to eval mode
        policy.eval()
        
        # Collect gradients
        if obs_grad['image'].grad is not None:
            gradient_importances['cnn'].append(obs_grad['image'].grad.abs().mean().item())
        
        scalar_keys = ['direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']
        for key in scalar_keys:
            if obs_grad[key].grad is not None:
                gradient_importances[key].append(obs_grad[key].grad.abs().mean().item())
    
    # Plot gradient importance
    fig, ax = plt.subplots(figsize=(12, 6))
    
    feature_names = list(gradient_importances.keys())
    avg_importances = [np.mean(gradient_importances[f]) if gradient_importances[f] else 0 
                       for f in feature_names]
    std_importances = [np.std(gradient_importances[f]) if gradient_importances[f] else 0 
                       for f in feature_names]
    
    colors_imp = ['#e74c3c'] + ['#3498db'] * 6
    
    bars = ax.bar(feature_names, avg_importances, yerr=std_importances, capsize=5,
                  color=colors_imp, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Gradient Magnitude')
    ax.set_title('Gradient-Based Feature Importance')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    gradient_path = os.path.join(output_dir, 'feature_gradient_importance.png')
    plt.savefig(gradient_path, dpi=150)
    plt.close()
    print(f"✅ Gradient importance saved: {gradient_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 FEATURE IMPORTANCE SUMMARY")
    print("="*80)
    print(f"Baseline performance: {avg_rewards[0]:.2f} reward, {avg_scores[0]:.2f} score")
    print(f"Without CNN: {performance_drops[1]:.1f}% drop")
    print(f"Without Scalars: {performance_drops[2]:.1f}% drop")
    print(f"CNN only: {performance_drops[3]:.1f}% drop")
    print(f"Scalars only: {performance_drops[4]:.1f}% drop")
    print(f"\nMost important feature (gradient): {feature_names[np.argmax(avg_importances)]} ({max(avg_importances):.6f})")
    
    # Save results to CSV
    results_csv = os.path.join(output_dir, 'feature_importance_results.csv')
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ablation_type', 'avg_reward', 'std_reward', 'avg_score', 'std_score', 'performance_drop_pct'])
        for i, abl in enumerate(ablation_types):
            writer.writerow([
                labels[i],
                avg_rewards[i],
                std_rewards[i],
                avg_scores[i],
                std_scores[i],
                performance_drops[i]
            ])
    print(f"✅ Results saved to CSV: {results_csv}")


def analyze_bottleneck_architecture(model, env, output_dir, num_samples=100):
    """
    🔧 BOTTLENECK ARCHITECTURE ANALYSIS
    - Information flow through bottleneck stages
    - Main path vs residual path contribution
    - Alpha evolution (learnable weight)
    - Layer-wise feature visualization
    """
    print("\n" + "="*80)
    print("🔧 BOTTLENECK ARCHITECTURE ANALYSIS")
    print("="*80)
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # Check if model has bottleneck architecture
    if not hasattr(features_extractor, 'cnn_compress') or not hasattr(features_extractor, 'cnn_residual'):
        print("⚠️  Model does not have bottleneck architecture!")
        return
    
    # ==================== DATA COLLECTION ====================
    print("\n📊 Collecting bottleneck statistics...")
    
    bottleneck_data = {
        'cnn_raw': [],
        'main_path': [],
        'residual_path': [],
        'final_output': [],
        'alpha_values': [],
        'main_contribution': [],
        'residual_contribution': []
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
            
            # Forward through CNN
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # Conv layers
            x = features_extractor.conv1(image)
            x = features_extractor.bn1(x)
            x = torch.nn.functional.gelu(x)
            
            x = features_extractor.conv2(x)
            x = features_extractor.bn2(x)
            x = torch.nn.functional.gelu(x)
            x = features_extractor.dropout2(x)
            
            # Conv3 (if exists)
            if hasattr(features_extractor, 'has_conv3') and features_extractor.has_conv3:
                identity = features_extractor.residual_proj(x)
                
                local = features_extractor.conv3_local(x)
                local = features_extractor.bn3_local(local)
                
                global_ctx = features_extractor.conv3_global(x)
                global_ctx = features_extractor.bn3_global(global_ctx)
                
                x = torch.cat([local, global_ctx], dim=1)
                x = features_extractor.dropout3(x)
                x = torch.nn.functional.gelu(x)
                
                x = x + identity
            
            cnn_raw = features_extractor.flatten(x).float()
            
            # Bottleneck paths
            main_path = features_extractor.cnn_compress(cnn_raw)
            residual_path = features_extractor.cnn_residual(cnn_raw)
            
            # Get alpha
            if hasattr(features_extractor, 'alpha'):
                if features_extractor.alpha_mode == 'learnable':
                    alpha = torch.sigmoid(features_extractor.alpha).item()
                else:
                    alpha = features_extractor.alpha.item()
            else:
                alpha = 0.5  # Default
            
            # Final output
            final_output = alpha * main_path + (1 - alpha) * residual_path
            
            # Store statistics
            bottleneck_data['cnn_raw'].append(cnn_raw.abs().mean().item())
            bottleneck_data['main_path'].append(main_path.abs().mean().item())
            bottleneck_data['residual_path'].append(residual_path.abs().mean().item())
            bottleneck_data['final_output'].append(final_output.abs().mean().item())
            bottleneck_data['alpha_values'].append(alpha)
            bottleneck_data['main_contribution'].append((alpha * main_path.abs().mean()).item())
            bottleneck_data['residual_contribution'].append(((1-alpha) * residual_path.abs().mean()).item())
        
        if (sample_idx + 1) % 20 == 0:
            print(f"  Processed {sample_idx + 1}/{num_samples} samples")
    
    # ==================== VISUALIZATION ====================
    
    # Plot 1: Information Flow
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Activation magnitudes
    ax = axes[0, 0]
    data_to_plot = [
        bottleneck_data['cnn_raw'],
        bottleneck_data['main_path'],
        bottleneck_data['residual_path'],
        bottleneck_data['final_output']
    ]
    labels_plot = ['CNN Raw', 'Main Path', 'Residual Path', 'Final Output']
    colors_plot = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    
    bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Activation Magnitude')
    ax.set_title('Information Flow Through Bottleneck')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Alpha distribution
    ax = axes[0, 1]
    ax.hist(bottleneck_data['alpha_values'], bins=30, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.axvline(np.mean(bottleneck_data['alpha_values']), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {np.mean(bottleneck_data['alpha_values']):.3f}")
    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Alpha Distribution (Main Path Weight)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Contribution comparison
    ax = axes[1, 0]
    contribution_data = [
        bottleneck_data['main_contribution'],
        bottleneck_data['residual_contribution']
    ]
    bp2 = ax.boxplot(contribution_data, labels=['Main Path', 'Residual Path'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#3498db')
    bp2['boxes'][1].set_facecolor('#f39c12')
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    
    ax.set_ylabel('Weighted Contribution')
    ax.set_title('Main vs Residual Path Contribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Compression ratio visualization
    ax = axes[1, 1]
    compression_ratios = np.array(bottleneck_data['cnn_raw']) / (np.array(bottleneck_data['main_path']) + 1e-8)
    ax.hist(compression_ratios, bins=30, color='#e67e22', alpha=0.8, edgecolor='black')
    ax.axvline(np.mean(compression_ratios), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {np.mean(compression_ratios):.2f}x")
    ax.set_xlabel('Compression Ratio (Raw / Compressed)')
    ax.set_ylabel('Frequency')
    ax.set_title('Effective Compression Ratio')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    bottleneck_flow_path = os.path.join(output_dir, 'bottleneck_information_flow.png')
    plt.savefig(bottleneck_flow_path, dpi=150)
    plt.close()
    print(f"✅ Bottleneck flow analysis saved: {bottleneck_flow_path}")
    
    # Plot 2: Path Importance Over Activations
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scatter_x = bottleneck_data['main_path']
    scatter_y = bottleneck_data['residual_path']
    scatter_c = bottleneck_data['alpha_values']
    
    scatter = ax.scatter(scatter_x, scatter_y, c=scatter_c, cmap='viridis', 
                        s=50, alpha=0.6, edgecolor='black')
    ax.plot([0, max(max(scatter_x), max(scatter_y))], 
            [0, max(max(scatter_x), max(scatter_y))], 
            'r--', alpha=0.5, label='Equal contribution')
    
    ax.set_xlabel('Main Path Activation')
    ax.set_ylabel('Residual Path Activation')
    ax.set_title('Main vs Residual Path Activations (color = alpha)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha Value')
    
    plt.tight_layout()
    bottleneck_scatter_path = os.path.join(output_dir, 'bottleneck_path_comparison.png')
    plt.savefig(bottleneck_scatter_path, dpi=150)
    plt.close()
    print(f"✅ Path comparison saved: {bottleneck_scatter_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 BOTTLENECK ARCHITECTURE SUMMARY")
    print("="*80)
    
    avg_alpha = np.mean(bottleneck_data['alpha_values'])
    std_alpha = np.std(bottleneck_data['alpha_values'])
    
    avg_main = np.mean(bottleneck_data['main_contribution'])
    avg_residual = np.mean(bottleneck_data['residual_contribution'])
    
    total_contribution = avg_main + avg_residual
    main_pct = (avg_main / total_contribution) * 100 if total_contribution > 0 else 0
    residual_pct = (avg_residual / total_contribution) * 100 if total_contribution > 0 else 0
    
    avg_compression = np.mean(compression_ratios)
    
    print(f"Average Alpha: {avg_alpha:.3f} ± {std_alpha:.3f}")
    print(f"Main path contribution: {main_pct:.1f}%")
    print(f"Residual path contribution: {residual_pct:.1f}%")
    print(f"Average compression ratio: {avg_compression:.2f}x")
    print(f"CNN raw activation: {np.mean(bottleneck_data['cnn_raw']):.4f}")
    print(f"Final output activation: {np.mean(bottleneck_data['final_output']):.4f}")
    
    if avg_alpha > 0.7:
        print("\n💡 Insight: Model heavily relies on MAIN PATH (deep compression)")
    elif avg_alpha < 0.3:
        print("\n💡 Insight: Model heavily relies on RESIDUAL PATH (direct projection)")
    else:
        print("\n💡 Insight: Balanced usage of both paths")
    
    # Save statistics
    stats_csv = os.path.join(output_dir, 'bottleneck_statistics.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'min', 'max'])
        
        for key in ['cnn_raw', 'main_path', 'residual_path', 'final_output', 'alpha_values']:
            data = bottleneck_data[key]
            writer.writerow([
                key,
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data)
            ])