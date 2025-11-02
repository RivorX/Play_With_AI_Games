"""
🎯 PERFORMANCE & BEHAVIOR ANALYSIS MODULE

Scalona analiza wydajności:
- Critical moments (near-death, food acquisition, tight spaces)
- Feature importance (ablation study)

Wszystkie wyniki w jednym katalogu: 05_performance/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import Counter

def analyze_performance_metrics(model, env, output_dir, action_names, num_episodes=30, num_samples=100):
    """
    🎯 KOMPLEKSOWA ANALIZA WYDAJNOŚCI (FIXED)
    Łączy: critical moments + ablation study
    """
    print("\n" + "="*80)
    print("🎯 PERFORMANCE & BEHAVIOR ANALYSIS")
    print("="*80)
    
    policy = model.policy
    
    # ==================== [1/2] CRITICAL MOMENTS ====================
    print("\n⚠️  Analyzing critical moments...")
    
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
                snake_length = info.get('snake_length', 3)
                map_occupancy = info.get('map_occupancy', 0.0)
                
                front_coll = obs['front_coll'][0]
                left_coll = obs['left_coll'][0]
                right_coll = obs['right_coll'][0]
                
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
                
                if lstm_states is not None:
                    lstm_states_tensor = (
                        torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device),
                        torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device)
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
                lstm_output, new_lstm_states_tensor = policy.lstm_actor(features_seq, lstm_states_tensor)
                latent_pi = lstm_output.squeeze(1)
                latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
                logits = policy.action_net(latent_pi_mlp)
                action_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
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
                
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
                max_prob = np.max(action_probs)
                
                moment = {
                    'step': step,
                    'action': action_idx,
                    'action_probs': action_probs.copy(),
                    'entropy': entropy,
                    'confidence': max_prob,
                    'snake_length': snake_length,
                    'map_occupancy': map_occupancy,
                    'front_coll': front_coll,
                    'left_coll': left_coll,
                    'right_coll': right_coll,
                    'food_dist': np.sqrt(obs['dx_head'][0]**2 + obs['dy_head'][0]**2)
                }
                
                episode_history.append(moment)
            
            prev_obs = obs
            prev_info = info
            obs, reward, done, truncated, info = env.step(action_idx)
            episode_starts = np.array([done or truncated], dtype=bool)
            
            # Food eaten
            if reward > 5.0:
                food_acquisitions.append({
                    'episode': episode_idx,
                    'step': step,
                    'snake_length': prev_info.get('snake_length', 3),
                    'steps_to_acquire': len(episode_history),
                    'efficiency': step / max(prev_info.get('score', 1), 1)
                })
            
            # Tight spaces
            if map_occupancy > 0.3:
                tight_space_moments.append(moment.copy())
            
            # Death
            if done or truncated:
                death_positions.append({
                    'episode': episode_idx,
                    'step': step,
                    'snake_length': prev_info.get('snake_length', 3),
                    'map_occupancy': map_occupancy,
                    'reason': info.get('termination_reason', 'unknown')
                })
                
                # Last 5 steps before death
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
    
    # Plot critical moments
    if len(near_death_moments) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Actions before death
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
        
        # 🔧 FIX: Collision awareness - CORRECTED VERSION
        front_colls = [m['front_coll'] for m in near_death_moments]
        left_colls = [m['left_coll'] for m in near_death_moments]
        right_colls = [m['right_coll'] for m in near_death_moments]
        
        # Count occurrences properly
        front_no = sum(1 for c in front_colls if c < 0.5)
        front_yes = sum(1 for c in front_colls if c >= 0.5)
        left_no = sum(1 for c in left_colls if c < 0.5)
        left_yes = sum(1 for c in left_colls if c >= 0.5)
        right_no = sum(1 for c in right_colls if c < 0.5)
        right_yes = sum(1 for c in right_colls if c >= 0.5)
        
        # Plot as grouped bar chart
        x_coll = np.arange(2)
        width_coll = 0.25
        
        axes[0, 1].bar(x_coll - width_coll, [front_no, front_yes], width_coll, 
                      label='Front', color='#e74c3c', alpha=0.8, edgecolor='black')
        axes[0, 1].bar(x_coll, [left_no, left_yes], width_coll, 
                      label='Left', color='#f39c12', alpha=0.8, edgecolor='black')
        axes[0, 1].bar(x_coll + width_coll, [right_no, right_yes], width_coll, 
                      label='Right', color='#2ecc71', alpha=0.8, edgecolor='black')
        
        axes[0, 1].set_xlabel('Collision Detected')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Collision Awareness Before Death')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(x_coll)
        axes[0, 1].set_xticklabels(['No Collision', 'Collision Detected'])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add percentages on bars
        total_moments = len(near_death_moments)
        for i, (no_val, yes_val) in enumerate([(front_no, front_yes), (left_no, left_yes), (right_no, right_yes)]):
            x_pos = x_coll + (i - 1) * width_coll
            
            # No collision percentage
            pct_no = (no_val / total_moments) * 100
            if no_val > 0:
                axes[0, 1].text(x_pos[0], no_val + 0.5, f'{pct_no:.0f}%', 
                               ha='center', va='bottom', fontsize=8)
            
            # Yes collision percentage
            pct_yes = (yes_val / total_moments) * 100
            if yes_val > 0:
                axes[0, 1].text(x_pos[1], yes_val + 0.5, f'{pct_yes:.0f}%', 
                               ha='center', va='bottom', fontsize=8)
        
        # Decision confidence before death
        entropy_by_step = {step: [] for step in steps_range}
        for moment in near_death_moments:
            entropy_by_step[moment['steps_before_death']].append(moment['entropy'])
        
        entropy_data = [entropy_by_step[s] for s in steps_range]
        bp = axes[1, 0].boxplot(entropy_data, labels=list(steps_range), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#9b59b6')
            patch.set_alpha(0.7)
        
        axes[1, 0].set_xlabel('Steps Before Death')
        axes[1, 0].set_ylabel('Decision Entropy')
        axes[1, 0].set_title('Decision Confidence Before Death (Higher = Uncertain)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(np.log(3), color='red', linestyle='--', alpha=0.5, label='Max entropy')
        axes[1, 0].legend()
        
        # Death conditions
        death_lengths = [d['snake_length'] for d in death_positions]
        death_occupancy = [d['map_occupancy'] for d in death_positions]
        
        scatter = axes[1, 1].scatter(death_lengths, death_occupancy, 
                                     c=death_occupancy, cmap='Reds', 
                                     s=100, alpha=0.6, edgecolor='black')
        axes[1, 1].set_xlabel('Snake Length at Death')
        axes[1, 1].set_ylabel('Map Occupancy at Death')
        axes[1, 1].set_title('Death Conditions: Length vs Occupancy')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].axvline(np.mean(death_lengths), color='blue', linestyle='--', 
                          alpha=0.5, label=f'Mean length: {np.mean(death_lengths):.1f}')
        axes[1, 1].axhline(np.mean(death_occupancy), color='green', linestyle='--', 
                          alpha=0.5, label=f'Mean occupancy: {np.mean(death_occupancy):.2f}')
        axes[1, 1].legend()
        plt.colorbar(scatter, ax=axes[1, 1], label='Occupancy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'critical_near_death.png'), dpi=150)
        plt.close()
        print(f"✅ Near-death analysis saved")
    
    # Food acquisition
    if len(food_acquisitions) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        lengths = [f['snake_length'] for f in food_acquisitions]
        efficiencies = [f['efficiency'] for f in food_acquisitions]
        
        axes[0].scatter(lengths, efficiencies, alpha=0.6, s=50, color='#e74c3c', edgecolor='black')
        axes[0].set_xlabel('Snake Length')
        axes[0].set_ylabel('Steps per Apple (lower = better)')
        axes[0].set_title('Food Acquisition Efficiency vs Snake Length')
        axes[0].grid(alpha=0.3)
        
        if len(lengths) > 1:
            z = np.polyfit(lengths, efficiencies, 1)
            p = np.poly1d(z)
            axes[0].plot(sorted(lengths), p(sorted(lengths)), 
                        color='blue', linestyle='--', linewidth=2, label='Trend')
            axes[0].legend()
        
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
        plt.savefig(os.path.join(output_dir, 'critical_food_acquisition.png'), dpi=150)
        plt.close()
        print(f"✅ Food acquisition analysis saved")
    
    # Tight spaces
    if len(tight_space_moments) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
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
        plt.savefig(os.path.join(output_dir, 'critical_tight_spaces.png'), dpi=150)
        plt.close()
        print(f"✅ Tight spaces analysis saved")
    
    # ==================== [2/2] FEATURE IMPORTANCE (ABLATION) ====================
    print("\n🔬 Running ablation study...")
    
    def run_episode_with_ablation(ablation_type='none'):
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
                    obs_tensor['image'] = torch.zeros_like(obs_tensor['image'])
                elif ablation_type == 'no_scalar':
                    obs_tensor['direction'] = torch.zeros_like(obs_tensor['direction'])
                    obs_tensor['dx_head'] = torch.zeros_like(obs_tensor['dx_head'])
                    obs_tensor['dy_head'] = torch.zeros_like(obs_tensor['dy_head'])
                    obs_tensor['front_coll'] = torch.zeros_like(obs_tensor['front_coll'])
                    obs_tensor['left_coll'] = torch.zeros_like(obs_tensor['left_coll'])
                    obs_tensor['right_coll'] = torch.zeros_like(obs_tensor['right_coll'])
                elif ablation_type == 'cnn_only':
                    obs_tensor['direction'] = torch.zeros_like(obs_tensor['direction'])
                    obs_tensor['dx_head'] = torch.zeros_like(obs_tensor['dx_head'])
                    obs_tensor['dy_head'] = torch.zeros_like(obs_tensor['dy_head'])
                    obs_tensor['front_coll'] = torch.zeros_like(obs_tensor['front_coll'])
                    obs_tensor['left_coll'] = torch.zeros_like(obs_tensor['left_coll'])
                    obs_tensor['right_coll'] = torch.zeros_like(obs_tensor['right_coll'])
                elif ablation_type == 'scalar_only':
                    obs_tensor['image'] = torch.zeros_like(obs_tensor['image'])
                
                # Convert back to numpy
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
    
    ablation_types = ['none', 'no_cnn', 'no_scalar', 'cnn_only', 'scalar_only']
    ablation_results = {abl: {'rewards': [], 'steps': [], 'scores': []} for abl in ablation_types}
    
    episodes_per_ablation = max(10, num_samples // 5)
    
    for abl_type in ablation_types:
        print(f"  Testing: {abl_type}...")
        for _ in range(episodes_per_ablation):
            reward, steps, score = run_episode_with_ablation(abl_type)
            ablation_results[abl_type]['rewards'].append(reward)
            ablation_results[abl_type]['steps'].append(steps)
            ablation_results[abl_type]['scores'].append(score)
    
    # Plot ablation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    labels = ['Full Model', 'No CNN', 'No Scalars', 'CNN Only', 'Scalars Only']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    avg_rewards = [np.mean(ablation_results[abl]['rewards']) for abl in ablation_types]
    std_rewards = [np.std(ablation_results[abl]['rewards']) for abl in ablation_types]
    
    axes[0].bar(labels, avg_rewards, yerr=std_rewards, capsize=5, 
               color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Average Total Reward')
    axes[0].set_title('Ablation Study: Rewards')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(avg_rewards[0], color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].legend()
    
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
    
    baseline_reward = avg_rewards[0]
    performance_drops = [(baseline_reward - r) / baseline_reward * 100 if baseline_reward != 0 else 0 for r in avg_rewards]
    
    axes[2].bar(labels, performance_drops, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Performance Drop (%)')
    axes[2].set_title('Ablation Study: Performance Impact')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_ablation_study.png'), dpi=150)
    plt.close()
    print(f"✅ Ablation study saved")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("📋 PERFORMANCE SUMMARY")
    print("="*80)
    
    if near_death_moments:
        most_common_action = Counter([m['action'] for m in near_death_moments]).most_common(1)[0]
        avg_entropy = np.mean([m['entropy'] for m in near_death_moments])
        
        # 🔧 NEW: Collision detection stats
        total_moments = len(near_death_moments)
        front_detected = sum(1 for m in near_death_moments if m['front_coll'] >= 0.5)
        left_detected = sum(1 for m in near_death_moments if m['left_coll'] >= 0.5)
        right_detected = sum(1 for m in near_death_moments if m['right_coll'] >= 0.5)
        
        print(f"Most common action before death: {action_names[most_common_action[0]]} ({most_common_action[1]} times)")
        print(f"Average entropy before death: {avg_entropy:.3f}")
        print(f"\n🔍 Collision Detection Stats (before death):")
        print(f"  Front collisions detected: {front_detected}/{total_moments} ({front_detected/total_moments*100:.1f}%)")
        print(f"  Left collisions detected:  {left_detected}/{total_moments} ({left_detected/total_moments*100:.1f}%)")
        print(f"  Right collisions detected: {right_detected}/{total_moments} ({right_detected/total_moments*100:.1f}%)")
        
        any_collision = sum(1 for m in near_death_moments 
                           if m['front_coll'] >= 0.5 or m['left_coll'] >= 0.5 or m['right_coll'] >= 0.5)
        print(f"  ANY collision detected:    {any_collision}/{total_moments} ({any_collision/total_moments*100:.1f}%)")
    
    if death_positions:
        print(f"\nAverage snake length at death: {np.mean([d['snake_length'] for d in death_positions]):.1f}")
        print(f"Average map occupancy at death: {np.mean([d['map_occupancy'] for d in death_positions])*100:.1f}%")
    
    if food_acquisitions:
        print(f"Average food efficiency: {np.mean([f['efficiency'] for f in food_acquisitions]):.2f} steps/apple")
    
    print(f"\nAblation study:")
    print(f"  Baseline: {avg_rewards[0]:.2f} reward, {avg_scores[0]:.2f} score")
    print(f"  Without CNN: {performance_drops[1]:.1f}% drop")
    print(f"  Without Scalars: {performance_drops[2]:.1f}% drop")
    print(f"  CNN only: {performance_drops[3]:.1f}% drop")
    print(f"  Scalars only: {performance_drops[4]:.1f}% drop")