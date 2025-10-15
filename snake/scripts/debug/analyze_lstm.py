import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def analyze_lstm_memory(model, env, output_dir, action_names, config):
    """Analiza konsystencji LSTM (pamięć)"""
    print("\n=== Analiza konsystencji LSTM (pamięć) ===")
    
    policy = model.policy
    
    # Zbieramy dane z pełnego epizodu
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    episode_lstm_data = []
    step_count = 0
    max_steps = config['model']['n_steps'] * 5
    
    while step_count < max_steps:
        # Predict
        with torch.no_grad():
            action, new_lstm_states = model.predict(
                obs, 
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
            
            # Zapisz dane LSTM
            if lstm_states is not None:
                hidden_state = lstm_states[0]  # Shape: [n_layers, batch, hidden_size]
                cell_state = lstm_states[1]
                
                # Mean magnitude per layer
                hidden_mean = np.abs(hidden_state).mean(axis=(1, 2))  # [n_layers]
                cell_mean = np.abs(cell_state).mean(axis=(1, 2))
                
                # Change from previous step
                if len(episode_lstm_data) > 0:
                    prev_hidden = episode_lstm_data[-1]['hidden_state']
                    hidden_change = np.abs(hidden_state - prev_hidden).mean()
                else:
                    hidden_change = 0.0
                
                episode_lstm_data.append({
                    'step': step_count,
                    'action': action_names[action_idx],
                    'hidden_state': hidden_state.copy(),
                    'cell_state': cell_state.copy(),
                    'hidden_mean_layer0': hidden_mean[0] if len(hidden_mean) > 0 else 0,
                    'cell_mean_layer0': cell_mean[0] if len(cell_mean) > 0 else 0,
                    'hidden_change': hidden_change
                })
            
            lstm_states = new_lstm_states
        
        # Step
        obs, reward, done, truncated, info = env.step(action_idx)
        episode_starts = np.array([done or truncated], dtype=bool)
        step_count += 1
        
        if done or truncated:
            obs, _ = env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
    
    # Wizualizacja LSTM memory evolution
    if len(episode_lstm_data) > 0:
        steps = [d['step'] for d in episode_lstm_data]
        hidden_means = [d['hidden_mean_layer0'] for d in episode_lstm_data]
        cell_means = [d['cell_mean_layer0'] for d in episode_lstm_data]
        hidden_changes = [d['hidden_change'] for d in episode_lstm_data]
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Hidden state magnitude over time
        axes[0].plot(steps, hidden_means, label='Hidden State (Layer 0)', color='#9b59b6', linewidth=2)
        axes[0].set_xlabel('Krok')
        axes[0].set_ylabel('Średnia magnitude')
        axes[0].set_title('Ewolucja Hidden State LSTM')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Cell state magnitude over time
        axes[1].plot(steps, cell_means, label='Cell State (Layer 0)', color='#8e44ad', linewidth=2)
        axes[1].set_xlabel('Krok')
        axes[1].set_ylabel('Średnia magnitude')
        axes[1].set_title('Ewolucja Cell State LSTM')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Plot 3: Hidden state change (memory update rate)
        axes[2].plot(steps[1:], hidden_changes[1:], label='Zmiana Hidden State', color='#e74c3c', linewidth=2)
        axes[2].set_xlabel('Krok')
        axes[2].set_ylabel('Wielkość zmiany')
        axes[2].set_title('Tempo aktualizacji pamięci LSTM')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        lstm_evolution_path = os.path.join(output_dir, 'lstm_memory_evolution.png')
        plt.savefig(lstm_evolution_path, dpi=150)
        plt.close()
        print(f'✓ LSTM memory evolution zapisana: {lstm_evolution_path}')
        
        # Heatmapa hidden state w czasie
        hidden_states_matrix = np.array([d['hidden_state'][0, 0, :] for d in episode_lstm_data])
        
        plt.figure(figsize=(16, 8))
        plt.imshow(hidden_states_matrix.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Wartość aktywacji')
        plt.xlabel('Krok czasowy')
        plt.ylabel('Neuron LSTM')
        plt.title('Aktywacja wszystkich neuronów LSTM w czasie')
        plt.tight_layout()
        lstm_heatmap_path = os.path.join(output_dir, 'lstm_neurons_heatmap.png')
        plt.savefig(lstm_heatmap_path, dpi=150)
        plt.close()
        print(f'✓ LSTM neurons heatmap zapisana: {lstm_heatmap_path}')
        
        # Zapisz dane LSTM do CSV
        lstm_csv_path = os.path.join(output_dir, 'lstm_memory_data.csv')
        with open(lstm_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'action', 'hidden_mean_layer0', 'cell_mean_layer0', 'hidden_change'])
            for d in episode_lstm_data:
                writer.writerow([
                    d['step'],
                    d['action'],
                    d['hidden_mean_layer0'],
                    d['cell_mean_layer0'],
                    d['hidden_change']
                ])
        print(f'✓ LSTM memory data zapisana: {lstm_csv_path}')


def analyze_confusion_matrix(model, env, output_dir, action_names, num_episodes=20):
    """Analiza Confusion Matrix - porównanie z heurystyką"""
    print("\n=== Zbieranie danych dla Confusion Matrix ===")
    
    confusion_matrix = np.zeros((3, 3))  # [expected_action, actual_action]
    action_history = []
    
    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        step = 0
        
        while not done and step < 100:
            # Predict
            with torch.no_grad():
                action, lstm_states = model.predict(
                    obs, 
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
            
            # Heurystyka: "oczekiwana akcja" na podstawie prostej logiki
            dx = obs['dx_head'][0]
            dy = obs['dy_head'][0]
            front_coll = obs['front_coll'][0]
            left_coll = obs['left_coll'][0]
            right_coll = obs['right_coll'][0]
            
            expected_action = 1  # default: prosto
            
            if front_coll > 0.5:  # kolizja z przodu
                if left_coll < 0.5:
                    expected_action = 0  # lewo
                elif right_coll < 0.5:
                    expected_action = 2  # prawo
            else:
                # Skręcaj w stronę jedzenia
                food_angle = np.arctan2(dy, dx) * 180 / np.pi
                if abs(food_angle) < 30:
                    expected_action = 1  # prosto
                elif food_angle < -30:
                    expected_action = 0  # lewo
                elif food_angle > 30:
                    expected_action = 2  # prawo
            
            confusion_matrix[expected_action, action_idx] += 1
            action_history.append({
                'episode': episode_idx,
                'step': step,
                'expected': action_names[expected_action],
                'actual': action_names[action_idx],
                'match': expected_action == action_idx
            })
            
            # Step
            obs, reward, done, truncated, info = env.step(action_idx)
            episode_starts = np.array([done or truncated], dtype=bool)
            done = done or truncated
            step += 1
        
        if (episode_idx + 1) % 5 == 0:
            print(f"  Przetworzono {episode_idx + 1}/{num_episodes} epizodów")
    
    # Wizualizacja Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')
    
    # Dodaj wartości do komórek
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, int(confusion_matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(action_names)
    ax.set_yticklabels(action_names)
    ax.set_xlabel('Akcja Modelu (Actual)', fontsize=12)
    ax.set_ylabel('Oczekiwana Akcja (Expected)', fontsize=12)
    ax.set_title('Confusion Matrix - Porównanie z Heurystyką', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Liczba przypadków')
    plt.tight_layout()
    confusion_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_path, dpi=150)
    plt.close()
    print(f'✓ Confusion matrix zapisana: {confusion_path}')
    
    # Accuracy
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Confusion Matrix Stats:")
    print(f"   Zgodność z heurystyką: {accuracy*100:.1f}%")
    print(f"   Całkowita liczba akcji: {int(total)}")
    print(f"   Zgodnych akcji: {int(correct)}")
    
    # Zapisz confusion matrix do CSV
    confusion_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    confusion_df = np.vstack([
        [''] + action_names,
        *[[action_names[i]] + list(confusion_matrix[i, :]) for i in range(3)]
    ])
    np.savetxt(confusion_csv_path, confusion_df, delimiter=',', fmt='%s')
    print(f'✓ Confusion matrix CSV zapisana: {confusion_csv_path}')


def analyze_uncertainty(model, env, output_dir, action_names, num_episodes=10):
    """Rozszerzona analiza uncertainty"""
    print("\n=== Rozszerzona analiza uncertainty ===")
    
    policy = model.policy
    features_extractor = policy.features_extractor
    
    extended_uncertainty = []
    
    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        step = 0
        
        while not done and step < 50:
            with torch.no_grad():
                # Przygotuj obs_tensor
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
                
                # Get features
                features = features_extractor(obs_tensor)
                
                # LSTM
                lstm_states_tensor = (
                    torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device) if lstm_states is not None else None,
                    torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device) if lstm_states is not None else None
                )
                
                if lstm_states_tensor[0] is not None:
                    features_seq = features.unsqueeze(1)
                    lstm_output, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_tensor)
                    latent_pi = lstm_output.squeeze(1)
                else:
                    # Initial state
                    batch_size = 1
                    n_layers = policy.lstm_actor.num_layers
                    hidden_size = policy.lstm_actor.hidden_size
                    device = policy.device
                    lstm_states_init = (
                        torch.zeros(n_layers, batch_size, hidden_size, device=device),
                        torch.zeros(n_layers, batch_size, hidden_size, device=device)
                    )
                    features_seq = features.unsqueeze(1)
                    lstm_output, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_init)
                    latent_pi = lstm_output.squeeze(1)
                
                # MLP and action
                latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
                logits = policy.action_net(latent_pi_mlp)
                action_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                action_idx = np.argmax(action_probs)
                
                # Uncertainty metrics
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
                max_prob = np.max(action_probs)
                margin = np.partition(action_probs, -2)[-1] - np.partition(action_probs, -2)[-2]
                
                lstm_states = (new_lstm_states[0].cpu().numpy(), new_lstm_states[1].cpu().numpy())
            
            # Step
            obs, reward, done, truncated, info = env.step(action_idx)
            episode_starts = np.array([done or truncated], dtype=bool)
            
            extended_uncertainty.append({
                'episode': episode_idx,
                'step': step,
                'entropy': entropy,
                'max_prob': max_prob,
                'margin': margin,
                'action': action_names[action_idx],
                'reward': reward
            })
            
            done = done or truncated
            step += 1
    
    # Wizualizacja Uncertainty
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Histogram entropii
    entropies = [d['entropy'] for d in extended_uncertainty]
    axes[0, 0].hist(entropies, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', linewidth=2, label=f'Średnia: {np.mean(entropies):.3f}')
    axes[0, 0].set_xlabel('Entropia')
    axes[0, 0].set_ylabel('Częstość')
    axes[0, 0].set_title('Rozkład Entropii Decyzji')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Max probability distribution
    max_probs = [d['max_prob'] for d in extended_uncertainty]
    axes[0, 1].hist(max_probs, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, label=f'Średnia: {np.mean(max_probs):.3f}')
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Częstość')
    axes[0, 1].set_title('Rozkład Pewności Modelu')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Entropy vs Reward (per episode)
    episode_stats = []
    current_episode_entropies = []
    current_episode_rewards = []
    current_episode = extended_uncertainty[0]['episode'] if extended_uncertainty else 0
    for d in extended_uncertainty:
        if d['episode'] != current_episode:
            if current_episode_entropies:
                mean_entropy = np.mean(current_episode_entropies)
                sum_reward = np.sum(current_episode_rewards)
                episode_stats.append({'mean_entropy': mean_entropy, 'sum_reward': sum_reward})
            current_episode_entropies = []
            current_episode_rewards = []
            current_episode = d['episode']
        current_episode_entropies.append(d['entropy'])
        current_episode_rewards.append(d['reward'])
    
    if current_episode_entropies:
        mean_entropy = np.mean(current_episode_entropies)
        sum_reward = np.sum(current_episode_rewards)
        episode_stats.append({'mean_entropy': mean_entropy, 'sum_reward': sum_reward})
    
    axes[1, 0].scatter([e['mean_entropy'] for e in episode_stats],
                       [e['sum_reward'] for e in episode_stats],
                       alpha=0.7, c='#e74c3c', s=40)
    axes[1, 0].set_xlabel('Średnia entropia epizodu')
    axes[1, 0].set_ylabel('Suma rewardów epizodu')
    axes[1, 0].set_title('Entropia vs Reward (epizod)')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Certainty categories
    high_cert = [d for d in extended_uncertainty if d['max_prob'] > 0.8]
    medium_cert = [d for d in extended_uncertainty if 0.5 <= d['max_prob'] <= 0.8]
    low_cert = [d for d in extended_uncertainty if d['max_prob'] < 0.5]
    
    categories = ['High\n(>0.8)', 'Medium\n(0.5-0.8)', 'Low\n(<0.5)']
    counts = [len(high_cert), len(medium_cert), len(low_cert)]
    colors_cert = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = axes[1, 1].bar(categories, counts, color=colors_cert, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Liczba decyzji')
    axes[1, 1].set_title('Kategorie Pewności Modelu')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/len(extended_uncertainty)*100:.1f}%)',
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    uncertainty_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(uncertainty_path, dpi=150)
    plt.close()
    print(f'✓ Uncertainty analysis zapisana: {uncertainty_path}')
    
    # Zapisz uncertainty data do CSV
    uncertainty_csv_path = os.path.join(output_dir, 'uncertainty_data.csv')
    with open(uncertainty_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'step', 'entropy', 'max_prob', 'margin', 'action', 'reward'])
        for d in extended_uncertainty:
            writer.writerow([
                d['episode'],
                d['step'],
                d['entropy'],
                d['max_prob'],
                d['margin'],
                d['action'],
                d['reward']
            ])
    print(f'✓ Uncertainty data zapisana: {uncertainty_csv_path}')
    
    print(f"\n📊 Uncertainty Stats:")
    print(f"   Średnia entropia: {np.mean(entropies):.3f}")
    print(f"   Średnia pewność (max_prob): {np.mean(max_probs):.3f}")
    print(f"   Decyzje wysokiej pewności: {len(high_cert)} ({len(high_cert)/len(extended_uncertainty)*100:.1f}%)")
    print(f"   Decyzje średniej pewności: {len(medium_cert)} ({len(medium_cert)/len(extended_uncertainty)*100:.1f}%)")
    print(f"   Decyzje niskiej pewności: {len(low_cert)} ({len(low_cert)/len(extended_uncertainty)*100:.1f}%)")