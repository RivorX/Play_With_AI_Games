import os
import pickle
import yaml
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import cv2
import psutil  # Do monitorowania pamięci RAM
import gc  # Do garbage collection (opcjonalne czyszczenie mem)

# Import custom
from model import CnnLstmPolicy, CnnLstmExtractor
from conv_lstm import ConvLSTM

# Wczytaj konfigurację

# Wczytaj konfigurację względnie do lokalizacji skryptu
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Custom Env (uproszczony placeholder – do realnego użycia potrzebny wrapper Selenium)
class AgarIoEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # screen_size = [W, H] = [256, 192]
        # observation_space = (T, C, H, W) = (4, 3, 192, 256)
        screen_w, screen_h = config['environment']['screen_size']
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(config['environment']['frame_history'], 3, screen_h, screen_w),
            dtype=np.float32
        )
        # Zamień shape na tuple int jeśli jest stringiem (np. '(3,)')
        shape_val = config['environment']['actions']['shape']
        if isinstance(shape_val, str):
            # Usuwa nawiasy i przecinki, dzieli po ',' i konwertuje na int
            shape_val = tuple(int(s) for s in shape_val.replace('(', '').replace(')', '').replace(',', ' ').split() if s.isdigit())
        elif isinstance(shape_val, list):
            shape_val = tuple(shape_val)
        self.action_space = spaces.Box(
            low=np.array(config['environment']['actions']['low']),
            high=np.array(config['environment']['actions']['high']),
            shape=shape_val,
            dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = config['environment']['max_episode_steps']
        self.score = 0

    def reset(self, seed=None, options=None):
        # Poprawka: Dodano seed i options zgodnie z Gymnasium API
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.score = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Placeholder: losowa nagroda (dostosuj do realnego środowiska z detekcją OpenCV)
        reward = np.random.uniform(0, 1) * self.config['environment']['reward_scale']
        self.score += reward
        # Poprawka: Gymnasium używa terminated i truncated zamiast done
        terminated = self.current_step >= self.max_steps or np.random.rand() > 0.99
        truncated = False
        obs = np.random.rand(*self.observation_space.shape).astype(np.float32)
        return obs, reward, terminated, truncated, {'score': self.score}

# Automatyczny split sesji na train/val
def split_sessions_train_val(recordings_dir, val_ratio=0.2, seed=42):
    """
    Automatycznie dzieli sesje z recordings/ na train/ i val/.
    Zwraca (train_sessions, val_sessions) - listy ścieżek do folderów sesji.
    """
    import random
    random.seed(seed)
    
    # Znajdź wszystkie sesje w głównym folderze recordings
    all_sessions = []
    train_dir = os.path.join(recordings_dir, 'train')
    val_dir = os.path.join(recordings_dir, 'val')
    
    # Zbierz sesje z głównego folderu (nie w train/val)
    for item in os.listdir(recordings_dir):
        item_path = os.path.join(recordings_dir, item)
        if os.path.isdir(item_path) and item not in ['train', 'val']:
            all_sessions.append(item)
    
    if not all_sessions:
        # Sprawdź czy już są w train/val
        train_sessions = []
        val_sessions = []
        if os.path.exists(train_dir):
            train_sessions = [os.path.join(train_dir, s) for s in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, s))]
        if os.path.exists(val_dir):
            val_sessions = [os.path.join(val_dir, s) for s in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, s))]
        
        if train_sessions or val_sessions:
            print(f"Sesje już podzielone: {len(train_sessions)} train, {len(val_sessions)} val")
            return train_sessions, val_sessions
        else:
            print("Brak sesji do podziału")
            return [], []
    
    # Podziel sesje
    random.shuffle(all_sessions)
    val_count = max(1, int(len(all_sessions) * val_ratio))
    val_sessions_names = all_sessions[:val_count]
    train_sessions_names = all_sessions[val_count:]
    
    print(f"\n=== Automatyczny split sesji ===")
    print(f"Wszystkie sesje: {len(all_sessions)}")
    print(f"Train: {len(train_sessions_names)} ({100*(1-val_ratio):.0f}%)")
    print(f"Val: {len(val_sessions_names)} ({100*val_ratio:.0f}%)")
    
    # Utwórz foldery train/val
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Przenieś sesje
    import shutil
    train_sessions = []
    val_sessions = []
    
    for session in train_sessions_names:
        src = os.path.join(recordings_dir, session)
        dst = os.path.join(train_dir, session)
        if not os.path.exists(dst):
            shutil.move(src, dst)
        train_sessions.append(dst)
        print(f"  [TRAIN] {session}")
    
    for session in val_sessions_names:
        src = os.path.join(recordings_dir, session)
        dst = os.path.join(val_dir, session)
        if not os.path.exists(dst):
            shutil.move(src, dst)
        val_sessions.append(dst)
        print(f"  [VAL] {session}")
    
    print(f"Split zakończony!\n")
    return train_sessions, val_sessions

# Wczytaj sesje i stwórz dataset
def load_sessions_to_dataset(session_dirs, config, dataset_name="dataset"):
    """Wczytuje sesje i tworzy dataset (states, rewards)"""
    frame_history = config['environment']['frame_history']
    all_states = []
    all_rewards = []
    
    for sess_dir in session_dirs:
        session_name = os.path.basename(sess_dir)
        print(f"Wczytywanie sesji: {session_name}")
        
        frames_files = [f for f in sorted(os.listdir(sess_dir)) if f.endswith('.png')]
        
        if len(frames_files) == 0:
            print(f"  Pomijanie: brak plików PNG")
            continue
        
        print(f"  Liczba klatek: {len(frames_files)}")
        
        # Wczytaj i przetworz klatki
        frames = []
        for f in frames_files:
            img = cv2.imread(os.path.join(sess_dir, f))
            if img is None:
                continue
            # cv2.resize przyjmuje (width, height)
            screen_w, screen_h = config['environment']['screen_size']
            img_resized = cv2.resize(img, (screen_w, screen_h))  # (256, 192)
            frames.append(img_resized / 255.0)
        
        if len(frames) < frame_history:
            print(f"  Pomijanie: za mało klatek ({len(frames)} < {frame_history})")
            continue
        
        frames = np.array(frames)
        
        # Wczytaj OCR results
        ocr_path = os.path.join(sess_dir, 'ocr_results.txt')
        scores = []
        if os.path.exists(ocr_path):
            with open(ocr_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(' - ')
                    if len(parts) == 2:
                        try:
                            scores.append(int(parts[1]))
                        except ValueError:
                            scores.append(0)
        else:
            scores = [0] * len(frames)
        
        # Upewnij się że len(scores) == len(frames)
        if len(scores) < len(frames):
            scores.extend([0] * (len(frames) - len(scores)))
        elif len(scores) > len(frames):
            scores = scores[:len(frames)]
        
        # Sliding window
        num_windows = len(frames) - frame_history + 1
        for i in range(num_windows):
            state_seq = frames[i:i + frame_history]
            state_seq = np.transpose(state_seq, (0, 3, 1, 2))
            all_states.append(state_seq)
            
            start_score = scores[i]
            end_score = scores[i + frame_history - 1]
            score_delta = end_score - start_score
            reward = score_delta / max(1, start_score) if start_score > 0 else score_delta
            all_rewards.append(reward)
        
        print(f"  Dodano {num_windows} trajectories")
    
    if all_states:
        states = np.array(all_states)
        rewards = np.array(all_rewards)
        print(f"\n{dataset_name}: {len(states)} trajectories")
        print(f"  Rewards: mean={rewards.mean():.3f}, std={rewards.std():.3f}, range=[{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"  Positives: {(rewards > 0).sum()} ({100*(rewards > 0).mean():.1f}%)")
        return states, rewards
    else:
        return None, None

# Reward-weighted pre-training (Imitation z OCR rewards)
def reward_weighted_pretraining(model, train_dataset_path, val_dataset_path, config, device='cuda'):
    """
    Pre-trenuj model używając trajectory z wysokim reward (z OCR score).
    Uczy value function rozpoznawać dobre stany.
    Waliduje co eval_interval epok.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd
    
    # Wczytaj train data
    train_data = np.load(train_dataset_path)
    train_states = train_data['states']
    train_rewards = train_data['rewards']
    
    print(f"\n=== Reward-Weighted Pre-training ===")
    print(f"Train dataset: {len(train_states)} trajectories")
    print(f"Train reward range: [{train_rewards.min():.3f}, {train_rewards.max():.3f}]")
    
    # Filtruj tylko pozytywne rewards
    positive_mask = train_rewards > 0
    if positive_mask.sum() == 0:
        print("Brak trajectories z pozytywnym reward - pomijam pre-training")
        return
    
    good_states = train_states[positive_mask]
    good_rewards = train_rewards[positive_mask]
    print(f"Trajectories z pozytywnym reward: {len(good_states)} ({100*len(good_states)/len(train_states):.1f}%)")
    
    # Validation data (opcjonalne)
    has_val = val_dataset_path and os.path.exists(val_dataset_path)
    if has_val:
        val_data = np.load(val_dataset_path)
        val_states = val_data['states']
        val_rewards = val_data['rewards']
        print(f"Val dataset: {len(val_states)} trajectories")
    
    # Przygotuj tensory
    train_states_tensor = torch.tensor(good_states, dtype=torch.float32, device=device)
    train_rewards_tensor = torch.tensor(good_rewards, dtype=torch.float32, device=device)
    
    # Normalizuj rewards jako wagi
    weights = (train_rewards_tensor - train_rewards_tensor.min()) / (train_rewards_tensor.max() - train_rewards_tensor.min() + 1e-8)
    
    dataset = TensorDataset(train_states_tensor, train_rewards_tensor, weights)
    loader = DataLoader(dataset, batch_size=config['imitation']['batch_size'], shuffle=True)
    
    # Trenuj value network
    params_to_train = (
        list(model.policy.vf_features_extractor.parameters()) +
        list(model.policy.mlp_extractor.value_net.parameters()) +
        list(model.policy.value_net.parameters())
    )
    optimizer = torch.optim.Adam(params_to_train, lr=config['imitation']['learning_rate'])
    
    # Logi
    log_path = os.path.join(os.path.dirname(train_dataset_path), '..', config['paths']['imitation_log_path'])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logs = []
    
    model.policy.train()
    eval_interval = config['imitation'].get('eval_interval', 10)
    
    for epoch in range(config['imitation']['epochs']):
        total_loss = 0
        for batch_states, batch_rewards, batch_weights in loader:
            features = model.policy.extract_features(batch_states, model.policy.vf_features_extractor)
            latent_vf = model.policy.mlp_extractor.forward_critic(features)
            pred_values = model.policy.value_net(latent_vf).squeeze(-1)
            
            loss = (batch_weights * (pred_values - batch_rewards) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        # Ewaluacja na val
        val_loss = None
        val_mse = None
        if has_val and (epoch + 1) % eval_interval == 0:
            model.policy.eval()
            with torch.no_grad():
                val_states_tensor = torch.tensor(val_states, dtype=torch.float32, device=device)
                val_rewards_tensor = torch.tensor(val_rewards, dtype=torch.float32, device=device)
                
                features = model.policy.extract_features(val_states_tensor, model.policy.vf_features_extractor)
                latent_vf = model.policy.mlp_extractor.forward_critic(features)
                pred_values = model.policy.value_net(latent_vf).squeeze(-1)
                
                val_mse = ((pred_values - val_rewards_tensor) ** 2).mean().item()
                val_loss = val_mse
            model.policy.train()
        
        # Log
        log_entry = {'epoch': epoch + 1, 'train_loss': avg_loss}
        if val_loss is not None:
            log_entry['val_loss'] = val_loss
            log_entry['val_mse'] = val_mse
        logs.append(log_entry)
        
        # Print
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{config['imitation']['epochs']}: Train Loss {avg_loss:.4f}"
            if val_loss is not None:
                msg += f", Val MSE {val_mse:.4f}"
            print(msg)
    
    # Zapisz logi
    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False)
    print(f"Logi zapisane w: {log_path}")
    print("Pre-training zakończony – model nauczył się rozpoznawać dobre stany\n")

# Główny trening
def main():

    # Katalog agario
    agario_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    model_path = os.path.join(agario_dir, config['paths']['model_path'])
    state_path = os.path.join(agario_dir, config['paths']['state_path'])
    train_csv = os.path.join(agario_dir, config['paths']['train_csv_path'])
    dataset_path = os.path.join(agario_dir, config['paths']['dataset_path'])
    
    # Upewnij się, że katalogi istnieją
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    load_model = os.path.exists(model_path)
    total_timesteps = config['training']['total_timesteps']
    
    if load_model:
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                total_timesteps = state['total_timesteps']
            print(f"Wznowienie treningu od {total_timesteps} kroków.")
        except Exception as e:
            print(f"Błąd ładowania stanu: {e}. Zaczynam od zera.")
        
        try:
            resp = input(f"Znaleziono istniejący model pod {model_path}. Czy kontynuować trening? [[Y]/n]: ").strip()
        except:
            resp = ''
        
        if resp.lower() in ('n', 'no'):
            print("Użytkownik wybrał rozpoczęcie treningu od nowa.")
            load_model = False
            total_timesteps = 0
            use_config_hyperparams = True
            try:
                if os.path.exists(train_csv):
                    os.remove(train_csv)
                    print(f"Usunięto {train_csv}")
            except Exception as e:
                print(f"Nie udało się usunąć: {e}")
        else:
            try:
                resp2 = input("Użyć hyperparametrów z configu zamiast z modelu? [[Y]/n]: ").strip()
            except:
                resp2 = ''
            use_config_hyperparams = False if resp2.lower() in ('n', 'no') else True
    else:
        use_config_hyperparams = True
    
    # Stwórz envs
    env_fn = lambda: AgarIoEnv(config)
    vec_env = DummyVecEnv([env_fn for _ in range(config['training']['n_envs'])])
    
    # Model
    policy_kwargs = config['model']['policy_kwargs']
    # Poprawka: Ustaw class na CnnLstmExtractor (nie ConvLSTM)
    policy_kwargs['features_extractor_class'] = CnnLstmExtractor
    # Poprawka: Przekazuj bezpośrednio conv_lstm_kwargs (bez zagnieżdżonego klucza)
    policy_kwargs['features_extractor_kwargs'] = config['model']['conv_lstm_kwargs']
    # Usuń lub nadpisz net_arch z configu (już jest)
    policy_kwargs['net_arch'] = config['model']['policy_kwargs']['net_arch']
    
    print("Inicjalizacja modelu PPO...")
    mem_before_model = psutil.Process().memory_info().rss / 1e9
    print(f"Zużycie RAM przed inicjalizacją modelu: {mem_before_model:.2f} GB")
    
    # Opcjonalne: Czyszczenie przed PPO
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA dostępna - wyczyszczono cache")
    
    # Hack: SB3 print "Using cuda device" jest w środku PPO.__init__(), więc nie da się złapać printem
    # Ale sprawdzimy mem po (spike będzie widoczny)
    model = PPO(
        CnnLstmPolicy,
        vec_env,
        verbose=1,
        device=config['model']['device'],
        learning_rate=config['model']['learning_rate'],
        n_steps=config['model']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['model']['n_epochs'],
        gamma=config['model']['gamma'],
        gae_lambda=config['model']['gae_lambda'],
        clip_range=config['model']['clip_range'],
        ent_coef=config['model']['ent_coef'],
        vf_coef=config['model']['vf_coef'],
        policy_kwargs=policy_kwargs
    )
    
    mem_after_model = psutil.Process().memory_info().rss / 1e9
    print(f"Zużycie RAM po inicjalizacji modelu (po 'Using cuda device'): {mem_after_model:.2f} GB")
    print(f"Skok pamięci w PPO: {mem_after_model - mem_before_model:.2f} GB")
    print(f"Rozmiar bufora rollout (szacunkowy): {config['model']['n_steps'] * config['training']['n_envs'] * 4 * 3 * config['environment']['screen_size'][0] * config['environment']['screen_size'][1] * config['environment']['frame_history'] / 1e9:.2f} GB (tylko obs)")
    
    if load_model and not (resp.lower() in ('n', 'no')):
        model = PPO.load(model_path, env=vec_env)
    
    # === IMITATION LEARNING (jeśli włączone) ===
    use_imitation = config['imitation'].get('enabled', True)
    imitation_model_path = os.path.join(agario_dir, config['paths']['imitation_model_path'])
    
    if use_imitation:
        print("\n" + "="*50)
        print("FAZA 1: IMITATION LEARNING (Reward-Weighted)")
        print("="*50)
        
        # Sprawdź czy model po imitation już istnieje
        if os.path.exists(imitation_model_path):
            try:
                resp = input(f"Znaleziono model po Imitation Learning:\n  {imitation_model_path}\nCzy pominąć pre-training i załadować ten model? [Y/n]: ").strip()
            except:
                resp = ''
            
            if resp.lower() not in ('n', 'no'):
                print(f"Ładowanie modelu po imitation: {imitation_model_path}")
                model = PPO.load(imitation_model_path, env=vec_env)
                print("✅ Model załadowany - pomijam pre-training\n")
                use_imitation = False  # Pomiń pre-training
            else:
                print("Użytkownik wybrał ponowny pre-training\n")
        
        if use_imitation:  # Tylko jeśli nie załadowano modelu
            recordings_dir = os.path.join(agario_dir, config['dataset']['recordings_dir'])
            train_dataset_path = os.path.join(agario_dir, config['paths']['train_dataset_path'])
            val_dataset_path = os.path.join(agario_dir, config['paths']['val_dataset_path'])
            
            # Automatyczny split train/val
            if config['dataset'].get('train_val_split', True):
                train_sessions, val_sessions = split_sessions_train_val(
                    recordings_dir, 
                    val_ratio=config['dataset'].get('val_split_ratio', 0.2)
                )
            else:
                # Ręczny podział (train/ i val/ foldery)
                train_dir = os.path.join(recordings_dir, 'train')
                val_dir = os.path.join(recordings_dir, 'val')
                train_sessions = [os.path.join(train_dir, s) for s in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, s))] if os.path.exists(train_dir) else []
                val_sessions = [os.path.join(val_dir, s) for s in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, s))] if os.path.exists(val_dir) else []
            
            # Wczytaj dane
            if train_sessions:
                print(f"\n--- Wczytywanie TRAIN dataset ---")
                train_states, train_rewards = load_sessions_to_dataset(train_sessions, config, "Train")
                if train_states is not None:
                    np.savez(train_dataset_path, states=train_states, rewards=train_rewards)
                    print(f"Train dataset zapisany: {train_dataset_path}")
            
            if val_sessions:
                print(f"\n--- Wczytywanie VAL dataset ---")
                val_states, val_rewards = load_sessions_to_dataset(val_sessions, config, "Val")
                if val_states is not None:
                    np.savez(val_dataset_path, states=val_states, rewards=val_rewards)
                    print(f"Val dataset zapisany: {val_dataset_path}")
            
            # Pre-training
            if train_sessions and os.path.exists(train_dataset_path):
                mem_before_pretrain = psutil.Process().memory_info().rss / 1e9
                print(f"\nZużycie RAM przed pre-training: {mem_before_pretrain:.2f} GB")
                
                reward_weighted_pretraining(
                    model, 
                    train_dataset_path, 
                    val_dataset_path if val_sessions else None,
                    config, 
                    device=config['model']['device']
                )
                
                mem_after_pretrain = psutil.Process().memory_info().rss / 1e9
                print(f"Zużycie RAM po pre-training: {mem_after_pretrain:.2f} GB")
                
                # Zapisz model po Imitation Learning
                model.save(imitation_model_path)
                print(f"\n✅ Model po Imitation Learning zapisany: {imitation_model_path}")
                print(f"   (Możesz go załadować później bez potrzeby ponownego pre-trainingu)\n")
            else:
                print("Brak train sessions - pomijam imitation learning")
    else:
        print("\n⏩ Imitation Learning wyłączone (config: imitation.enabled=false)")
    
    # === REINFORCEMENT LEARNING (jeśli włączone) ===
    use_rl = config['training'].get('use_rl', True)
    
    if not use_rl:
        print("\n⏩ Reinforcement Learning wyłączone (config: training.use_rl=false)")
        print("Trening zakończony - model wytrenowany tylko na imitation learning")
        return
    
    
    # === REINFORCEMENT LEARNING PPO ===
    print("\n" + "="*50)
    print("FAZA 2: REINFORCEMENT LEARNING (PPO)")
    print("="*50)
    print("Rozpoczynanie treningu PPO...")
    mem_before_ppo = psutil.Process().memory_info().rss / 1e9
    print(f"Zużycie RAM przed treningiem PPO: {mem_before_ppo:.2f} GB\n")
    
    best_mean_reward = -np.inf
    eval_freq = config['training']['eval_freq']
    no_improvement = 0
    min_evals = config['training']['min_evals']
    max_no_improvement = config['training']['max_no_improvement_evals']
    rewards_history = []
    
    for i in range(total_timesteps // config['model']['n_steps']):
        model.learn(total_timesteps=config['model']['n_steps'], reset_num_timesteps=False)
        
        if model.num_timesteps % eval_freq == 0:
            mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)
            print(f"Step {model.num_timesteps}: Mean reward {mean_reward}")
            rewards_history.append(mean_reward)
            
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                model.save(model_path)
                no_improvement = 0
                state = {'total_timesteps': model.num_timesteps}
                with open(state_path, 'wb') as f:
                    pickle.dump(state, f)
            else:
                no_improvement += 1
            
            if no_improvement >= max_no_improvement and i >= min_evals:
                print("Early stopping.")
                break
    
    # Zapisz wykres
    plt.plot(rewards_history)
    plt.xlabel("Evaluation Step")
    plt.ylabel("Mean Reward")
    plt.title("Training Progress")
    plt.savefig(config['paths']['plot_path'])
    plt.close()
    vec_env.close()

if __name__ == "__main__":
    main()