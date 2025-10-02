import os
import pickle
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import cv2

# Import custom
from model import CnnLstmPolicy
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
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(config['environment']['frame_history'], 3, *config['environment']['screen_size']),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array(config['environment']['actions']['low']),
            high=np.array(config['environment']['actions']['high']),
            shape=config['environment']['actions']['shape'],
            dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = config['environment']['max_episode_steps']
        self.score = 0

    def reset(self):
        self.current_step = 0
        self.score = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Placeholder: losowa nagroda (dostosuj do realnego środowiska z detekcją OpenCV)
        reward = np.random.uniform(0, 1) * self.config['environment']['reward_scale']
        self.score += reward
        done = self.current_step >= self.max_steps or np.random.rand() > 0.99
        obs = np.random.rand(*self.observation_space.shape).astype(np.float32)
        return obs, reward, done, {'score': self.score}

# Prosta implementacja Behavioral Cloning
def simple_bc_train(model, dataset_path, config):
    # Załaduj dane z npz
    data = np.load(dataset_path)
    states = data['states']  # Shape: (N, frame_history, C, H, W)
    actions = data['actions']  # Shape: (N, action_dim)
    
    # Konwertuj na tensory
    states_tensor = torch.tensor(states, dtype=torch.float32).permute(0, 1, 4, 2, 3)  # Do formatu (batch, time, C, H, W)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    
    # Dataset i loader
    dataset = TensorDataset(states_tensor, actions_tensor)
    loader = DataLoader(dataset, batch_size=config['imitation']['batch_size'], shuffle=True)
    
    # Optymalizator i strata (MSE dla ciągłych akcji)
    optimizer = optim.Adam(model.policy.parameters(), lr=config['imitation']['learning_rate'])
    criterion = nn.MSELoss()
    
    model.policy.train()
    for epoch in range(config['imitation']['epochs']):
        total_loss = 0
        for batch_states, batch_actions in loader:
            pred_actions = model.policy(batch_states)  # Przewidywane akcje
            loss = criterion(pred_actions, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"BC Epoch {epoch+1}/{config['imitation']['epochs']}: Avg Loss {total_loss / len(loader):.4f}")
    
    print("Faza BC zakończona – model pre-trenowany na nagraniach.")

# Główny trening
def main():

    # Katalog agario
    agario_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    model_path = os.path.join(agario_dir, config['paths']['model_path'])
    state_path = os.path.join(agario_dir, config['paths']['state_path'])
    train_csv = os.path.join(agario_dir, config['paths']['train_csv_path'])
    
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
    policy_kwargs['features_extractor_class'] = ConvLSTM
    policy_kwargs['features_extractor_kwargs'] = {'conv_lstm_kwargs': config['model']['conv_lstm_kwargs']}
    
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
    
    if load_model and not (resp.lower() in ('n', 'no')):
        model = PPO.load(model_path, env=vec_env)
    
    # Faza Imitation: preprocess dataset
    recordings_dir = config['dataset']['recordings_dir']
    all_states, all_actions = [], []
    for session in os.listdir(recordings_dir):
        sess_dir = os.path.join(recordings_dir, session)
        if os.path.exists(os.path.join(sess_dir, 'actions.json')):
            frames = [cv2.imread(os.path.join(sess_dir, f)) for f in sorted(os.listdir(sess_dir)) if f.endswith('.png')]
            frames = np.array([cv2.resize(f, tuple(config['environment']['screen_size'])) / 255.0 for f in frames])
            with open(os.path.join(sess_dir, 'actions.json'), 'r') as f:
                acts = json.load(f)
            acts = np.array([[a['mouse_delta'][0], a['mouse_delta'][1], int(a['keys']['split'] or a['keys']['eject'])] for a in acts])
            all_states.append(frames[:len(acts)])  # Dopasuj długości
            all_actions.append(acts)
    
    # Trening BC
    if all_states:
        dataset = np.savez(config['paths']['dataset_path'], states=np.concatenate(all_states), actions=np.concatenate(all_actions))
        simple_bc_train(model, config['paths']['dataset_path'], config)
    
    # Trening PPO
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