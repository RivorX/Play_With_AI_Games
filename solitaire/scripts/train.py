import os
import yaml
import torch
import shutil
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from model import make_env
from cnn import SolitaireFeaturesExtractor
from utils.callbacks import TrainProgressCallback, CustomEvalCallback

# Load config
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def mask_fn(env):
    return env.action_masks()

def make_masked_env(rank=0):
    def _init():
        env = make_env()()
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def train():
    # Config
    n_envs = config['training']['n_envs']
    total_timesteps = config['training']['total_timesteps']
    
    # Paths
    models_dir = os.path.join(base_dir, config['paths']['models_dir'])
    logs_dir = os.path.join(base_dir, config['paths']['logs_dir'])
    train_csv_path = os.path.join(base_dir, config['paths']['train_csv_path'])
    
    # Cleanup previous run
    if os.path.exists(train_csv_path):
        try:
            os.remove(train_csv_path)
            print("🧹 Usunięto stare statystyki.")
        except Exception as e:
            print(f"⚠️ Nie udało się usunąć statystyk: {e}")
        
    if os.path.exists(logs_dir):
        for item in os.listdir(logs_dir):
            item_path = os.path.join(logs_dir, item)
            if "ppo" in item.lower() and os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"⚠️ Nie udało się usunąć logu {item}: {e}")
        print("🧹 Usunięto stare logi TensorBoard.")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Environment
    # Use DummyVecEnv for simplicity and debugging, Subproc for speed
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    
    # Create vectorized environment with masking
    env = make_vec_env(make_masked_env(), n_envs=n_envs, vec_env_cls=vec_env_cls)
    
    # Eval Env
    eval_env = make_vec_env(make_masked_env(), n_envs=1, vec_env_cls=DummyVecEnv)
    
    # Model
    policy_kwargs = config['model']['policy_kwargs'].copy()
    policy_kwargs['features_extractor_class'] = SolitaireFeaturesExtractor
    
    model = MaskablePPO(
        config['model']['policy'],
        env,
        verbose=1,
        learning_rate=config['model']['learning_rate'],
        n_steps=config['model']['n_steps'],
        batch_size=config['model']['batch_size'],
        n_epochs=config['model']['n_epochs'],
        gamma=config['model']['gamma'],
        gae_lambda=config['model']['gae_lambda'],
        clip_range=config['model']['clip_range'],
        ent_coef=config['model']['ent_coef'],
        vf_coef=config['model']['vf_coef'],
        max_grad_norm=config['model']['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        tensorboard_log=logs_dir,
        device=config['model']['device']
    )
    
    # Callbacks
    # Usunięto CheckpointCallback, aby nie zapisywać modelu co X kroków
    
    plot_script_path = os.path.join(os.path.dirname(__file__), 'utils', 'plot_train_progress.py')
    
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=config['training']['eval_freq'],
        plot_script_path=plot_script_path
    )
    
    train_progress_callback = TrainProgressCallback(
        log_path=os.path.join(base_dir, config['paths']['train_csv_path'])
    )
    
    print("Starting training with MaskablePPO...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, train_progress_callback],
        tb_log_name="ppo"
    )
    
    model.save(os.path.join(base_dir, config['paths']['model_path']))
    print("Training finished!")

if __name__ == "__main__":
    train()
