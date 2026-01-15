import os
import yaml
import torch
import shutil
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

from model import make_env
from cnn import SolitaireFeaturesExtractor
from utils.callbacks import (
    TrainProgressCallback, 
    CustomEvalCallback, 
    LossRecorderCallback,
    EntropySchedulerCallback,
    WinTrackerCallback,
    PeriodicSaveCallback
)

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
    eval_freq = config['training']['eval_freq']
    plot_interval = config['training']['plot_interval']
    eval_n_envs = config['training'].get('eval_n_envs', 4)
    eval_n_repeats = config['training'].get('eval_n_repeats', 3)
    
    # Paths
    models_dir = os.path.join(base_dir, config['paths']['models_dir'])
    logs_dir = os.path.join(base_dir, config['paths']['logs_dir'])
    train_csv_path = os.path.join(base_dir, config['paths']['train_csv_path'])
    model_path = os.path.join(base_dir, config['paths']['model_path'])
    
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
    # Use SubprocVecEnv for speed with multiple envs
    vec_env_cls = SubprocVecEnv if n_envs > 4 else DummyVecEnv
    print(f"🗃️  Używam {vec_env_cls.__name__} dla {n_envs} środowisk\n")
    
    # Create vectorized environment with masking
    env = make_vec_env(make_masked_env(), n_envs=n_envs, vec_env_cls=vec_env_cls)
    
    # Eval Env - użyj tego samego typu co training env
    eval_env = make_vec_env(make_masked_env(), n_envs=eval_n_envs, vec_env_cls=vec_env_cls)
    
    # Model
    policy_kwargs = config['model']['policy_kwargs'].copy()
    policy_kwargs['features_extractor_class'] = SolitaireFeaturesExtractor
    
    # Optimizer config
    optimizer_config = config['model'].get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adam')
    
    # Build optimizer kwargs
    optimizer_kwargs = {
        'eps': optimizer_config.get('eps', 1e-8),
        'betas': tuple(optimizer_config.get('betas', [0.9, 0.999]))
    }
    
    if optimizer_type == 'adamw':
        optimizer_kwargs['weight_decay'] = optimizer_config.get('weight_decay', 0.0001)
        optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.Adam
    
    print(f"\n{'='*70}")
    print(f"[OPTIMIZER CONFIG]")
    print(f"{'='*70}")
    print(f"  Type:           {optimizer_type.upper()}")
    print(f"  Weight Decay:   {optimizer_kwargs.get('weight_decay', 'N/A')}")
    print(f"  Epsilon:        {optimizer_kwargs['eps']}")
    print(f"  Betas:          {optimizer_kwargs['betas']}")
    print(f"{'='*70}\n")
    
    # Create model
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
    
    # Set optimizer
    model.policy.optimizer = optimizer_class(model.policy.parameters(), 
                                             lr=config['model']['learning_rate'],
                                             **optimizer_kwargs)
    
    # Callbacks
    plot_script_path = os.path.join(os.path.dirname(__file__), 'utils', 'plot_train_progress.py')
    
    # Stop training callback
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config['training']['max_no_improvement_evals'],
        min_evals=config['training']['min_evals'],
        verbose=1
    )
    
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=eval_freq // n_envs,  # Adjust for vectorized env
        plot_script_path=plot_script_path,
        plot_interval=plot_interval,
        callback_on_new_best=stop_train_callback,  # Poprawnie przekazany callback
        deterministic=False,  # Zmienione na False - lepiej ocenia eksploracyjną grę
        verbose=1
    )
    
    train_progress_callback = TrainProgressCallback(
        log_path=train_csv_path,
        initial_timesteps=0
    )
    
    loss_recorder_callback = LossRecorderCallback()
    
    # Entropy scheduler
    entropy_scheduler = EntropySchedulerCallback(
        initial_ent_coef=config['model']['ent_coef'],
        min_ent_coef=config['model'].get('min_ent_coef', 0.005),
        total_timesteps=total_timesteps
    )
    
    win_tracker = WinTrackerCallback(verbose=1)
    
    # Periodic save callback - zapisz model co 10 ewaluacji
    periodic_save_callback = PeriodicSaveCallback(
        save_path=model_path,
        eval_freq=eval_freq,
        save_interval=10,  # Zapisz co 10 ewaluacji
        verbose=1
    )
    
    callbacks = [
        eval_callback,
        train_progress_callback,
        loss_recorder_callback,
        entropy_scheduler,
        win_tracker,
        periodic_save_callback
    ]
    
    print("\n" + "="*70)
    print("[STARTING TRAINING]")
    print("="*70)
    print(f"  Total timesteps:    {total_timesteps:,}")
    print(f"  Environments:       {n_envs}")
    print(f"  Eval frequency:     {eval_freq:,}")
    print(f"  Batch size:         {config['model']['batch_size']:,}")
    print(f"  N steps:            {config['model']['n_steps']:,}")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo"
    )
    
    model.save(model_path)
    print("\n✅ Training finished!")
    print(f"📦 Model saved to: {model_path}")

if __name__ == "__main__":
    train()
