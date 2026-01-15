import os
import sys
import yaml
import pickle
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

# Dodaj katalog scripts do ścieżki
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import make_env
from cnn import CustomFeaturesExtractor
from utils.callbacks import TrainProgressCallback, CustomEvalCallback, LossRecorderCallback

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def ensure_directories():
    os.makedirs(os.path.join(base_dir, config['paths']['models_dir']), exist_ok=True)
    os.makedirs(os.path.join(base_dir, config['paths']['logs_dir']), exist_ok=True)

def prompt_training_mode(model_path, state_path, models_dir):
    """
    🎯 Prompt użytkownika o wybór trybu treningu
    
    Returns:
        tuple: (mode, use_config_hyperparams, total_timesteps, model_to_load)
            mode: 'continue' | 'restart'
            model_to_load: 'best' | 'latest'
    """
    has_full_model = os.path.exists(model_path)
    has_best_model = os.path.exists(os.path.join(models_dir, 'best_model.zip'))
    has_state = os.path.exists(state_path)
    
    total_timesteps = 0
    
    # Wczytaj timesteps dla pełnego modelu
    if has_full_model and has_state:
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                total_timesteps = state['total_timesteps']
        except:
            pass
    
    print(f"\n{'='*70}")
    print(f"[TRAINING MODE SELECTION]")
    print(f"{'='*70}")
    
    if has_full_model:
        print(f"✅ Znaleziono pełny model: {model_path}")
        print(f"   Timesteps: {total_timesteps:,}")
        
        # Pokaż dostępne modele
        print(f"\n📁 Dostępne modele:")
        if has_best_model:
            print(f"   ✅ best_model.zip (najlepszy - najwyższy reward)")
        print(f"   ✅ minesweeper_ppo.zip (bieżący)")
        
        print(f"\nWybierz tryb:")
        print(f"  [1] Kontynuuj trening (zachowaj hyperparametry z modelu)")
        print(f"  [2] Kontynuuj trening (użyj hyperparametrów z config.yaml)")
        print(f"  [3] Rozpocznij od zera (usuń stary model)")
        
        try:
            choice = input(f"\nWybór [1-3] (domyślnie: 2): ").strip()
        except:
            choice = '2'
        
        # Pytaj o wybór modelu przy kontynuacji
        model_choice = 'latest'
        if choice in ['1', '2', '']:
            if has_best_model:
                print(f"\nJaki model wczytać?")
                print(f"  [1] Najlepszy (best_model.zip - najwyższy reward)")
                print(f"  [2] Bieżący (minesweeper_ppo.zip)")
                try:
                    model_input = input(f"\nWybór [1-2] (domyślnie: 1): ").strip()
                    model_choice = 'best' if model_input in ['1', ''] else 'latest'
                except:
                    model_choice = 'best'
        
        if choice == '1':
            return 'continue', False, total_timesteps, model_choice
        elif choice == '3':
            return 'restart', True, 0, 'latest'
        else:  # '2' lub pusty
            return 'continue', True, total_timesteps, model_choice
    
    else:
        print(f"ℹ️  Nie znaleziono zapisanego modelu")
        print(f"   Rozpoczynam trening od zera")
        return 'restart', True, 0, 'latest'

def train():
    ensure_directories()
    
    # Parametry treningu
    n_envs = config['training']['n_envs']
    total_timesteps_config = config['training']['total_timesteps']
    eval_freq = config['training']['eval_freq']
    plot_interval = config['training']['plot_interval']
    eval_n_envs = config['training'].get('eval_n_envs', 4)
    eval_n_repeats = config['training'].get('eval_n_repeats', 5)
    
    # Normalizacja
    norm_config = config['training'].get('normalization', {})
    norm_obs = norm_config.get('norm_obs', False)
    norm_reward = norm_config.get('norm_reward', True)
    clip_obs = norm_config.get('clip_obs', 10.0)
    clip_reward = norm_config.get('clip_reward', 10.0)
    norm_gamma = norm_config.get('gamma', 0.99)
    epsilon = norm_config.get('epsilon', 1e-8)
    
    # Ścieżki
    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    vec_norm_path = model_path_absolute.replace('.zip', '_vecnorm.pkl')
    vec_norm_eval_path = model_path_absolute.replace('.zip', '_vecnorm_eval.pkl')
    state_path = model_path_absolute.replace('.zip', '_state.pkl')
    best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))
    
    # 🎯 INTERACTIVE MODE SELECTION
    mode, use_config_hyperparams, total_timesteps, model_choice = prompt_training_mode(
        model_path_absolute,
        state_path,
        best_model_save_path
    )
    
    print(f"\n{'='*70}")
    print(f"[SELECTED MODE]")
    print(f"{'='*70}")
    print(f"  Mode:                   {mode}")
    print(f"  Use config hyperparams: {use_config_hyperparams}")
    print(f"  Model to load:          {model_choice}")
    print(f"  Starting timesteps:     {total_timesteps:,}")
    print(f"{'='*70}\n")
    
    # Wybierz ścieżkę modelu
    if model_choice == 'best':
        actual_model_path = os.path.join(best_model_save_path, 'best_model.zip')
        print(f"📁 Wczytam: best_model.zip (najlepszy)")
    else:
        actual_model_path = model_path_absolute
        print(f"📁 Wczytam: minesweeper_ppo.zip (bieżący)")
    print()
    
    # Reset logs przy restarcie
    if mode == 'restart':
        for csv_path in [
            os.path.join(base_dir, config['paths']['train_csv_path']),
        ]:
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                    print(f"🗑️  Usunięto: {csv_path}")
            except Exception as e:
                print(f"⚠️  Nie udało się usunąć {csv_path}: {e}")
        print(f"✅ Zresetowano logi\n")
    else:
        print(f"✅ Kontynuacja treningu - logi będą kontynuowane\n")
    
    print(f"{'='*70}")
    print(f"[NORMALIZATION CONFIG]")
    print(f"{'='*70}")
    print(f"  norm_obs:        {norm_obs}")
    print(f"  norm_reward:     {norm_reward}")
    print(f"  clip_reward:     {clip_reward}")
    print(f"{'='*70}\n")
    
    # Wybór typu vec_env
    vec_env_cls = DummyVecEnv if n_envs < 8 else SubprocVecEnv
    print(f"🗃️  Używam {vec_env_cls.__name__} dla {n_envs} środowisk\n")
    
    # Środowisko treningowe
    env = make_vec_env(make_env(), n_envs=n_envs, vec_env_cls=vec_env_cls)
    if mode == 'continue' and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        print(f"✅ Załadowano VecNormalize z {vec_norm_path}")
    else:
        env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=norm_gamma,
            epsilon=epsilon
        )
        print(f"✅ Utworzono nowy VecNormalize")
    
    # Środowisko ewaluacyjne
    eval_env = make_vec_env(make_env(), n_envs=eval_n_envs, vec_env_cls=DummyVecEnv)
    if mode == 'continue' and os.path.exists(vec_norm_eval_path):
        eval_env = VecNormalize.load(vec_norm_eval_path, eval_env)
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=norm_gamma,
            epsilon=epsilon
        )
    
    # Model Setup
    policy_kwargs = config['model']['policy_kwargs'].copy()
    policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor
    
    # 🎯 MODEL LOADING BASED ON MODE
    if mode == 'continue' and not use_config_hyperparams:
        # Tryb 1: Kontynuacja z hyperparametrami z modelu
        print(f"\n📥 Ładowanie pełnego modelu (hyperparametry z modelu)...\n")
        model = PPO.load(actual_model_path, env=env)
        model.num_timesteps = total_timesteps
    
    elif mode == 'continue' and use_config_hyperparams:
        # Tryb 2: Kontynuacja z hyperparametrami z config
        print(f"\n📥 Ładowanie modelu (hyperparametry z config.yaml)...\n")
        model = PPO(
            config['model']['policy'],
            env,
            learning_rate=config['model']['learning_rate'],
            n_steps=config['model']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config['model']['device']
        )
        
        # Wczytaj tylko wagi policy
        model_tmp = PPO.load(actual_model_path)
        model.policy.load_state_dict(model_tmp.policy.state_dict())
        del model_tmp
        
        model.num_timesteps = total_timesteps
    
    else:  # mode == 'restart'
        # Tryb 3: Nowy model od zera
        print(f"\n🆕 Tworzenie nowego modelu od zera...\n")
        model = PPO(
            config['model']['policy'],
            env,
            learning_rate=config['model']['learning_rate'],
            n_steps=config['model']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config['model']['device']
        )
    
    # Callbacks
    stop_on_plateau = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config['training'].get('max_no_improvement_evals', 50),
        min_evals=config['training'].get('min_evals', 5),
        verbose=1
    )
    
    plot_script_path = os.path.join(os.path.dirname(__file__), 'utils', 'plot_train_progress.py')
    n_eval_episodes = eval_n_envs * eval_n_repeats
    
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_save_path,
        log_path=os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir'])),
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
        callback_after_eval=stop_on_plateau,
        plot_interval=plot_interval,
        plot_script_path=plot_script_path,
        initial_timesteps=total_timesteps,
        n_eval_episodes=n_eval_episodes
    )
    
    train_progress_callback = TrainProgressCallback(
        os.path.join(base_dir, config['paths']['train_csv_path']),
        initial_timesteps=total_timesteps
    )
    
    loss_recorder = LossRecorderCallback()
    
    print(f"Rozpoczynanie treningu...")
    
    try:
        remaining_timesteps = total_timesteps_config - total_timesteps
        
        if total_timesteps_config > 0 and total_timesteps / total_timesteps_config >= 0.8:
            print(f"ℹ️  Użyto {total_timesteps}/{total_timesteps_config} kroków ({total_timesteps/total_timesteps_config:.1%}).")
            try:
                extra = input("Ile dodatkowych kroków dodać? (0 = brak): ").strip()
                extra_int = int(extra) if extra != '' else 0
                remaining_timesteps += extra_int
            except:
                pass
        
        if remaining_timesteps > 0:
            print(f"\n🚀 Rozpoczynam trening: {remaining_timesteps:,} kroków\n")
            model.num_timesteps = 0
            model.learn(
                total_timesteps=remaining_timesteps,
                reset_num_timesteps=False,
                callback=[eval_callback, train_progress_callback, loss_recorder],
                progress_bar=True
            )
        else:
            print(f"ℹ️  Trening zakończony: osiągnięto {total_timesteps} kroków.")
        
        # Zapisz model i state
        model.save(model_path_absolute)
        env.save(vec_norm_path)
        eval_env.save(vec_norm_eval_path)
        
        # Zapisz state z timesteps
        with open(state_path, 'wb') as f:
            pickle.dump({
                'total_timesteps': total_timesteps + remaining_timesteps
            }, f)
        
        print("✅ Trening zakończony i model zapisany.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Przerwanie treningu przez użytkownika (Ctrl+C)")
        model.save(model_path_absolute)
        env.save(vec_norm_path)
        eval_env.save(vec_norm_eval_path)
        with open(state_path, 'wb') as f:
            pickle.dump({
                'total_timesteps': total_timesteps + model.num_timesteps
            }, f)
        print("Zapisano.")
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Trening przerwany.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Błąd podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
