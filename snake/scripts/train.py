import os
import sys
import yaml
import pickle
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
import matplotlib
matplotlib.use('Agg')

from model import make_env
from cnn import CustomFeaturesExtractor
from utils.callbacks import TrainProgressCallback, CustomEvalCallback, LossRecorderCallback, EntropySchedulerCallback, VictoryTrackerCallback
from utils.gradient_monitor import GradientWeightMonitor
from utils.training_utils import (
    linear_schedule, 
    entropy_schedule,
    apply_gradient_clipping, 
    ensure_directories,
    reset_channel_logs,
    init_channel_loggers,
    log_observation,
    clear_gpu_cache,
    enable_pin_memory,
    AsyncRolloutPrefetcher,
    setup_adamw_optimizer,
    load_policy_weights_only,
    cleanup_all_training_csvs
)

base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ensure_directories(base_dir)

enable_channel_logs = config.get('training', {}).get('enable_channel_logs', False)
channel_loggers = init_channel_loggers(base_dir) if enable_channel_logs else {}


def prompt_training_mode(model_path, policy_path, state_path, models_dir):
    """
    🎯 Prompt użytkownika o wybór trybu treningu
    
    Returns:
        tuple: (mode, use_config_hyperparams, total_timesteps, model_to_load)
            mode: 'continue' | 'restart' | 'policy_only'
            model_to_load: 'best' | 'latest' | model_path
    """
    has_full_model = os.path.exists(model_path)
    has_best_model = os.path.exists(os.path.join(models_dir, 'best_model.zip'))
    has_latest_model = os.path.exists(os.path.join(models_dir, 'snake_ppo_model.zip'))
    has_policy_only = os.path.exists(policy_path) and not has_full_model
    has_state = os.path.exists(state_path)
    
    total_timesteps = 0
    
    # ✅ FIX: Wczytuj timesteps TYLKO dla pełnego modelu
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
        
        # 🔍 Pokaż dostępne modele
        print(f"\n📁 Dostępne modele:")
        if has_best_model:
            print(f"   ✅ best_model.zip (najlepszy - najwyższy reward)")
        if has_latest_model:
            print(f"   ✅ snake_ppo_model.zip (najnowszy - bieżący)")
        
        print(f"\nWybierz tryb:")
        print(f"  [1] Kontynuuj trening (zachowaj hyperparametry z modelu)")
        print(f"  [2] Kontynuuj trening (użyj hyperparametrów z config.yaml)")
        print(f"  [3] Rozpocznij od zera (usuń stary model)")
        print(f"  [4] Wczytaj tylko wagi z policy.pth")
        
        try:
            choice = input(f"\nWybór [1-4] (domyślnie: 2): ").strip()
        except:
            choice = '2'
        
        # 🎯 Pytaj o wybór modelu przy kontynuacji (PRZED sprawdzeniem choice)
        model_choice = 'latest'
        if choice in ['1', '2', '']:  # ✅ Dodaj '' dla domyślnej opcji
            if has_best_model and has_latest_model:
                print(f"\nJaki model wczytać?")
                print(f"  [1] Najlepszy (best_model.zip - najwyższy reward)")
                print(f"  [2] Najnowszy (snake_ppo_model.zip - bieżący)")
                try:
                    model_input = input(f"\nWybór [1-2] (domyślnie: 1): ").strip()
                    model_choice = 'best' if model_input in ['1', ''] else 'latest'
                except:
                    model_choice = 'best'
            elif has_best_model:
                model_choice = 'best'
                print(f"\n✅ Wczytam best_model.zip (brak snake_ppo_model.zip)")
            elif has_latest_model:
                model_choice = 'latest'
                print(f"\n✅ Wczytam snake_ppo_model.zip (brak best_model.zip)")
        
        if choice == '1':
            return 'continue', False, total_timesteps, model_choice
        elif choice == '3':
            return 'restart', True, 0, 'latest'  # ✅ Timesteps = 0
        elif choice == '4':
            if os.path.exists(policy_path):
                return 'policy_only', True, 0, 'policy_only'  # ✅ Timesteps = 0
            else:
                print(f"❌ Nie znaleziono policy.pth: {policy_path}")
                print(f"   Kontynuuję z domyślną opcją (2)")
                return 'continue', True, total_timesteps, model_choice
        else:  # '2' lub pusty
            return 'continue', True, total_timesteps, model_choice
    
    elif has_policy_only:
        print(f"⚠️  Znaleziono tylko policy.pth: {policy_path}")
        print(f"   (brak pełnego modelu .zip)")
        print(f"\nWybierz tryb:")
        print(f"  [1] Wczytaj wagi z policy.pth (optimizer od zera)")
        print(f"  [2] Rozpocznij od zera (ignoruj policy.pth)")
        
        try:
            choice = input(f"\nWybór [1-2] (domyślnie: 1): ").strip()
        except:
            choice = '1'
        
        if choice == '2':
            return 'restart', True, 0, 'latest'
        else:
            return 'policy_only', True, 0, 'policy_only'  # ✅ Zawsze 0 dla policy_only
    
    else:
        print(f"ℹ️  Nie znaleziono zapisanego modelu")
        print(f"   Rozpoczynam trening od zera")
        return 'restart', True, 0, 'latest'

def train(use_progress_bar=False):
    n_envs = config['training']['n_envs']
    n_steps = config['model']['n_steps']
    eval_freq = config['training']['eval_freq']
    plot_interval = config['training']['plot_interval']
    eval_n_envs = config['training'].get('eval_n_envs', 4)
    eval_n_repeats = config['training'].get('eval_n_repeats', 2)
    
    norm_config = config['training'].get('normalization', {})
    norm_obs = norm_config.get('norm_obs', False)
    norm_reward = norm_config.get('norm_reward', True)
    clip_obs = norm_config.get('clip_obs', 10.0)
    clip_reward = norm_config.get('clip_reward', 10.0)
    norm_gamma = norm_config.get('gamma', 0.99)
    epsilon = norm_config.get('epsilon', 1e-8)

    model_path_absolute = os.path.normpath(os.path.join(base_dir, config['paths']['model_path']))
    vec_norm_path = model_path_absolute.replace('.zip', '_vecnorm.pkl')
    vec_norm_eval_path = model_path_absolute.replace('.zip', '_vecnorm_eval.pkl')
    state_path = model_path_absolute.replace('.zip', '_state.pkl')
    policy_only_path = os.path.join(base_dir, 'models', 'policy.pth')
    best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))

    # 🎯 INTERACTIVE MODE SELECTION
    mode, use_config_hyperparams, total_timesteps, model_choice = prompt_training_mode(
        model_path_absolute, 
        policy_only_path, 
        state_path,
        best_model_save_path
    )
    
    print(f"\n{'='*70}")
    print(f"[SELECTED MODE]")
    print(f"{'='*70}")
    print(f"  Mode:                  {mode}")
    print(f"  Use config hyperparams: {use_config_hyperparams}")
    print(f"  Model to load:         {model_choice}")
    print(f"  Starting timesteps:     {total_timesteps:,}")
    print(f"{'='*70}\n")
    
    # 🎯 Wybierz ścieżkę modelu na podstawie model_choice
    if model_choice == 'best':
        actual_model_path = os.path.join(best_model_save_path, 'best_model.zip')
        print(f"📁 Wczytam: best_model.zip (najlepszy)")
    elif model_choice == 'latest':
        actual_model_path = os.path.join(best_model_save_path, 'snake_ppo_model.zip')
        print(f"📁 Wczytam: snake_ppo_model.zip (najnowszy)")
        # Fallback na domyślny jeśli nie istnieje
        if not os.path.exists(actual_model_path):
            actual_model_path = model_path_absolute
            print(f"⚠️  Fallback: {model_path_absolute}")
    else:  # policy_only
        actual_model_path = policy_only_path
        print(f"📁 Wczytam: policy.pth")
    print()

    # 🔧 CLEANUP CSVs przy kontynuacji (usuń nadmiarowe wiersze)
    if mode == 'continue' and total_timesteps > 0:
        cleanup_all_training_csvs(base_dir, total_timesteps, verbose=True)
    
    # Reset logs przy restarcie lub policy_only
    if mode in ['restart', 'policy_only']:
        for csv_path in [
            os.path.join(base_dir, config['paths']['train_csv_path']),
            os.path.join(base_dir, 'logs', 'gradient_monitor.csv'),
            os.path.join(base_dir, 'logs', 'victories.log')
        ]:
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                    print(f"🗑️  Usunięto: {csv_path}")
            except Exception as e:
                print(f"⚠️  Nie udało się usunąć {csv_path}: {e}")
        
        reset_channel_logs(base_dir)
        print(f"✅ Zresetowano logi\n")
    else:
        print(f"✅ Kontynuacja treningu - logi będą kontynuowane\n")

    # Inicjalizuj channel loggers
    if enable_channel_logs:
        channel_loggers.update(init_channel_loggers(base_dir))

    print(f"{'='*70}")
    print(f"[NORMALIZATION CONFIG]")
    print(f"{'='*70}")
    print(f"  norm_obs:        {norm_obs}")
    print(f"  norm_reward:     {norm_reward}")
    print(f"  clip_reward:     {clip_reward}")
    print(f"{'='*70}\n")

    # ⚡ OPTIMIZATION: DummyVecEnv dla małej liczby środowisk
    vec_env_cls = DummyVecEnv if n_envs < 8 else SubprocVecEnv
    print(f"🗃️  Używam {vec_env_cls.__name__} dla {n_envs} środowisk\n")
    
    env = make_vec_env(make_env(render_mode=None, grid_size=None), n_envs=n_envs, vec_env_cls=vec_env_cls)
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

    # Eval env
    eval_env = make_vec_env(make_env(render_mode=None, grid_size=16), n_envs=eval_n_envs, vec_env_cls=DummyVecEnv)
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

    # ⚡ OPTIMIZATION: Enable pin_memory
    env, eval_env = enable_pin_memory(env, eval_env)

    clear_gpu_cache()

    policy_kwargs = config['model']['policy_kwargs'].copy()
    policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor

    # 🎯 MODEL LOADING BASED ON MODE
    if mode == 'continue' and not use_config_hyperparams:
        # Tryb 1: Kontynuacja z hyperparametrami z modelu
        print(f"\n📥 Ładowanie pełnego modelu (hyperparametry z modelu)...\n")
        model = RecurrentPPO.load(actual_model_path, env=env)
        model = setup_adamw_optimizer(model, config)
        clear_gpu_cache()
    
    elif mode == 'continue' and use_config_hyperparams:
        # Tryb 2: Kontynuacja z hyperparametrami z config
        print(f"\n📥 Ładowanie modelu (hyperparametry z config.yaml)...\n")
        model = RecurrentPPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate']),
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
        model_tmp = RecurrentPPO.load(actual_model_path)
        model.policy.load_state_dict(model_tmp.policy.state_dict())
        del model_tmp
        
        model = setup_adamw_optimizer(model, config)
        clear_gpu_cache()
    
    elif mode == 'policy_only':
        # Tryb 4: Wczytaj tylko wagi z policy.pth
        print(f"\n📥 Tworzenie nowego modelu (wagi z policy.pth)...\n")
        model = RecurrentPPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate']),
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
        
        model = load_policy_weights_only(model, policy_only_path)
        model = setup_adamw_optimizer(model, config)
        clear_gpu_cache()
    
    else:  # mode == 'restart'
        # Tryb 3: Nowy model od zera
        print(f"\n🆕 Tworzenie nowego modelu od zera...\n")
        model = RecurrentPPO(
            config['model']['policy'],
            env,
            learning_rate=linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate']),
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
        model = setup_adamw_optimizer(model, config)
    
    clip_value = config['model'].get('lstm', {}).get('gradient_clip_val', 5.0)
    apply_gradient_clipping(model, clip_value=clip_value)

    # ✅ best_model_save_path już zdefiniowany wcześniej (linia ~163)

    stop_on_plateau = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config['training']['max_no_improvement_evals'],
        min_evals=config['training']['min_evals'],
        verbose=1
    )

    plot_script_path = os.path.join(os.path.dirname(__file__), 'utils', 'plot_train_progress.py')
    n_eval_episodes = eval_n_envs * eval_n_repeats
    
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_save_path,
        log_path=os.path.normpath(os.path.join(base_dir, config['paths']['logs_dir'])),
        eval_freq=eval_freq,
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
    
    entropy_scheduler = EntropySchedulerCallback(
        entropy_schedule_fn=entropy_schedule(
            config['model']['ent_coef'], 
            config['model'].get('min_ent_coef', 0.001)
        ),
        initial_timesteps=total_timesteps,
        verbose=1
    )

    gradient_monitor = GradientWeightMonitor(
        csv_path=os.path.join(base_dir, 'logs', 'gradient_monitor.csv'),
        log_freq=config['training'].get('gradient_log_freq', 2000),
        initial_timesteps=total_timesteps
    )
    
    victory_tracker = VictoryTrackerCallback(
        log_dir=os.path.join(base_dir, 'logs'),
        verbose=1
    )

    # ⚡ OPTIMIZATION: Async Rollout Prefetcher
    prefetcher = AsyncRolloutPrefetcher(env, batch_queue_size=2)
    prefetcher.start()

    try:
        configured_total = config['training'].get('total_timesteps', 0)
        remaining_timesteps = configured_total - total_timesteps
        
        if configured_total > 0 and total_timesteps / configured_total >= 0.8:
            print(f"ℹ️  Użyto {total_timesteps}/{configured_total} kroków ({total_timesteps/configured_total:.1%}).")
            try:
                extra = input("Ile dodatkowych kroków dodać? (0 = brak): ").strip()
                extra_int = int(extra) if extra != '' else 0
                remaining_timesteps += extra_int
            except:
                pass
        
        if remaining_timesteps > 0:
            print(f"\n🚀 Rozpoczynam trening: {remaining_timesteps:,} kroków\n")
            model.learn(
                total_timesteps=remaining_timesteps,
                reset_num_timesteps=(mode != 'continue'),
                callback=[
                    eval_callback, 
                    train_progress_callback, 
                    loss_recorder, 
                    entropy_scheduler, 
                    gradient_monitor,
                    victory_tracker
                ],
                progress_bar=use_progress_bar,
                tb_log_name=f"recurrent_ppo_snake_{total_timesteps}"
            )
        else:
            print(f"ℹ️  Trening zakończony: osiągnięto {total_timesteps} kroków.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Przerwanie treningu przez użytkownika (Ctrl+C)")
        try:
            env.close()
            eval_env.close()
        except:
            pass
        clear_gpu_cache()
        raise
    except Exception as e:
        print(f"❌ Błąd podczas treningu: {e}")
        raise
    finally:
        try:
            env.close()
            eval_env.close()
        except:
            pass
        clear_gpu_cache()

    print("\n✅ Trening zakończony!")


if __name__ == "__main__":
    try:
        train(use_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Trening przerwany.")
        clear_gpu_cache()
        os._exit(0)
    except Exception as e:
        print(f"❌ Błąd podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_cache()
        sys.exit(1)