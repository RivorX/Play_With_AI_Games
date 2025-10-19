import os
import sys
import yaml
import pickle
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
import matplotlib
matplotlib.use('Agg')

# Import lokalnych modułów
from model import make_env
from cnn import CustomFeaturesExtractor
from utils.callbacks import TrainProgressCallback, CustomEvalCallback, LossRecorderCallback, EntropySchedulerCallback
from utils.gradient_monitor import GradientWeightMonitor
from utils.training_utils import (
    linear_schedule, 
    entropy_schedule,
    apply_gradient_clipping, 
    ensure_directories,
    reset_channel_logs,
    init_channel_loggers,
    log_observation
)

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ensure_directories(base_dir)

# Channel logging (opcjonalne)
enable_channel_logs = config.get('training', {}).get('enable_channel_logs', False)
channel_loggers = init_channel_loggers(base_dir) if enable_channel_logs else {}


def setup_adamw_optimizer(model, config):
    """
    Konfiguruje optimizer AdamW dla modelu RecurrentPPO
    
    Args:
        model: Model RecurrentPPO
        config: Konfiguracja z config.yaml
    """
    opt_config = config['model'].get('optimizer', {})
    optimizer_type = opt_config.get('type', 'adam').lower()
    weight_decay = opt_config.get('weight_decay', 0.01)
    eps = opt_config.get('eps', 1e-5)
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    
    if optimizer_type == 'adamw':
        # ✅ Utwórz AdamW optimizer
        optimizer = torch.optim.AdamW(
            model.policy.parameters(),
            lr=model.learning_rate if isinstance(model.learning_rate, float) else model.learning_rate(1.0),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
        # Zastąp optimizer w modelu
        model.policy.optimizer = optimizer
        
        print(f"\n{'='*70}")
        print(f"[OPTIMIZER] ✅ AdamW ENABLED")
        print(f"{'='*70}")
        print(f"  Type:          AdamW")
        print(f"  Weight Decay:  {weight_decay} {'(L2 regularization)' if weight_decay > 0 else '(DISABLED)'}")
        print(f"  Epsilon:       {eps}")
        print(f"  Betas:         {betas}")
        print(f"  Learning Rate: {model.learning_rate if isinstance(model.learning_rate, float) else 'schedule'}")
        print(f"{'='*70}\n")
        
    elif optimizer_type == 'adam':
        # Domyślny Adam (już jest w SB3)
        print(f"\n{'='*70}")
        print(f"[OPTIMIZER] Standard Adam (default SB3)")
        print(f"{'='*70}")
        print(f"  Type:          Adam")
        print(f"  Epsilon:       {eps}")
        print(f"  Betas:         {betas}")
        print(f"{'='*70}\n")
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'adamw'.")
    
    return model


def train(use_progress_bar=False, use_config_hyperparams=True):
    global best_model_save_path

    n_envs = config['training']['n_envs']
    n_steps = config['model']['n_steps']
    eval_freq = config['training']['eval_freq']
    plot_interval = config['training']['plot_interval']
    eval_n_envs = config['training'].get('eval_n_envs', 4)
    eval_n_repeats = config['training'].get('eval_n_repeats', 2)
    
    # Ustawienia normalizacji z configu
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

    load_model = os.path.exists(model_path_absolute)
    total_timesteps = 0
    
    if load_model:
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                total_timesteps = state['total_timesteps']
            print(f"Wznowienie treningu od {total_timesteps} kroków.")
        except Exception as e:
            print(f"Błąd ładowania stanu: {e}. Zaczynam od zera.")
        
        try:
            resp = input(f"Znaleziono istniejący model pod {model_path_absolute}. Czy kontynuować trening? [[Y]/n]: ").strip()
        except Exception:
            resp = ''

        if resp.lower() in ('n', 'no'):
            print("Użytkownik wybrał rozpoczęcie treningu od nowa. Zaczynam od zera i używam hyperparametrów z configu.")
            load_model = False
            total_timesteps = 0
            use_config_hyperparams = True
            
            # ✅ RESETOWANIE CSV: train_progress.csv
            try:
                train_csv = os.path.join(base_dir, config['paths']['train_csv_path'])
                if os.path.exists(train_csv):
                    os.remove(train_csv)
                    print(f"Usunięto istniejący plik postępu treningu: {train_csv}")
            except Exception as e:
                print(f"Nie udało się usunąć pliku postępu treningu: {e}")
            
            # ✅ RESETOWANIE CSV: gradient_monitor.csv
            try:
                gradient_csv = os.path.join(base_dir, 'logs', 'gradient_monitor.csv')
                if os.path.exists(gradient_csv):
                    os.remove(gradient_csv)
                    print(f"Usunięto istniejący plik gradient monitor: {gradient_csv}")
            except Exception as e:
                print(f"Nie udało się usunąć pliku gradient monitor: {e}")
        else:
            try:
                resp2 = input("Użyć hyperparametrów z configu zamiast z modelu? [[Y]/n]: ").strip()
            except Exception:
                resp2 = ''
            use_config_hyperparams = False if resp2.lower() in ('n', 'no') else True

    reset_channel_logs(base_dir)
    if enable_channel_logs:
        channel_loggers.update(init_channel_loggers(base_dir))

    # Wyświetl info o normalizacji
    print(f"\n{'='*70}")
    print(f"[NORMALIZATION CONFIG]")
    print(f"{'='*70}")
    print(f"  norm_obs:        {norm_obs} {'❌ DISABLED (recommended for images)' if not norm_obs else '✅ ENABLED'}")
    print(f"  norm_reward:     {norm_reward} {'✅ ENABLED (stabilizes training)' if norm_reward else '❌ DISABLED'}")
    print(f"  clip_obs:        {clip_obs}")
    print(f"  clip_reward:     {clip_reward}")
    print(f"  gamma:           {norm_gamma}")
    print(f"  epsilon:         {epsilon}")
    print(f"{'='*70}\n")

    # Tworzenie środowisk treningowych
    env = make_vec_env(make_env(render_mode=None, grid_size=None), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    if load_model and os.path.exists(vec_norm_path):
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
        print(f"✅ Utworzono nowy VecNormalize z ustawieniami z config.yaml")

    # Tworzenie środowisk ewaluacyjnych
    eval_env = make_vec_env(make_env(render_mode=None, grid_size=16), n_envs=eval_n_envs, vec_env_cls=SubprocVecEnv)
    if load_model and os.path.exists(vec_norm_eval_path):
        eval_env = VecNormalize.load(vec_norm_eval_path, eval_env)
        print(f"✅ Załadowano eval VecNormalize z {vec_norm_eval_path}")
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
        print(f"✅ Utworzono nowy eval VecNormalize z ustawieniami z config.yaml")

    # Tworzenie lub ładowanie modelu
    policy_kwargs = config['model']['policy_kwargs'].copy()
    policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor

    if load_model:
        model = RecurrentPPO.load(model_path_absolute, env=env)
        model.ent_coef = config['model']['ent_coef']
        model.learning_rate = linear_schedule(config['model']['learning_rate'], config['model']['min_learning_rate'])
        model._setup_lr_schedule()
        
        # ✅ NOWE: Skonfiguruj AdamW dla załadowanego modelu
        model = setup_adamw_optimizer(model, config)
    else:
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
        
        # ✅ NOWE: Skonfiguruj AdamW dla nowego modelu
        model = setup_adamw_optimizer(model, config)

    if use_config_hyperparams and load_model:
        policy_kwargs = config['model']['policy_kwargs'].copy()
        policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor
        model_new = RecurrentPPO(
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
        model_tmp = RecurrentPPO.load(model_path_absolute)
        model_new.policy.load_state_dict(model_tmp.policy.state_dict())
        model = model_new
        del model_tmp
        
        # ✅ NOWE: Skonfiguruj AdamW po przeniesieniu wag
        model = setup_adamw_optimizer(model, config)

    # Zastosuj gradient clipping (kluczowa naprawa!)
    apply_gradient_clipping(model, clip_value=1.0)

    global best_model_save_path
    best_model_save_path = os.path.normpath(os.path.join(base_dir, config['paths']['models_dir']))

    # Callbacki
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
    
    # Entropy scheduler - zmniejsza entropię liniowo w czasie
    entropy_scheduler = EntropySchedulerCallback(
        entropy_schedule_fn=entropy_schedule(
            config['model']['ent_coef'], 
            config['model'].get('min_ent_coef', 0.001)
        ),
        initial_timesteps=total_timesteps,
        verbose=1
    )
    
    print(f"\n{'='*70}")
    print(f"[ENTROPY SCHEDULE]")
    print(f"{'='*70}")
    print(f"  Initial ent_coef:  {config['model']['ent_coef']}")
    print(f"  Min ent_coef:      {config['model'].get('min_ent_coef', 0.001)}")
    print(f"  Schedule:          Linear decay (similar to learning rate)")
    print(f"{'='*70}\n")

    # ✅ NOWE: Gradient & Weight Monitor
    gradient_monitor = GradientWeightMonitor(
        csv_path=os.path.join(base_dir, 'logs', 'gradient_monitor.csv'),
        log_freq=config['training'].get('gradient_log_freq', 2000)
    )

    # Debug logging (jeśli włączone)
    if enable_channel_logs:
        for logger in channel_loggers.values():
            logger.info(f"\n--- Debug: Rozpoczęcie treningu ---")
        debug_env = make_env(render_mode=None, grid_size=None)()
        obs, _ = debug_env.reset()
        grid_size = debug_env.grid_size
        log_observation(obs, channel_loggers, grid_size, step=0)
        for debug_step in range(5):
            action = debug_env.action_space.sample()
            obs, reward, terminated, truncated, info = debug_env.step(action)
            log_observation(obs, channel_loggers, grid_size, step=debug_step + 1)
            if enable_channel_logs:
                for logger in channel_loggers.values():
                    logger.info(f"Akcja: {action}, Nagroda: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
            if terminated or truncated:
                break
        debug_env.close()
        if enable_channel_logs:
            for logger in channel_loggers.values():
                logger.info(f"--- Koniec debug dla grid_size={grid_size} ---")

    # Rozpocznij trening
    try:
        configured_total = config['training'].get('total_timesteps', 0)
        remaining_timesteps = configured_total - total_timesteps
        
        try:
            if configured_total > 0 and total_timesteps / configured_total >= 0.8:
                print(f"Użyto {total_timesteps}/{configured_total} kroków ({total_timesteps/configured_total:.1%}). To >=80% limitu.")
                extra = input("Ile dodatkowych kroków dodać? (0 = brak, domyślnie 0): ").strip()
                try:
                    extra_int = int(extra) if extra != '' else 0
                except ValueError:
                    print("Niepoprawna wartość, używam 0 dodatkowych kroków.")
                    extra_int = 0
                remaining_timesteps += extra_int
        except Exception:
            pass
        
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                reset_num_timesteps=not load_model,
                callback=[eval_callback, train_progress_callback, loss_recorder, entropy_scheduler, gradient_monitor],
                progress_bar=use_progress_bar,
                tb_log_name=f"recurrent_ppo_snake_{total_timesteps}"
            )
        else:
            print(f"Trening zakończony: osiągnięto {total_timesteps} kroków.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Przerwanie treningu przez użytkownika (Ctrl+C)")
        print("Zamykanie środowisk...")
        try:
            env.close()
            eval_env.close()
        except:
            pass
        raise
    except Exception as e:
        print(f"Błąd podczas treningu: {e}")
        raise
    finally:
        try:
            env.close()
            eval_env.close()
        except:
            pass

    print("Trening zakończony!")


if __name__ == "__main__":
    try:
        train(use_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nTrening przerwany.")
        os._exit(0)
    except Exception as e:
        print(f"Błąd podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("Trening zakończony!")