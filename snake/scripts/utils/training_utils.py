import os
import time
import pickle
import torch
import logging
import numpy as np


def linear_schedule(initial_value, min_value=0.00005):
    """Learning rate scheduler - linear decay"""
    def func(progress_remaining):
        return min_value + (initial_value - min_value) * progress_remaining
    return func


def entropy_schedule(initial_value, min_value=0.001):
    """Entropy coefficient scheduler - linear decay
    
    Args:
        initial_value: Początkowa wartość entropii (np. 0.05)
        min_value: Minimalna wartość entropii (np. 0.001)
    
    Returns:
        Function that returns current entropy based on progress
    """
    def func(progress_remaining):
        return min_value + (initial_value - min_value) * progress_remaining
    return func


def apply_gradient_clipping(model, clip_value=1.0):
    """
    ✅ KLUCZOWA NAPRAWA: Gradient Clipping dla LSTM
    Zapobiega exploding gradients w LSTM (MLP gradient 1.0 → 0.4)
    """
    def clip_grad_hook(module, grad_input, grad_output):
        """Hook który clippuje gradienty w backward pass"""
        if grad_input is not None:
            clipped = tuple(
                torch.clamp(g, -clip_value, clip_value) if g is not None else g 
                for g in grad_input
            )
            return clipped
        return grad_input
    
    # Zarejestruj hook dla LSTM actor
    if hasattr(model.policy, 'lstm_actor'):
        model.policy.lstm_actor.register_full_backward_hook(clip_grad_hook)
        print(f"[GRADIENT CLIPPING] ✅ Enabled for LSTM Actor (clip_value={clip_value})")
    
    # Opcjonalnie dla LSTM critic (jeśli enable_critic_lstm=true)
    if hasattr(model.policy, 'lstm_critic'):
        model.policy.lstm_critic.register_full_backward_hook(clip_grad_hook)
        print(f"[GRADIENT CLIPPING] ✅ Enabled for LSTM Critic (clip_value={clip_value})")


def save_training_state(model, env, eval_env, total_timesteps, save_path):
    """Zapisz stan treningu (model + VecNormalize + timesteps)"""
    model_path_absolute = os.path.normpath(os.path.join(save_path, 'snake_ppo_model.zip'))
    vec_norm_path = model_path_absolute.replace('.zip', '_vecnorm.pkl')
    vec_norm_eval_path = model_path_absolute.replace('.zip', '_vecnorm_eval.pkl')
    state_path = model_path_absolute.replace('.zip', '_state.pkl')

    for attempt in range(5):
        try:
            model.save(model_path_absolute)
            env.save(vec_norm_path)
            eval_env.save(vec_norm_eval_path)
            with open(state_path, 'wb') as f:
                pickle.dump({'total_timesteps': total_timesteps}, f)
            print(f"Zaktualizowano najnowszy model: {model_path_absolute} po {total_timesteps} krokach.")
            break
        except PermissionError as e:
            print(f"Próba {attempt + 1}/5: Nie udało się zapisać stanu: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"Błąd zapisu stanu: {e}")
            break
    else:
        print("Nie udało się zapisać stanu po 5 próbach.")


def ensure_directories(base_dir):
    """Utwórz wymagane katalogi"""
    dirs = [
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'logs', 'Training_channels')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def reset_channel_logs(base_dir):
    """Resetuj logi kanałów"""
    log_dir = os.path.join(base_dir, 'logs', 'Training_channels')
    os.makedirs(log_dir, exist_ok=True)
    for channel_name in ['mapa', 'direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']:
        log_path = os.path.join(log_dir, f'training_{channel_name}.log')
        with open(log_path, 'w', encoding='utf-8'):
            pass


def init_channel_loggers(base_dir):
    """Inicjalizuj loggery dla kanałów"""
    loggers = {}
    log_dir = os.path.join(base_dir, 'logs', 'Training_channels')
    for channel_name in ['mapa', 'direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']:
        log_path = os.path.join(log_dir, f'training_{channel_name}.log')
        logger = logging.getLogger(f'channel_{channel_name}')
        handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        loggers[channel_name] = logger
    return loggers


def log_observation(obs, channel_loggers, grid_size, step):
    """Loguj obserwację do plików debug"""
    if not channel_loggers:
        return
    
    image = obs['image']
    mapa = image[:, :, 0]
    head_pos = np.where(mapa == 1.0)
    food_pos = np.where(mapa == 0.75)
    
    if len(head_pos[0]) > 0:
        head_x, head_y = head_pos[0][0], head_pos[1][0]
    else:
        head_x, head_y = -1, -1
    if len(food_pos[0]) > 0:
        food_x, food_y = food_pos[0][0], food_pos[1][0]
    else:
        food_x, food_y = -1, -1
    
    if head_x >= 0 and food_x >= 0:
        distance = abs(head_x - food_x) + abs(head_y - food_y)
    else:
        distance = float('inf')
    
    logger = channel_loggers.get('mapa')
    if logger:
        logger.info(f"--- Obserwacja dla grid_size={grid_size}, krok={step} ---")
        logger.info(f"Kanał mapa:\n{np.array_str(mapa, precision=2, suppress_small=True, max_line_width=120)}")
        logger.info(f"Pozycja głowy: ({head_x}, {head_y}) | Pozycja jedzenia: ({food_x}, {food_y})")
        logger.info(f"Dystans Manhattan: {distance}")
        logger.info("-" * 60)

    for scalar_name in ['direction', 'dx_head', 'dy_head', 'front_coll', 'left_coll', 'right_coll']:
        logger = channel_loggers.get(scalar_name)
        if logger:
            logger.info(f"--- Obserwacja dla grid_size={grid_size}, krok={step} ---")
            if scalar_name == 'direction':
                logger.info(f"{scalar_name.capitalize()}: sin={obs[scalar_name][0]:.3f}, cos={obs[scalar_name][1]:.3f}")
            else:
                logger.info(f"{scalar_name.capitalize()}: {obs[scalar_name][0]}")
            logger.info("-" * 60)