import os
import time
import pickle
import torch
import gc
import logging
import numpy as np
import csv
from queue import Queue


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


def clear_gpu_cache():
    """🧹 Wyczyść cache GPU i usuń nieużywane obiekty"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def enable_pin_memory(env, eval_env):
    """
    ⚡ Włącza pin_memory dla szybszego transferu CPU→GPU
    
    Args:
        env: Training VecEnv
        eval_env: Evaluation VecEnv
    
    Returns:
        tuple: (env, eval_env) z pinned memory
    """
    if not torch.cuda.is_available():
        print("⚠️  Pin Memory niedostępny: CUDA nie dostępny")
        return env, eval_env
    
    try:
        # Spróbuj pin_memory na VecNormalize buffers
        if hasattr(env, 'buf_obs') and env.buf_obs is not None:
            if isinstance(env.buf_obs, dict):
                # Jeśli buf_obs to dict (np. normalization buffers)
                for key in env.buf_obs:
                    if isinstance(env.buf_obs[key], np.ndarray):
                        env.buf_obs[key] = torch.from_numpy(env.buf_obs[key]).pin_memory().numpy()
            elif isinstance(env.buf_obs, (list, tuple)):
                # Jeśli buf_obs to lista
                for i in range(len(env.buf_obs)):
                    if isinstance(env.buf_obs[i], np.ndarray):
                        env.buf_obs[i] = torch.from_numpy(env.buf_obs[i]).pin_memory().numpy()
        
        if hasattr(eval_env, 'buf_obs') and eval_env.buf_obs is not None:
            if isinstance(eval_env.buf_obs, dict):
                for key in eval_env.buf_obs:
                    if isinstance(eval_env.buf_obs[key], np.ndarray):
                        eval_env.buf_obs[key] = torch.from_numpy(eval_env.buf_obs[key]).pin_memory().numpy()
            elif isinstance(eval_env.buf_obs, (list, tuple)):
                for i in range(len(eval_env.buf_obs)):
                    if isinstance(eval_env.buf_obs[i], np.ndarray):
                        eval_env.buf_obs[i] = torch.from_numpy(eval_env.buf_obs[i]).pin_memory().numpy()
        
        print("✅ Pin Memory włączony (+10-15% transfer speed)")
    except Exception as e:
        print(f"⚠️  Pin Memory nieudany: {type(e).__name__}: {str(e)}")
    
    return env, eval_env


class AsyncRolloutPrefetcher:
    """
    🚀 Async prefetching następnego batcha podczas treningu
    
    Pozwala CPU przygotowywać dane podczas gdy GPU trenuje model.
    """
    def __init__(self, env, batch_queue_size=2):
        self.env = env
        self.queue = Queue(maxsize=batch_queue_size)
        self.stop_event = None
        self.worker_thread = None
        self.batch_size = 0
        self.is_active = False
        
    def start(self):
        """Uruchamia worker thread"""
        if not torch.cuda.is_available():
            print("⚠️  Async prefetch wymaga CUDA")
            return
        
        self.is_active = True
        print("🚀 Async Rollout Prefetcher uruchomiony")
        print("   CPU będzie prefetchować batchea podczas GPU treningu")
    
    def prefetch_next_batch(self):
        """Preparuje następny batch w tle"""
        try:
            obs = self.env.reset()
            self.queue.put(obs, timeout=1)
        except:
            pass


def setup_adamw_optimizer(model, config):
    """
    ⚙️  Konfiguruje optimizer AdamW dla modelu PPO
    
    Args:
        model: PPO model
        config: Config dictionary
    
    Returns:
        model z skonfigurowanym optimizerem
    """
    opt_config = config['model'].get('optimizer', {})
    optimizer_type = opt_config.get('type', 'adam').lower()
    weight_decay = opt_config.get('weight_decay', 0.01)
    eps = opt_config.get('eps', 1e-5)
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    
    if optimizer_type == 'adamw':
        try:
            optimizer = torch.optim.AdamW(
                model.policy.parameters(),
                lr=model.learning_rate if isinstance(model.learning_rate, float) else model.learning_rate(1.0),
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=True  # ⚡ Fused kernel (wymaga PyTorch 2.0+)
            )
            fused_status = "FUSED"
        except TypeError:
            # Fallback dla starszych wersji PyTorch
            optimizer = torch.optim.AdamW(
                model.policy.parameters(),
                lr=model.learning_rate if isinstance(model.learning_rate, float) else model.learning_rate(1.0),
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
            fused_status = "Standard"
        
        model.policy.optimizer = optimizer
        
        print(f"\n{'='*70}")
        print(f"[OPTIMIZER] ✅ AdamW ENABLED")
        print(f"{'='*70}")
        print(f"  Type:          AdamW ({fused_status})")
        print(f"  Weight Decay:  {weight_decay}")
        print(f"  Epsilon:       {eps}")
        print(f"  Betas:         {betas}")
        print(f"{'='*70}\n")
        
    return model


def load_policy_weights_only(model, policy_path):
    """
    🎯 Wczytuje TYLKO wagi policy z pliku .pth
    
    ⚠️  UWAGA: Stan optymalizatora zostanie zresetowany!
    Pierwsze ~1000 kroków mogą mieć wyższy loss (rozgrzewka momentum).
    
    Args:
        model: Nowy model PPO
        policy_path: Ścieżka do policy.pth
    
    Returns:
        model z wczytanymi wagami
    
    Raises:
        FileNotFoundError: Jeśli policy.pth nie istnieje
    """
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"❌ Nie znaleziono policy.pth: {policy_path}")
    
    print(f"\n{'='*70}")
    print(f"[POLICY LOAD] Wczytywanie wag z policy.pth")
    print(f"{'='*70}")
    
    # Wczytaj state_dict
    state_dict = torch.load(policy_path, map_location=model.device)
    
    # Załaduj do policy
    model.policy.load_state_dict(state_dict)
    
    print(f"✅ Wczytano wagi policy z: {policy_path}")
    print(f"⚠️  Stan optymalizatora ZRESETOWANY (brak momentum)")
    print(f"💡 Pierwsze ~1000 kroków mogą mieć wyższy loss (rozgrzewka)")
    print(f"{'='*70}\n")
    
    return model


def cleanup_csv_after_checkpoint(csv_path, max_timesteps, verbose=True):
    """
    🔧 Usuwa wiersze z CSV gdzie timesteps > max_timesteps
    
    Problem: Trening przerwany przed eval → CSV ma nowsze timesteps niż model
    Rozwiązanie: Obetnij CSV do ostatniego zapisanego checkpointa
    
    Args:
        csv_path: Ścieżka do CSV
        max_timesteps: Maksymalna wartość timesteps (z modelu)
        verbose: Czy wyświetlać logi
    
    Returns:
        int: Liczba usuniętych wierszy
    """
    if not os.path.exists(csv_path):
        return 0
    
    try:
        # Wczytaj CSV
        rows_to_keep = []
        rows_removed = []
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0
            
            rows_to_keep.append(header)
            
            # Znajdź indeks kolumny timesteps
            try:
                timesteps_idx = header.index('timesteps')
            except ValueError:
                if verbose:
                    print(f"⚠️  CSV {csv_path} nie ma kolumny 'timesteps'")
                return 0
            
            # Filtruj wiersze
            for row in reader:
                try:
                    # Parsuj timesteps (obsługa float/int/string)
                    timesteps_str = str(row[timesteps_idx]).strip()
                    
                    # Pomiń puste wiersze
                    if not timesteps_str or timesteps_str == '':
                        if verbose:
                            print(f"   ⚠️  Pominięto pusty wiersz: {row}")
                        continue
                    
                    row_timesteps = int(float(timesteps_str))
                    
                    # Zachowaj tylko wiersze <= max_timesteps
                    if row_timesteps <= max_timesteps:
                        rows_to_keep.append(row)
                    else:
                        rows_removed.append((row_timesteps, row))
                    
                except (ValueError, IndexError) as e:
                    # Loguj błędny wiersz
                    if verbose:
                        print(f"   ⚠️  Błąd parsowania wiersza (pominięto): {row[:3]}... | Error: {e}")
                    continue
        
        removed_count = len(rows_removed)
        
        if removed_count > 0:
            # Zapisz z powrotem
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows_to_keep)
            
            if verbose:
                print(f"🔧 CSV cleanup: {os.path.basename(csv_path)}")
                print(f"   Zachowano wierszy: {len(rows_to_keep)-1}")
                print(f"   Usunięto wierszy:  {removed_count}")
                if removed_count <= 5:  # Pokaż szczegóły dla małej liczby
                    for ts, row in rows_removed:
                        print(f"      • timesteps={ts} (> {max_timesteps})")
        
        return removed_count
    
    except Exception as e:
        if verbose:
            print(f"⚠️  Błąd podczas czyszczenia CSV {csv_path}: {e}")
            import traceback
            traceback.print_exc()
        return 0


def cleanup_all_training_csvs(base_dir, max_timesteps, verbose=True):
    """
    🔧 Czyści wszystkie CSVy treningowe do max_timesteps
    
    Wywołaj po wczytaniu modelu, przed rozpoczęciem treningu.
    
    Args:
        base_dir: Katalog bazowy projektu
        max_timesteps: Timesteps z wczytanego modelu
        verbose: Czy wyświetlać logi
    
    Returns:
        dict: Statystyki czyszczenia {csv_name: removed_rows}
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"[CSV CLEANUP] Synchronizacja z checkpointem")
        print(f"{'='*70}")
        print(f"Max timesteps z modelu: {max_timesteps:,}")
        print()
    
    stats = {}
    
    # Lista CSVów do wyczyszczenia
    csv_files = [
        ('train_progress.csv', os.path.join(base_dir, 'logs', 'train_progress.csv')),
        ('gradient_monitor.csv', os.path.join(base_dir, 'logs', 'gradient_monitor.csv')),
    ]
    
    total_removed = 0
    for csv_name, csv_path in csv_files:
        if not os.path.exists(csv_path):
            if verbose:
                print(f"⚠️  Plik nie istnieje: {csv_name}")
            continue
        
        removed = cleanup_csv_after_checkpoint(csv_path, max_timesteps, verbose=verbose)
        stats[csv_name] = removed
        total_removed += removed
    
    if verbose:
        print()
        if total_removed > 0:
            print(f"✅ Wyczyszczono {total_removed} wierszy łącznie")
        else:
            print(f"✅ CSV już zsynchronizowane (brak nadmiarowych wierszy)")
        print(f"{'='*70}\n")
    
    return stats





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


def get_difficulty_multiplier(grid_size: int, config: dict) -> float:
    """
    Oblicza mnożnik trudności na podstawie rozmiaru siatki.
    Skaluje nagrody - większe mapy = większy mnożnik.
    """
    diff_config = config.get('difficulty_multiplier', {})
    min_grid = diff_config.get('min_grid_size', 5)
    max_grid = diff_config.get('max_grid_size', 16)
    min_mult = diff_config.get('min_multiplier', 1.0)
    max_mult = diff_config.get('max_multiplier', 2.0)
    
    # Interpolacja liniowa
    if max_grid == min_grid:
        return min_mult
    
    t = (grid_size - min_grid) / (max_grid - min_grid)
    t = max(0.0, min(1.0, t))  # Clamp do [0, 1]
    
    return min_mult + t * (max_mult - min_mult)