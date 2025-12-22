import os
import argparse
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import pygame
import torch
from stable_baselines3 import PPO
import yaml

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model import make_env
from cnn import CustomFeaturesExtractor


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def select_grid_size_interactive(default_size=8):
    """
    🎯 Interaktywny wybór rozmiaru siatki
    
    Returns:
        int: Wybrany rozmiar siatki
    """
    print(f"\n{'='*70}")
    print(f"[GRID SIZE SELECTION]")
    print(f"{'='*70}")
    print(f"  [1] 🟩 8x8   (Easy - Mała siatka)")
    print(f"  [2] 🟦 12x12 (Medium)")
    print(f"  [3] 🟨 16x16 (Hard - Duża siatka)")
    print(f"  [4] 🟪 Custom (Własny rozmiar)")
    print(f"{'='*70}")
    
    while True:
        choice = input(f"\nWybierz rozmiar siatki [1-4] (default: {default_size}x{default_size}): ").strip()
        
        if choice == '' or choice == '0':
            print(f"✅ Używam domyślnego: {default_size}x{default_size}\n")
            return default_size
        elif choice == '1':
            return 8
        elif choice == '2':
            return 12
        elif choice == '3':
            return 16
        elif choice == '4':
            while True:
                try:
                    custom = input("Podaj rozmiar siatki (4-32): ").strip()
                    custom_size = int(custom)
                    if 4 <= custom_size <= 32:
                        return custom_size
                    else:
                        print("❌ Rozmiar musi być między 4 a 32.")
                except ValueError:
                    print("❌ Nieprawidłowa wartość. Podaj liczbę.")
        else:
            print("❌ Nieprawidłowy wybór. Wybierz 1-4 lub Enter dla domyślnego.")


def load_model_interactive(model_path, policy_path, base_dir):
    """
    🎯 Interaktywny wybór źródła modelu
    
    Returns:
        tuple: (model, source_name)
    """
    has_full_model = os.path.exists(model_path)
    has_best_model = os.path.exists(base_dir / 'models' / 'best_model.zip')
    has_policy = os.path.exists(policy_path)
    
    print(f"\n{'='*70}")
    print(f"[MODEL SOURCE SELECTION]")
    print(f"{'='*70}")
    
    options = []
    
    if has_best_model:
        options.append(('1', 'best_model.zip', base_dir / 'models' / 'best_model.zip'))
        print(f"  [1] 🏆 best_model.zip (najlepszy model z treningu)")
    
    if has_full_model and str(model_path) != str(base_dir / 'models' / 'best_model.zip'):
        options.append(('2', 'snake_ppo_model.zip', model_path))
        print(f"  [2] 📦 snake_ppo_model.zip (ostatni checkpoint)")
    
    if has_policy:
        key = str(len(options) + 1)
        options.append((key, 'policy.pth', policy_path))
        print(f"  [{key}] 🎯 policy.pth (tylko wagi sieci)")
    
    print(f"{'='*70}")
    
    if not options:
        raise FileNotFoundError("Nie znaleziono żadnego modelu! Sprawdź folder models/")
    
    if len(options) == 1:
        choice = options[0][0]
        print(f"\n✅ Automatycznie wybrany: {options[0][1]}\n")
    else:
        while True:
            choice = input("\nWybierz źródło modelu [1-{}]: ".format(len(options))).strip()
            if any(choice == opt[0] for opt in options):
                break
            print(f"❌ Nieprawidłowy wybór. Wybierz 1-{len(options)}.")
    
    selected = next(opt for opt in options if opt[0] == choice)
    source_name = selected[1]
    source_path = selected[2]
    
    print(f"\n🎬 Ładowanie: {source_name}...")
    
    # Załaduj model
    if source_name == 'policy.pth':
        # Wczytaj config
        config_path = base_dir / 'config' / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Stwórz env do sprawdzenia observation_space
        temp_env = make_env(render_mode=None, grid_size=8)()
        
        # Stwórz pusty model
        policy_kwargs = config['model']['policy_kwargs'].copy()
        policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor
        
        model = PPO(
            config['model']['policy'],
            temp_env,
            learning_rate=0.0001,
            n_steps=config['model']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=config['model']['device']
        )
        
        # Załaduj wagi
        state_dict = torch.load(source_path, map_location=config['model']['device'])
        model.policy.load_state_dict(state_dict)
        
        temp_env.close()
        print(f"✅ Załadowano policy.pth\n")
    else:
        model = PPO.load(source_path)
        print(f"✅ Załadowano {source_name}\n")
    
    return model, source_name


def capture_run_as_gif(model, grid_size, episodes, out_path, fps=10, max_frames=10000, death_pause_duration=3.0):
    """
    Nagrywa gameplay jako GIF z ulepszonymi funkcjami:
    - Automatyczne czyszczenie folderu tymczasowego
    - Zapętlanie GIFa (loop=0)
    - Pauza 3s na ostatniej klatce gdy wąż zginie
    
    Args:
        model: Załadowany model PPO
        death_pause_duration: Czas (w sekundach) pauzy na ostatniej klatce gdy wąż zginie
    """
    # Przygotuj env
    env = make_env(render_mode="human", grid_size=grid_size)()

    # Folder na klatki
    base_dir = Path(__file__).resolve().parents[2]
    frames_dir = base_dir / 'logs' / 'gif_frames'
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    frame_files = []
    last_frame_is_death = False

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            
            # Initialize rendering before first frame
            env.render()

            while not done and len(frame_files) < max_frames:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                # render to screen and capture surface
                env.render()
                
                # Get screen surface - env.screen is set in _render_frame()
                surf = env.screen
                arr = pygame.surfarray.array3d(surf)
                # array3d returns (W,H,3) with x horizontal; convert to HxW and flip axes
                arr = np.transpose(arr, (1, 0, 2))

                # Save frame
                fname = frames_dir / f"frame_{len(frame_files):05d}.png"
                imageio.imwrite(fname, arr)
                frame_files.append(str(fname))

                # ✅ Sprawdź czy to był ostatni frame (śmierć)
                if done:
                    last_frame_is_death = True

                steps += 1
                # small delay so rendering has time
                pygame.time.wait(int(1000 / fps))

            # small pause between episodes
            time.sleep(0.2)

    finally:
        env.close()

    if not frame_files:
        raise RuntimeError("No frames captured; check that the environment renders when render_mode='human'.")

    # ✅ Załaduj wszystkie klatki
    print(f"📸 Załadowano {len(frame_files)} klatek...")
    images = []
    
    # Użyj imageio.v2 żeby pozbyć się deprecation warning
    import imageio.v2 as iio
    
    for f in frame_files:
        images.append(iio.imread(f))
    
    # ✅ Jeśli wąż zginął na końcu, dodaj duplikaty ostatniej klatki dla 3s pauzy
    if last_frame_is_death:
        death_pause_frames = int(fps * death_pause_duration)
        print(f"💀 Wąż zginął! Dodaję {death_pause_frames} klatek pauzy ({death_pause_duration}s przy {fps} FPS)...")
        last_frame = images[-1]
        for _ in range(death_pause_frames):
            images.append(last_frame)
    
    # ✅ Zapisz GIF z zapętleniem (loop=0 = infinite loop)
    print(f"💾 Zapisuję GIF z zapętleniem (loop=0)...")
    iio.mimsave(out_path, images, fps=fps, loop=0)
    
    # ✅ Usuń folder tymczasowy
    print(f"🧹 Czyszczę folder tymczasowy: {frames_dir}")
    shutil.rmtree(frames_dir)
    
    print(f"✅ GIF zapisany: {out_path}")
    print(f"   - Liczba klatek: {len(images)}")
    print(f"   - FPS: {fps}")
    print(f"   - Zapętlony: TAK (nieskończona pętla)")
    if last_frame_is_death:
        print(f"   - Pauza na śmierć: {death_pause_duration}s")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture Snake model run as GIF')
    parser.add_argument('--grid_size', type=int, default=None, help='Grid size (if not provided, will prompt)')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--out', type=str, default=None, help='Output GIF path')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--death_pause', type=float, default=3.0,
                        help='Duration (seconds) to pause on last frame if snake dies (default: 3.0)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    default_model_path = base_dir / 'models' / 'snake_ppo_model.zip'
    policy_path = base_dir / 'models' / 'policy.pth'
    
    # 🎯 Interaktywny wybór modelu
    model, source_name = load_model_interactive(default_model_path, policy_path, base_dir)
    
    # 🎯 Interaktywny wybór rozmiaru siatki
    if args.grid_size is None:
        grid_size = select_grid_size_interactive(default_size=8)
    else:
        grid_size = args.grid_size
        print(f"\n✅ Używam rozmiaru z argumentu: {grid_size}x{grid_size}\n")

    # ✅ Automatyczna nazwa z rozmiarem siatki (bez tagu źródła)
    if args.out is None:
        out_path = base_dir / 'logs' / f'snake_run_{grid_size}.gif'
    else:
        out_path = Path(args.out)

    out_path = str(out_path)

    print(f"🎬 Rozpoczynam nagrywanie GIF...")
    print(f"   Source: {source_name}")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Episodes: {args.episodes}")
    print(f"   FPS: {args.fps}")
    print(f"   Death pause: {args.death_pause}s")
    print(f"   Output: {out_path}")
    print()
    
    gif = capture_run_as_gif(
        model,
        grid_size, 
        args.episodes, 
        out_path, 
        fps=args.fps,
        death_pause_duration=args.death_pause
    )
    
    print(f"\n🎉 Gotowe! GIF zapisany w: {gif}")