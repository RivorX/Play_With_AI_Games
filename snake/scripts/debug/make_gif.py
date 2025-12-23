import os
import argparse
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import pygame

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model import make_env
from utils.test_utils import select_visual_style, select_grid_size_interactive, load_model_interactive


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def capture_run_as_gif(model, grid_size, episodes, out_path, visual_style='classic', fps=10, max_frames=10000, death_pause_duration=3.0):
    """
    Nagrywa gameplay jako GIF z ulepszonymi funkcjami:
    - Automatyczne czyszczenie folderu tymczasowego
    - Zapętlanie GIFa (loop=0)
    - Pauza 3s na ostatniej klatce gdy wąż zginie
    
    Args:
        model: Załadowany model PPO
        grid_size: Rozmiar siatki
        episodes: Liczba epizodów
        out_path: Ścieżka wyjścia
        visual_style: Styl wizualny ('classic', 'modern', 'realistic')
        fps: Liczba klatek na sekundę
        max_frames: Maksymalna liczba klatek
        death_pause_duration: Czas (w sekundach) pauzy na ostatniej klatce gdy wąż zginie
    """
    # Przygotuj env
    env = make_env(render_mode="human", grid_size=grid_size, visual_style=visual_style)()

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
    parser.add_argument('--style', type=str, default=None, choices=['classic', 'modern', 'realistic'], 
                       help='Visual style (if not provided, will prompt)')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--death_pause', type=float, default=3.0,
                        help='Duration (seconds) to pause on last frame if snake dies (default: 3.0)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    default_model_path = base_dir / 'models' / 'snake_ppo_model.zip'
    policy_path = base_dir / 'models' / 'policy.pth'
    
    # 🎯 Interaktywny wybór modelu
    model, source_name = load_model_interactive(default_model_path, policy_path, base_dir)
    
    # 🎨 Interaktywny wybór stylu wizualnego
    if args.style is None:
        visual_style = select_visual_style()
    else:
        visual_style = args.style
        print(f"\n✅ Używam stylu z argumentu: {visual_style}\n")
    
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
    print(f"   Visual Style: {visual_style}")
    print(f"   Death pause: {args.death_pause}s")
    print(f"   Output: {out_path}")
    print()
    
    gif = capture_run_as_gif(
        model,
        grid_size, 
        args.episodes, 
        out_path, 
        visual_style=visual_style,
        fps=args.fps,
        death_pause_duration=args.death_pause
    )
    
    print(f"\n🎉 Gotowe! GIF zapisany w: {gif}")