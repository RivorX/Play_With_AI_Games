import os
import argparse
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import pygame
import yaml

import sys
# Add scripts directory to path to import local modules
# Current file is in scripts/debug, so we need parent directory (scripts)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import make_env
from stable_baselines3 import PPO

# Load config to get defaults
# base_dir is 2 levels up from scripts (scripts/debug -> scripts -> root)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_model_interactive(base_dir):
    """Simple interactive model loader for Minesweeper"""
    models_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(models_dir, 'minesweeper_ppo.zip')
    
    if not os.path.exists(model_path):
        print(f"❌ Nie znaleziono modelu głównego: {model_path}")
        # Look for checkpoints
        checkpoints = [f for f in os.listdir(models_dir) if f.endswith('.zip') and 'minesweeper' in f]
        if not checkpoints:
            raise FileNotFoundError("Brak dostępnych modeli!")
            
        print("Dostępne checkpointy:")
        for i, ckpt in enumerate(checkpoints):
            print(f"{i+1}: {ckpt}")
            
        choice = input("Wybierz numer modelu: ")
        try:
            model_path = os.path.join(models_dir, checkpoints[int(choice)-1])
        except:
            print("Nieprawidłowy wybór, używam pierwszego.")
            model_path = os.path.join(models_dir, checkpoints[0])
            
    print(f"📂 Ładowanie modelu: {os.path.basename(model_path)}")
    model = PPO.load(model_path)
    return model, os.path.basename(model_path)

def capture_run_as_gif(model, episodes, out_path, fps=4, max_frames=1000, death_pause_duration=2.0):
    """
    Nagrywa gameplay Sapera jako GIF.
    """
    # Przygotuj env
    # Render mode 'human' allows us to grab the Pygame surface from the env wrapper
    env = make_env(render_mode="human")()

    # Folder na klatki
    frames_dir = os.path.join(base_dir, 'logs', 'gif_frames')
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    frame_files = []
    last_frame_is_end = False
    
    # Track win/loss
    results = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            
            # Initialize rendering
            env.render()

            while not done and len(frame_files) < max_frames:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                # Render step
                env.render()
                
                # Capture frame from env.screen
                if env.screen is None:
                    continue
                    
                surf = env.screen
                arr = pygame.surfarray.array3d(surf)
                # array3d returns (W,H,3) -> transpose to (H,W,3)
                arr = np.transpose(arr, (1, 0, 2))

                # Save frame
                fname = os.path.join(frames_dir, f"frame_{len(frame_files):05d}.png")
                imageio.imwrite(fname, arr)
                frame_files.append(fname)

                if done:
                    last_frame_is_end = True
                    res = info.get('result', 'unknown')
                    results.append(res)
                    print(f"Epizod {ep+1} zakończony: {res}")

                steps += 1
                pygame.time.wait(int(1000 / fps))

            # Pause between episodes
            time.sleep(0.5)

    finally:
        env.close()

    if not frame_files:
        raise RuntimeError("No frames captured.")

    # Create GIF
    print(f"📸 Przetwarzanie {len(frame_files)} klatek...")
    images = []
    import imageio.v2 as iio
    
    for f in frame_files:
        images.append(iio.imread(f))
    
    # Add pause at the very end
    if last_frame_is_end:
        pause_frames = int(fps * death_pause_duration)
        last_frame = images[-1]
        for _ in range(pause_frames):
            images.append(last_frame)
    
    print(f"💾 Zapisywanie GIF: {out_path}")
    iio.mimsave(out_path, images, fps=fps, loop=0)
    
    # Cleanup
    shutil.rmtree(frames_dir)
    
    print(f"✅ GIF gotowy!")
    print(f"   Epizody: {episodes}")
    print(f"   Wyniki: {results}")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nagrywanie rozgrywki Minesweeper do GIF')
    parser.add_argument('--episodes', type=int, default=3, help='Liczba epizodów')
    parser.add_argument('--out', type=str, default=None, help='Ścieżka wyjściowa GIF')
    parser.add_argument('--fps', type=int, default=5, help='Klatki na sekundę')
    
    args = parser.parse_args()

    # Load model
    model, model_name = load_model_interactive(base_dir)
    
    # Determine output path
    if args.out is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(base_dir, 'logs', f'minesweeper_run_{timestamp}.gif')
    else:
        out_path = args.out

    capture_run_as_gif(
        model,
        episodes=args.episodes, 
        out_path=out_path, 
        fps=args.fps
    )
