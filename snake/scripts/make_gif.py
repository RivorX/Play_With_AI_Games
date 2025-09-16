import os
import argparse
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import pygame
from stable_baselines3 import PPO

from model import make_env, set_grid_size


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def capture_run_as_gif(model_path, grid_size, episodes, out_path, fps=10, max_frames=1000):
    # Przygotuj env
    set_grid_size(grid_size)
    env = make_env(render_mode="human", grid_size=grid_size)()

    # Załaduj model
    model = PPO.load(model_path)

    # Folder na klatki
    frames_dir = Path(env.__module__).parent / '..' / 'logs' / 'gif_frames'
    frames_dir = frames_dir.resolve()
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    frame_files = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0

            while not done and len(frame_files) < max_frames:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                # render to screen and capture surface
                env.render()
                # pygame surface -> array
                surf = env.env.screen if hasattr(env, 'env') and getattr(env.env, 'screen', None) is not None else env.screen
                arr = pygame.surfarray.array3d(surf)
                # array3d returns (W,H,3) with x horizontal; convert to HxW and flip axes
                arr = np.transpose(arr, (1, 0, 2))

                # Save frame
                fname = frames_dir / f"frame_{len(frame_files):05d}.png"
                imageio.imwrite(fname, arr)
                frame_files.append(str(fname))

                steps += 1
                # small delay so rendering has time (and GIF isn't too fast)
                pygame.time.wait(int(1000 / fps))

            # small pause between episodes
            time.sleep(0.2)

    finally:
        env.close()

    if not frame_files:
        raise RuntimeError("No frames captured; check that the environment renders when render_mode='human'.")

    # Compose GIF
    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(out_path, images, fps=fps)

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture Snake model run as GIF')
    parser.add_argument('--model_path', type=str, required=False,
                        help='Path to RL model (zip/pkl). If omitted uses default best_model.zip in models dir')
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--out', type=str, default=None, help='Output GIF path')
    parser.add_argument('--fps', type=int, default=8)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    if args.model_path is None:
        # try to find models dir via config
        default_model = base_dir / 'models' / 'best_model.zip'
        model_path = str(default_model)
    else:
        model_path = args.model_path

    if args.out is None:
        out_path = base_dir / 'logs' / 'snake_run.gif'
    else:
        out_path = Path(args.out)

    out_path = str(out_path)

    print(f"Loading model: {model_path}\nGrid: {args.grid_size} | Episodes: {args.episodes} | FPS: {args.fps}\nOutput: {out_path}")
    gif = capture_run_as_gif(model_path, args.grid_size, args.episodes, out_path, fps=args.fps)
    print(f"Saved GIF to: {gif}")
