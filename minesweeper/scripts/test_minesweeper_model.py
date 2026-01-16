import os
import sys
import yaml
import numpy as np
import time
import pygame
from sb3_contrib import MaskablePPO

# Dodaj katalog scripts do ścieżki
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import make_env, MinesweeperEnv
from cnn import CustomFeaturesExtractor

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def test_model(episodes=10, delay=0.1):
    model_path = os.path.join(base_dir, config['paths']['model_path'])
    
    if not os.path.exists(model_path):
        print(f"Brak modelu w: {model_path}. Uruchom najpierw train.py")
        return

    print(f"Ładowanie modelu: {model_path}")
    model = MaskablePPO.load(model_path)
    
    # Inicjalizacja Pygame
    pygame.init()
    
    # Użycie wbudowanego renderowania z visual styles ('human' mode w env)
    # Env sam obsłuży okno
    env_creator = make_env(render_mode="human")
    env = env_creator()
    
    print("\n[STEROWANIE]")
    print(" SPACE - Pauza / Wznowienie")
    print(" S     - Krok po kroku (gdy zapauzowane)")
    print(" ESC   - Wyjście")
    print(f"{'='*30}\n")
    
    try:
        for ep in range(episodes):
            print(f"{'='*30}")
            print(f"EPISODE {ep + 1}")
            print(f"{'='*30}")
            
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            paused = False
            
            # Initial render
            env.render()
            
            while not done:
                # Event handling
                step_once = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print("PAUSED" if paused else "RESUMED")
                        elif event.key == pygame.K_s and paused:
                            step_once = True
                
                if paused and not step_once:
                    env.render()
                    time.sleep(0.1)
                    continue
                
                # AI Action
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                
                # Execute
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Logging
                max_w = env.max_grid_size
                is_flag = action >= env.total_cells
                if is_flag:
                    real_act = action - env.total_cells
                    y, x = real_act // max_w, real_act % max_w
                    act_str = f"FLAG ({x}, {y})"
                else:
                    y, x = action // max_w, action % max_w
                    act_str = f"REVEAL ({x}, {y})"
                   
                print(f"Step {steps:3}: {act_str} | Rew: {reward:5.2f} | Tot: {total_reward:6.2f}")
                
                env.render()
                
                if delay > 0:
                    time.sleep(delay)
                    
                if done or truncated:
                    result = info.get('result', 'Incomplete')
                    print(f"GAME OVER! Result: {result} | Final Reward: {total_reward:.2f}")
                    time.sleep(1.0) # Chwila na zobaczenie wyniku
    except KeyboardInterrupt:
        print("\nPrzerwano.")
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    test_model()
