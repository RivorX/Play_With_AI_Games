import os
import sys
import yaml
import numpy as np
import time
from stable_baselines3 import PPO

# Dodaj katalog scripts do ścieżki
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import make_env, MinesweeperEnv
from cnn import CustomFeaturesExtractor

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def print_board(env: MinesweeperEnv, last_action=None):
    size = env.grid_size
    print(f"\nStep: {env.steps} | Mines: {env.current_mines_count}")
    
    # Header
    print("   " + " ".join([f"{i:2}" for i in range(size)]))
    print("   " + "-" * (size * 3))
    
    for r in range(size):
        line = []
        for c in range(size):
            char = "?"
            # Check last action
            is_last_action = False
            if last_action is not None:
                max_w = env.max_grid_size
                ay, ax = last_action // max_w, last_action % max_w
                if r == ay and c == ax:
                    is_last_action = True
            
            if env.revealed[r, c]:
                if env.board[r, c] == 1:
                    char = "X" # Mine
                elif env.neighbor_counts[r, c] == 0:
                    char = "."
                else:
                    char = str(env.neighbor_counts[r, c])
            else:
                char = "#"
                
            if is_last_action:
                line.append(f"[{char}]") # Highlight last action
            else:
                line.append(f" {char} ")
        print(f"{r:2} " + "".join(line))
    print("\n")

def test_model(episodes=5, delay=1.0):
    model_path = os.path.join(base_dir, config['paths']['model_path'])
    
    if not os.path.exists(model_path):
        print(f"Brak modelu w: {model_path}. Uruchom najpierw train.py")
        return

    print(f"Ładowanie modelu: {model_path}")
    model = PPO.load(model_path)
    
    env_creator = make_env(render_mode="human")
    env = env_creator()
    
    for ep in range(episodes):
        print(f"{'='*30}")
        print(f"EPISODE {ep + 1}")
        print(f"{'='*30}")
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print_board(env)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Decode action for info
            max_w = env.max_grid_size
            y, x = action // max_w, action % max_w
            
            # Check validity just for logging
            valid_str = "VALID"
            if y >= env.grid_size or x >= env.grid_size:
                valid_str = "INVALID (Out of bounds)"
            elif env.revealed[y, x]:
                valid_str = "INVALID (Already revealed)"
                
            print(f"Action: ({x}, {y}) -> {valid_str}")
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Wizualizacja warstw wejściowych dla debugowania
            if ep == 0 and total_reward == reward: # Tylko raz na początku
                 print("\nInput Channels Debug:")
                 print("Fog Channel (part):", obs['image'][0, :5, :5])
                 print("Values Channel (part):", obs['image'][1, :5, :5])
            
            print_board(env, last_action=action)
            print(f"Reward: {reward:.2f} | Total: {total_reward:.2f}")
            
            if delay > 0:
                time.sleep(delay)
                
        result = info.get('result', 'Incomplete')
        print(f"GAME OVER! Result: {result} | Final Reward: {total_reward:.2f}")
        time.sleep(2)

if __name__ == "__main__":
    test_model()
