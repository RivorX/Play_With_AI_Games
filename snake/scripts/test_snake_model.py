import argparse
import os
import time
import numpy as np
import pygame
from stable_baselines3 import PPO
import gymnasium as gym
from model import make_env, set_grid_size
import yaml

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def test_snake_model(model_path, grid_size, episodes):
    # Ustaw grid_size
    set_grid_size(grid_size)
    
    # Utwórz środowisko
    env = make_env(render_mode="human")()
    
    # Załaduj model
    try:
        model = PPO.load(model_path)
        print(f"Załadowano model z: {model_path}")
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return
    
    # Testowanie
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\nEpizod {episode + 1}")
        
        while not done:
            # Oblicz odległość Manhattan od głowy węża do jedzenia
            head = obs[obs[:, :, 0] == 2]  # Głowa węża (wartość 2 w kanale 0)
            if len(head) > 0:
                head_pos = np.where(obs[:, :, 0] == 2)
                head_x, head_y = head_pos[0][0], head_pos[1][0]
                food_pos = np.where(obs[:, :, 0] == 3)  # Jedzenie (wartość 3)
                food_x, food_y = food_pos[0][0], food_pos[1][0]
                distance = abs(head_x - food_x) + abs(head_y - food_y)
            else:
                distance = float('inf')  # W razie błędu (brak głowy)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            
            # Wyświetl krok, wynik, nagrodę i odległość
            print(f"Krok: {steps}, Wynik: {info['score']}, Nagroda: {total_reward}, Odległość: {distance}")
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            pygame.time.wait(50)
        
        print(f"Epizod {episode + 1} zakończony. Wynik: {info['score']}, Całkowita nagroda: {total_reward}, Kroki: {steps}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testowanie modelu Snake PPO")
    parser.add_argument("--model_path", type=str, default=os.path.join(base_dir, config['paths']['model_path']), help="Ścieżka do modelu")
    parser.add_argument("--grid_size", type=int, default=config['environment']['grid_size'], help="Rozmiar siatki")
    parser.add_argument("--episodes", type=int, default=5, help="Liczba epizodów testowych")
    args = parser.parse_args()
    
    test_snake_model(args.model_path, args.grid_size, args.episodes)