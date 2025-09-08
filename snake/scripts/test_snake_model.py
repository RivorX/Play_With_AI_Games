import os
import numpy as np
import pygame
from stable_baselines3 import PPO
from model import make_env
import yaml

# Wczytaj konfigurację
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def test_model(model_path, render_mode="human", episodes=5):
    """
    Testuje zapisany model PPO w środowisku Snake.
    
    Args:
        model_path (str): Ścieżka do pliku modelu (.zip).
        render_mode (str): "human" dla wizualizacji, None bez wizualizacji.
        episodes (int): Liczba epizodów do przetestowania.
    """
    print(f"Ładowanie modelu z: {model_path}")
    env = make_env(render_mode=render_mode)()
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Nie znaleziono modelu w {model_path}. Upewnij się, że plik istnieje.")
        env.close()
        return

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\nEpizod {episode + 1}")
        while not done:
            if render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            action, _ = model.predict(obs, deterministic=True)  # Deterministyczne akcje dla testu
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if render_mode == "human":
                env.render()
                pygame.time.wait(50)  # Płynniejsza wizualizacja
            print(f"Krok: {steps}, Wynik: {info['score']}, Nagroda: {info['total_reward']}")
        print(f"Epizod {episode + 1} zakończony. Łączna nagroda: {total_reward}, Długość węża: {info['score']}")
    env.close()

if __name__ == "__main__":
    # Ścieżka do modelu
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', config['paths']['model_path'])
    # Alternatywnie, użyj najlepszego modelu:
    # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', config['paths']['best_model_path'])
    
    test_model(model_path, render_mode="human", episodes=5)