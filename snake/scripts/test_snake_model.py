import argparse
import os
import numpy as np
import pygame
from sb3_contrib import RecurrentPPO
from model import make_env
import yaml
import logging

# Konfiguracja logowania
base_dir = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
debug_log_path = os.path.join(log_dir, 'debug.log')
# Resetuj plik debug.log na starcie
with open(debug_log_path, 'w', encoding='utf-8'):
    pass

logging.basicConfig(
    filename=debug_log_path,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a'
)

# Wczytaj konfigurację
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def test_snake_model(model_path, grid_size, episodes, policy_pth=None):
    # Utwórz środowisko
    env = make_env(render_mode="human", grid_size=grid_size)()

    # Załaduj model
    import torch
    model = None
    if policy_pth is not None:
        try:
            # Utwórz pusty model z odpowiednią architekturą
            policy_kwargs = config['model']['policy_kwargs'].copy()
            policy_kwargs['features_extractor_class'] = __import__('cnn').CustomFeaturesExtractor
            model = RecurrentPPO(
                config['model']['policy'],
                env,
                learning_rate=0.0001,  # nieistotne przy inferencji
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
            state_dict = torch.load(policy_pth, map_location=config['model']['device'])
            model.policy.load_state_dict(state_dict)
            logging.info(f"Załadowano policy.pth z: {policy_pth}")
        except Exception as e:
            logging.error(f"Błąd podczas ładowania policy.pth: {e}")
            return
    else:
        try:
            model = RecurrentPPO.load(model_path)
            logging.info(f"Załadowano model z: {model_path}")
        except Exception as e:
            logging.error(f"Błąd podczas ładowania modelu: {e}")
            return

    # Testowanie
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        logging.info(f"\nEpizod {episode + 1}")

        while not done:
            # Obraz z viewportu
            mapa = obs['image'][:, :, 0]  # [H, W], pojedynczy kanał

            # Wyodrębnij skalary
            direction = obs['direction']
            dx_head = obs['dx_head'][0]
            dy_head = obs['dy_head'][0]

            # Znajdź pozycję głowy i jedzenia w mapie
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

            # Oblicz odległość Manhattan
            if head_x >= 0 and food_x >= 0:
                distance = abs(head_x - food_x) + abs(head_y - food_y)
            else:
                distance = float('inf')

            # Zapisz informacje debugowania do pliku
            logging.info("--- Obserwacja (fragment) ---")
            logging.info(f"Kanał mapa:\n{np.array_str(mapa, precision=2, suppress_small=True, max_line_width=120)}")
            logging.info(f"Direction: {direction}")
            logging.info(f"dx_head: {dx_head}")
            logging.info(f"dy_head: {dy_head}")
            logging.info(f"Pozycja głowy: ({head_x}, {head_y}) | Pozycja jedzenia: ({food_x}, {food_y})")
            logging.info(f"Dystans Manhattan: {distance}")
            logging.info(f"Stan gry: done={done}, steps={steps}, snake={env.snake}, food={env.food}")
            logging.info("-" * 60)

            # Wykonaj akcję
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            logging.info(f"Krok: {steps}, Wynik: {info['score']}, Nagroda: {total_reward}, Akcja: {action}")
            logging.info("-" * 60)

            # Obsługa zdarzeń Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            pygame.time.wait(50)

        logging.info(f"Epizod {episode + 1} zakończony. Wynik: {info['score']}, Całkowita nagroda: {total_reward}, Kroki: {steps}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testowanie modelu Snake PPO")
    parser.add_argument("--model_path", type=str, default=os.path.join(base_dir, config['paths']['models_dir'], 'best_model.zip'), help="Ścieżka do modelu (zip)")
    parser.add_argument("--policy_pth", type=str, default=None, help="Ścieżka do policy.pth (opcjonalnie) ./snake/models/policy.pth")
    parser.add_argument("--grid_size", type=int, default=8, help="Rozmiar siatki")
    parser.add_argument("--episodes", type=int, default=1, help="Liczba epizodów testowych")
    args = parser.parse_args()

    test_snake_model(args.model_path, args.grid_size, args.episodes, policy_pth=args.policy_pth)