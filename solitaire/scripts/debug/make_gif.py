import os
import argparse
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import pygame
import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model import make_env
from cnn import SolitaireFeaturesExtractor

# Kolory
WHITE = (255, 255, 255)
GREEN = (0, 100, 0)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 200)

# Wymiary kart
CARD_WIDTH = 80
CARD_HEIGHT = 120
CARD_SPACING = 20
MARGIN = 20

class SolitaireRenderer:
    def __init__(self, env):
        self.env = env
        
        pygame.init()
        self.width = MARGIN * 2 + 7 * (CARD_WIDTH + CARD_SPACING)
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Solitaire GIF Recorder")
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)

    def draw_card(self, x, y, card, hidden=False):
        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        
        if hidden:
            pygame.draw.rect(self.screen, BLUE, rect)
            pygame.draw.rect(self.screen, WHITE, rect, 2)
            # Wzorek na tyle karty
            pygame.draw.line(self.screen, WHITE, (x, y), (x+CARD_WIDTH, y+CARD_HEIGHT))
            pygame.draw.line(self.screen, WHITE, (x+CARD_WIDTH, y), (x, y+CARD_HEIGHT))
        else:
            pygame.draw.rect(self.screen, WHITE, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)
            
            # Kolor tekstu
            color = RED if card.color() == 0 else BLACK
            
            # Ranga i Kolor
            suits = ['♥', '♦', '♣', '♠']
            ranks = ['_', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            
            rank_str = ranks[card.rank]
            suit_str = suits[card.suit]
            
            text = self.font.render(f"{rank_str}{suit_str}", True, color)
            self.screen.blit(text, (x + 5, y + 5))
            
            # Duży symbol na środku
            big_suit = self.font.render(suit_str, True, color)
            self.screen.blit(big_suit, (x + CARD_WIDTH//2 - 10, y + CARD_HEIGHT//2 - 10))

    def draw_placeholder(self, x, y, label=""):
        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        pygame.draw.rect(self.screen, GREEN, rect)
        pygame.draw.rect(self.screen, (0, 150, 0), rect, 2)
        if label:
            text = self.small_font.render(label, True, (0, 150, 0))
            self.screen.blit(text, (x + 10, y + CARD_HEIGHT//2))

    def render(self, info=None, total_reward=0.0):
        self.screen.fill(GREEN)
        
        # Używamy unwrapped, aby uniknąć ostrzeżeń i dostać się do zmiennych środowiska
        env = self.env.unwrapped
        
        # --- GÓRNY RZĄD ---
        
        # Stock (Talia)
        stock_x = MARGIN
        stock_y = MARGIN
        if env.stock:
            # Rysuj kilka kart żeby wyglądało na stos
            for i in range(min(3, len(env.stock))):
                self.draw_card(stock_x + i*2, stock_y + i*2, None, hidden=True)
            count_text = self.small_font.render(f"{len(env.stock)}", True, WHITE)
            self.screen.blit(count_text, (stock_x, stock_y - 20))
        else:
            self.draw_placeholder(stock_x, stock_y, "Empty")

        # Waste (Odkryte)
        waste_x = MARGIN + CARD_WIDTH + CARD_SPACING
        waste_y = MARGIN
        if env.waste:
            self.draw_card(waste_x, waste_y, env.waste[-1])
        else:
            self.draw_placeholder(waste_x, waste_y)

        # Foundations (Stosy bazowe)
        foundations_start_x = self.width - MARGIN - 4 * (CARD_WIDTH + CARD_SPACING)
        suits = ['♥', '♦', '♣', '♠']
        for i in range(4):
            fx = foundations_start_x + i * (CARD_WIDTH + CARD_SPACING)
            rank = int(env.foundations[i])
            
            if rank > 0:
                # Tworzymy tymczasową kartę do wyświetlenia
                # Suit i odpowiada i-temu stosowi
                dummy_card = type('obj', (object,), {'rank': rank, 'suit': i, 'color': lambda: 0 if i < 2 else 1})
                self.draw_card(fx, waste_y, dummy_card)
            else:
                self.draw_placeholder(fx, waste_y, suits[i])

        # --- TABLEAU (KOLUMNY) ---
        tableau_y = MARGIN + CARD_HEIGHT + MARGIN
        for i in range(7):
            tx = MARGIN + i * (CARD_WIDTH + CARD_SPACING)
            pile = env.tableau[i]
            
            if not pile:
                self.draw_placeholder(tx, tableau_y)
            else:
                for j, card in enumerate(pile):
                    cy = tableau_y + j * 25  # Przesunięcie w pionie
                    self.draw_card(tx, cy, card, hidden=not card.face_up)

        # --- INFO ---
        if info:
            score_text = self.font.render(f"Steps: {env.steps}", True, WHITE)
            self.screen.blit(score_text, (MARGIN, self.height - 40))
            
            if info.get('action_desc'):
                action_text = self.font.render(f"Action: {info['action_desc']}", True, WHITE)
                self.screen.blit(action_text, (200, self.height - 40))
                
            if info.get('reward') is not None:
                rew_text = self.font.render(f"Reward: {info['reward']:.1f}", True, WHITE)
                self.screen.blit(rew_text, (self.width - 250, self.height - 40))
                
            # Total Reward (Prawy dolny róg)
            total_text = self.font.render(f"Total: {total_reward:.1f}", True, WHITE)
            self.screen.blit(total_text, (self.width - 250, self.height - 70))

        pygame.display.flip()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model_interactive(model_path, base_dir):
    """
    🎯 Interaktywny wybór źródła modelu
    
    Returns:
        tuple: (model, source_name)
    """
    has_full_model = os.path.exists(model_path)
    best_model_path = os.path.join(base_dir, 'models', 'best_model.zip')
    has_best_model = os.path.exists(best_model_path)
    
    print(f"\n{'='*70}")
    print(f"[MODEL SOURCE SELECTION]")
    print(f"{'='*70}")
    
    options = []
    
    if has_best_model:
        options.append(('1', 'best_model.zip', best_model_path))
        print(f"  [1] 🏆 best_model.zip (najlepszy model z treningu)")
    
    if has_full_model:
        options.append(('2', 'solitaire_ppo_model.zip', model_path))
        print(f"  [2] 📦 solitaire_ppo_model.zip (ostatni checkpoint)")
    
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
    model = MaskablePPO.load(source_path)
    print(f"✅ Załadowano {source_name}\n")
    
    return model, source_name


def mask_fn(env):
    return env.action_masks()


def get_action_description(action):
    """Konwertuj action ID na opis"""
    if action == 0:
        return "Draw Stock"
    elif 1 <= action <= 7:
        return f"Waste → Col {action-1}"
    elif 8 <= action <= 11:
        return f"Waste → Found {action-8}"
    elif 12 <= action <= 53:
        idx = action - 12
        src = idx // 6
        dst = idx % 6
        if dst >= src:
            dst += 1
        return f"Col {src} → Col {dst}"
    elif 54 <= action <= 81:
        idx = action - 54
        src = idx // 4
        dst = idx % 4
        return f"Col {src} → Found {dst}"
    elif 82 <= action <= 85:
        f_idx = action - 82
        return f"Stock → Found {f_idx}"
    elif action == 86:
        return "Surrender"
    else:
        return "Unknown"


def capture_run_as_gif(model, episodes, out_path, fps=2, max_frames=5000, death_pause_duration=3.0):
    """
    Nagrywa gameplay jako GIF.
    
    Args:
        model: Załadowany model MaskablePPO
        episodes: Liczba epizodów do nagrania
        out_path: Ścieżka do pliku GIF
        fps: Klatki na sekundę
        max_frames: Maksymalna liczba klatek
        death_pause_duration: Czas (w sekundach) pauzy na ostatniej klatce gdy gra się skończy
    """
    # Przygotuj env
    def make_masked_env():
        env = make_env()()
        env = ActionMasker(env, mask_fn)
        return env
    
    env = make_masked_env()
    
    # Initialize renderer
    renderer = SolitaireRenderer(env)

    # Folder na klatki
    base_dir = Path(__file__).resolve().parents[2]
    frames_dir = base_dir / 'logs' / 'gif_frames'
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    frame_files = []
    last_frame_is_game_end = False

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            total_reward = 0.0
            
            # Initial render
            renderer.render(info={'action_desc': 'Start'}, total_reward=0.0)

            while not done and len(frame_files) < max_frames:
                # Process pygame events to keep window responsive
                pygame.event.pump()

                # Predict action
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                
                # Get action description
                action_desc = get_action_description(action)
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Add action description to info for renderer
                render_info = info.copy()
                render_info['action_desc'] = action_desc
                render_info['reward'] = reward

                # Render using our custom renderer
                renderer.render(render_info, total_reward)
                
                # Get screen surface from renderer
                surf = renderer.screen
                arr = pygame.surfarray.array3d(surf)
                # array3d returns (W,H,3) z x poziomem; konwertuj na HxW
                arr = np.transpose(arr, (1, 0, 2))

                # Save frame
                fname = frames_dir / f"frame_{len(frame_files):05d}.png"
                imageio.imwrite(fname, arr)
                frame_files.append(str(fname))

                # Sprawdź czy to był ostatni frame (koniec gry)
                if done:
                    last_frame_is_game_end = True

                steps += 1
                # small delay so rendering has time
                pygame.time.wait(int(1000 / fps))

            print(f"✅ Epizod {ep+1}/{episodes} - {steps} kroków, {len(frame_files)} klatek zaś")
            # small pause between episodes
            time.sleep(0.2)

    finally:
        env.close()
        pygame.quit()

    if not frame_files:
        raise RuntimeError("Nie zapisano żadnych klatek; sprawdź czy środowisko renderuje się.")

    # Załaduj wszystkie klatki
    print(f"\n📸 Załadowano {len(frame_files)} klatek...")
    images = []
    
    # Użyj imageio.v2 żeby pozbyć się deprecation warning
    import imageio.v2 as iio
    
    for f in frame_files:
        images.append(iio.imread(f))
    
    # Jeśli gra się skończyła, dodaj duplikaty ostatniej klatki dla pauzy
    if last_frame_is_game_end:
        death_pause_frames = int(fps * death_pause_duration)
        print(f"🎬 Koniec gry! Dodaję {death_pause_frames} klatek pauzy ({death_pause_duration}s przy {fps} FPS)...")
        last_frame = images[-1]
        for _ in range(death_pause_frames):
            images.append(last_frame)
    
    # Zapisz GIF z zapętleniem (loop=0 = infinite loop)
    print(f"\n💾 Zapisuję GIF z zapętleniem (loop=0)...")
    iio.mimsave(out_path, images, fps=fps, loop=0)
    
    # Usuń folder tymczasowy
    print(f"🧹 Czyszczę folder tymczasowy: {frames_dir}")
    shutil.rmtree(frames_dir)
    
    print(f"\n✅ GIF zapisany: {out_path}")
    print(f"   - Liczba klatek: {len(images)}")
    print(f"   - FPS: {fps}")
    print(f"   - Zapętlony: TAK (nieskończona pętla)")
    if last_frame_is_game_end:
        print(f"   - Pauza na koniec: {death_pause_duration}s")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture Solitaire model run as GIF')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--out', type=str, default=None, help='Output GIF path')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second (default: 2)')
    parser.add_argument('--pause', type=float, default=3.0,
                        help='Duration (seconds) to pause on last frame when game ends (default: 3.0)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    default_model_path = os.path.join(base_dir, 'models', 'solitaire_ppo_model.zip')
    
    # Interaktywny wybór modelu
    model, source_name = load_model_interactive(default_model_path, base_dir)
    
    # Automatyczna nazwa
    if args.out is None:
        out_path = base_dir / 'logs' / 'solitaire_run.gif'
    else:
        out_path = Path(args.out)

    out_path = str(out_path)

    print(f"🎬 Rozpoczynam nagrywanie GIF...")
    print(f"   Source: {source_name}")
    print(f"   Episodes: {args.episodes}")
    print(f"   FPS: {args.fps}")
    print(f"   Pause: {args.pause}s")
    print(f"   Output: {out_path}")
    print()
    
    gif = capture_run_as_gif(
        model,
        args.episodes, 
        out_path, 
        fps=args.fps,
        death_pause_duration=args.pause
    )
    
    print(f"\n🎉 Gotowe! GIF zapisany w: {gif}")
