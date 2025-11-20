import argparse
import os
import numpy as np
import pygame
import torch
import yaml
import logging
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from model import make_env
from cnn import SolitaireFeaturesExtractor

# Konfiguracja logowania
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Wczytaj konfigurację
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

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

class SolitaireVisualizer:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        
        pygame.init()
        self.width = MARGIN * 2 + 7 * (CARD_WIDTH + CARD_SPACING)
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Solitaire AI")
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()

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

    def draw_game_over(self, text="GAME OVER"):
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Draw text
        text_surf = self.font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=(self.width/2, self.height/2))
        self.screen.blit(text_surf, text_rect)
        
        # Draw instruction
        instr_surf = self.small_font.render("Click or Press Key to Continue", True, WHITE)
        instr_rect = instr_surf.get_rect(center=(self.width/2, self.height/2 + 40))
        self.screen.blit(instr_surf, instr_rect)
        
        pygame.display.flip()

    def run(self, episodes=1):
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            info = {}
            
            while not done:
                # Obsługa zdarzeń (zamknięcie okna)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                # Wybór akcji
                action_masks = self.env.action_masks()
                
                # Check if any move is possible
                if not any(action_masks):
                    print("No valid moves left!")
                    done = True
                    info['reason'] = "No Moves"
                
                if not done:
                    action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                    
                    # Opis akcji dla wizualizacji
                    action_desc = "Unknown"
                    if action == 0: action_desc = "Draw Stock"
                    elif 1 <= action <= 7: action_desc = f"Waste -> Col {action-1}"
                    elif 8 <= action <= 11: action_desc = f"Waste -> Found {action-8}"
                    elif 12 <= action <= 53: 
                        idx = action - 12
                        src = idx // 6
                        dst = idx % 6
                        if dst >= src: dst += 1
                        action_desc = f"Col {src} -> Col {dst}"
                    elif 54 <= action <= 81:
                        idx = action - 54
                        src = idx // 4
                        dst = idx % 4
                        action_desc = f"Col {src} -> Found {dst}"
                    elif 82 <= action <= 85:
                        f_idx = action - 82
                        action_desc = f"Stock -> Found {f_idx}"
                    elif action == 86:
                        action_desc = "Surrender"

                    # Krok środowiska
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    if action == 86:
                        info['reason'] = "Surrendered"
                    
                    done = terminated or truncated
                    total_reward += reward
                    
                    # Render
                    render_info = {
                        'action_desc': action_desc,
                        'reward': reward
                    }
                    self.render(render_info, total_reward)
                    
                    # Pauza żeby zdążyć zobaczyć ruch
                    self.clock.tick(2)  # 2 FPS (klatki na sekundę) - zmień na więcej by przyspieszyć

            print(f"Episode {episode+1} finished. Total Reward: {total_reward}")
            
            # Determine Game Over Message
            if info.get('is_success', False):
                msg = "VICTORY!"
            elif info.get('reason') == "No Moves":
                msg = "NO MOVES LEFT"
            elif info.get('reason') == "Surrendered":
                msg = "SURRENDERED"
            else:
                msg = "GAME OVER"
                
            self.draw_game_over(msg)
            
            # Wait for user input to continue
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False

        pygame.quit()

def mask_fn(env):
    return env.action_masks()

def load_model_interactive(model_path):
    has_full_model = os.path.exists(model_path)
    best_model_path = os.path.join(base_dir, 'models', 'best_model.zip')
    has_best_model = os.path.exists(best_model_path)
    
    print(f"\n{'='*70}")
    print(f"[MODEL SOURCE SELECTION]")
    print(f"{'='*70}")
    
    options = []
    if has_best_model:
        options.append(('1', 'best_model.zip', best_model_path))
        print(f"  [1] 🏆 best_model.zip")
    if has_full_model:
        options.append(('2', 'solitaire_ppo_model.zip', model_path))
        print(f"  [2] 📦 solitaire_ppo_model.zip")
    
    if not options:
        raise FileNotFoundError("Nie znaleziono modeli!")
    
    if len(options) == 1:
        choice = options[0][0]
    else:
        while True:
            choice = input("\nWybierz model [1-{}]: ".format(len(options))).strip()
            if any(choice == opt[0] for opt in options): break
    
    selected = next(opt for opt in options if opt[0] == choice)
    print(f"Ładowanie: {selected[1]}...")
    return MaskablePPO.load(selected[2]), selected[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    default_model_path = os.path.join(base_dir, 'models', 'solitaire_ppo_model.zip')
    
    try:
        model, _ = load_model_interactive(default_model_path)
        
        # Utwórz środowisko
        env = make_env()()
        env = ActionMasker(env, mask_fn)
        
        visualizer = SolitaireVisualizer(env, model)
        visualizer.run(args.episodes)
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
