"""
GUI helper functions for chess interface
"""

import pygame
import torch
from pathlib import Path
from datetime import datetime


def create_piece_surfaces(square_size=80):
    """
    Create colored surfaces with piece letters
    
    Args:
        square_size: Size of each square in pixels
    
    Returns:
        Dict mapping piece symbols to pygame surfaces
    """
    pieces = {}
    font = pygame.font.SysFont('Arial', int(square_size * 0.7), bold=True)
    
    piece_chars = {
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'P', 'n': 'N', 'b': 'B', 'r': 'R', 'q': 'Q', 'k': 'K'
    }
    
    for piece_symbol, char in piece_chars.items():
        surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        
        # Determine color
        if piece_symbol.isupper():
            # White pieces - white with black outline
            text_color = (255, 255, 255)
            outline_color = (0, 0, 0)
        else:
            # Black pieces - black with white outline
            text_color = (0, 0, 0)
            outline_color = (255, 255, 255)
        
        # Draw outline
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    outline_text = font.render(char, True, outline_color)
                    outline_rect = outline_text.get_rect(
                        center=(square_size//2 + dx, square_size//2 + dy)
                    )
                    surface.blit(outline_text, outline_rect)
        
        # Draw main text
        text = font.render(char, True, text_color)
        text_rect = text.get_rect(center=(square_size//2, square_size//2))
        surface.blit(text, text_rect)
        
        pieces[piece_symbol] = surface
    
    return pieces


def load_model_from_checkpoint(checkpoint_path, config, device, model_class):
    """
    Load model from checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: Device to load model on
        model_class: Model class (e.g., ChessNet)
    
    Returns:
        Loaded model in eval mode
    """
    model = model_class(config).to(device)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path.name}")
        
        # Display metadata if available
        if 'val_policy_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint.get('loss', 'N/A'):.4f}")
            print(f"  Policy Loss: {checkpoint['val_policy_loss']:.4f}")
            print(f"  Value Loss: {checkpoint['val_value_loss']:.4f}")
        elif 'win_rate' in checkpoint:
            print(f"  Win Rate: {checkpoint['win_rate']:.2%}")
    else:
        print(f"⚠️  Model file not found: {checkpoint_path}")
        print("Using untrained model!")
    
    model.eval()
    return model


def select_models(base_dir, config):
    """
    Interactive model selection for game modes
    
    Args:
        base_dir: Base directory of project
        config: Configuration dict
    
    Returns:
        Tuple of (model1_path, model2_path, game_mode)
        game_mode: "human_vs_ai", "ai_vs_ai", or "human_vs_human"
    """
    models_dir = base_dir / config['paths']['models_dir']
    
    # Find all .pt files
    all_models = list(models_dir.glob("*.pt"))
    all_models.extend(models_dir.glob("IL/*.pt"))
    all_models.extend(models_dir.glob("RL/*.pt"))
    
    if not all_models:
        print("⚠️  No models found in models directory!")
        return None, None, "human_vs_ai"
    
    # Sort by modification time (newest first)
    all_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("\n" + "="*70)
    print("GAME MODE SELECTION")
    print("="*70)
    print("\n1. Human vs AI")
    print("2. AI vs AI (Watch two models play)")
    print("3. Human vs Human")
    
    mode_choice = input("\nSelect game mode (1-3): ").strip()
    
    if mode_choice == "3":
        return None, None, "human_vs_human"
    
    game_mode = "ai_vs_ai" if mode_choice == "2" else "human_vs_ai"
    
    print("\n" + "="*70)
    print("MODEL SELECTION")
    print("="*70)
    print("\nAvailable models:\n")
    
    for idx, model_path in enumerate(all_models, 1):
        rel_path = model_path.relative_to(models_dir)
        size_mb = model_path.stat().st_size / (1024 ** 2)
        mod_time = model_path.stat().st_mtime
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        
        print(f"{idx:2d}. {rel_path}")
        print(f"    Size: {size_mb:.1f} MB | Modified: {timestamp}")
    
    # Select first model (or white in AI vs AI)
    print(f"\n{'Select White AI model:' if game_mode == 'ai_vs_ai' else 'Select AI model:'}")
    model1_idx = input(f"Enter number (1-{len(all_models)}): ").strip()
    
    try:
        model1_idx = int(model1_idx) - 1
        if model1_idx < 0 or model1_idx >= len(all_models):
            print("Invalid selection!")
            return None, None, game_mode
    except ValueError:
        print("Invalid input!")
        return None, None, game_mode
    
    model1_path = all_models[model1_idx]
    
    # If AI vs AI, select second model
    model2_path = None
    if game_mode == "ai_vs_ai":
        print("\nSelect Black AI model:")
        model2_idx = input(f"Enter number (1-{len(all_models)}): ").strip()
        
        try:
            model2_idx = int(model2_idx) - 1
            if model2_idx < 0 or model2_idx >= len(all_models):
                print("Invalid selection!")
                return None, None, game_mode
        except ValueError:
            print("Invalid input!")
            return None, None, game_mode
        
        model2_path = all_models[model2_idx]
    
    return model1_path, model2_path, game_mode