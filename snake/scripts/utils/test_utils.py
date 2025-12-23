"""
test_utils.py - 🎯 Wspólne funkcje dla skryptów testowania (test_snake_model.py, make_gif.py)
Unika duplikacji kodu między skryptami
"""
import os
import yaml
import torch
from pathlib import Path
from stable_baselines3 import PPO
from model import make_env
from cnn import CustomFeaturesExtractor


def select_visual_style():
    """
    🎨 Interaktywny wybór stylu wizualnego
    
    Returns:
        str: Nazwa stylu ('classic', 'modern', 'realistic')
    """
    print(f"\n{'='*70}")
    print(f"[VISUAL STYLE SELECTION]")
    print(f"{'='*70}")
    print(f"  [1] 🟩 Classic   - Prosty retro styl (szybki)")
    print(f"  [2] 🎮 Modern    - Nowoczesny z gradientami i animacjami")
    print(f"  [3] 🐍 Realistic - Realistyczny z teksturami (wolniejszy)")
    print(f"{'='*70}")
    
    while True:
        choice = input(f"\nWybierz styl [1-3] (default: Classic): ").strip()
        
        if choice == '' or choice == '1':
            print(f"✅ Wybrany styl: Classic\n")
            return 'classic'
        elif choice == '2':
            print(f"✅ Wybrany styl: Modern\n")
            return 'modern'
        elif choice == '3':
            print(f"✅ Wybrany styl: Realistic\n")
            return 'realistic'
        else:
            print("❌ Nieprawidłowy wybór. Wybierz 1-3.")


def select_grid_size_interactive(default_size=8):
    """
    🎯 Interaktywny wybór rozmiaru siatki
    
    Returns:
        int: Wybrany rozmiar siatki
    """
    print(f"\n{'='*70}")
    print(f"[GRID SIZE SELECTION]")
    print(f"{'='*70}")
    print(f"  [1] 🟩 8x8   (Easy - Mała siatka)")
    print(f"  [2] 🟦 12x12 (Medium)")
    print(f"  [3] 🟨 16x16 (Hard - Duża siatka)")
    print(f"  [4] 🟪 Custom (Własny rozmiar)")
    print(f"{'='*70}")
    
    while True:
        choice = input(f"\nWybierz rozmiar siatki [1-4] (default: {default_size}x{default_size}): ").strip()
        
        if choice == '' or choice == '0':
            print(f"✅ Używam domyślnego: {default_size}x{default_size}\n")
            return default_size
        elif choice == '1':
            return 8
        elif choice == '2':
            return 12
        elif choice == '3':
            return 16
        elif choice == '4':
            while True:
                try:
                    custom = input("Podaj rozmiar siatki (4-32): ").strip()
                    custom_size = int(custom)
                    if 4 <= custom_size <= 32:
                        return custom_size
                    else:
                        print("❌ Rozmiar musi być między 4 a 32.")
                except ValueError:
                    print("❌ Nieprawidłowa wartość. Podaj liczbę.")
        else:
            print("❌ Nieprawidłowy wybór. Wybierz 1-4 lub Enter dla domyślnego.")


def load_model_interactive(model_path, policy_path, base_dir):
    """
    🎯 Interaktywny wybór źródła modelu
    
    Args:
        model_path: Ścieżka do modelu
        policy_path: Ścieżka do policy.pth
        base_dir: Ścieżka bazowa projektu
    
    Returns:
        tuple: (model, source_name)
    """
    base_dir = Path(base_dir)
    has_full_model = os.path.exists(model_path)
    has_best_model = os.path.exists(base_dir / 'models' / 'best_model.zip')
    has_policy = os.path.exists(policy_path)
    
    print(f"\n{'='*70}")
    print(f"[MODEL SOURCE SELECTION]")
    print(f"{'='*70}")
    
    options = []
    
    if has_best_model:
        options.append(('1', 'best_model.zip', base_dir / 'models' / 'best_model.zip'))
        print(f"  [1] 🏆 best_model.zip (najlepszy model z treningu)")
    
    if has_full_model and str(model_path) != str(base_dir / 'models' / 'best_model.zip'):
        options.append(('2', 'snake_ppo_model.zip', model_path))
        print(f"  [2] 📦 snake_ppo_model.zip (ostatni checkpoint)")
    
    if has_policy:
        key = str(len(options) + 1)
        options.append((key, 'policy.pth', policy_path))
        print(f"  [{key}] 🎯 policy.pth (tylko wagi sieci)")
    
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
    
    # Załaduj model
    if source_name == 'policy.pth':
        # Wczytaj config
        config_path = base_dir / 'config' / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Stwórz env do sprawdzenia observation_space
        temp_env = make_env(render_mode=None, grid_size=8)()
        
        # Stwórz pusty model
        policy_kwargs = config['model']['policy_kwargs'].copy()
        policy_kwargs['features_extractor_class'] = CustomFeaturesExtractor
        
        model = PPO(
            config['model']['policy'],
            temp_env,
            learning_rate=0.0001,
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
        
        # Załaduj wagi
        state_dict = torch.load(source_path, map_location=config['model']['device'])
        model.policy.load_state_dict(state_dict)
        
        temp_env.close()
        print(f"✅ Załadowano policy.pth\n")
    else:
        model = PPO.load(source_path)
        print(f"✅ Załadowano {source_name}\n")
    
    return model, source_name
