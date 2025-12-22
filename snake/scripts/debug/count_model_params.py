"""
Model Parameter Counter - zlicz wszystkie parametry w modelu RecurrentPPO
"""
import torch
import yaml
from collections import defaultdict

def count_parameters(model, print_details=True):
    """
    Zlicz parametry w całym modelu z podziałem na kategorie
    
    Args:
        model: Model PPO (bez LSTM)
        print_details: Czy drukować szczegółowe informacje
    
    Returns:
        dict: Słownik z liczbą parametrów dla każdej kategorii
    """
    params = {
        'extractor': {
            'cnn': 0,
            'spatial_attention': 0,
            'bottleneck': 0,
            'scalars': 0,
            'fusion': 0,
        },
        'policy_head': 0,
        'value_head': 0,
        'other': 0
    }
    
    # Pobierz policy network
    policy = model.policy
    
    # 1. EXTRACTOR (CNN + Scalars + Fusion)
    if hasattr(policy, 'features_extractor') and hasattr(policy.features_extractor, 'get_params_info'):
        extractor_info = policy.features_extractor.get_params_info()
        params['extractor'] = extractor_info
    else:
        # Fallback: zlicz manualnie
        if hasattr(policy, 'features_extractor'):
            params['extractor']['total'] = sum(p.numel() for p in policy.features_extractor.parameters())
    
    # 2. POLICY HEAD (action_net)
    if hasattr(policy, 'action_net'):
        params['policy_head'] = sum(p.numel() for p in policy.action_net.parameters())
    
    # 3. VALUE HEAD (value_net)
    if hasattr(policy, 'value_net'):
        params['value_head'] = sum(p.numel() for p in policy.value_net.parameters())
    
    # 4. OTHER (latent_pi, latent_vf, itp.)
    total_counted = (sum(params['extractor'].values()) + 
                    params['policy_head'] + 
                    params['value_head'])
    
    total_all = sum(p.numel() for p in policy.parameters())
    params['other'] = total_all - total_counted
    
    # Oblicz sumy
    extractor_total = sum(params['extractor'].values())
    grand_total = extractor_total + params['policy_head'] + params['value_head'] + params['other']
    
    if print_details:
        print(f"\n{'='*80}")
        print(f"{'🎯 FULL MODEL PARAMETER COUNT':^80}")
        print(f"{'='*80}\n")
        
        # EXTRACTOR
        print(f"{'📸 FEATURE EXTRACTOR':<40} {extractor_total:>12,} ({extractor_total/grand_total*100:>5.1f}%)")
        print(f"{'─'*80}")
        for key, val in params['extractor'].items():
            if val > 0:
                print(f"  ├─ {key.replace('_', ' ').title():<36} {val:>12,} ({val/extractor_total*100:>5.1f}%)")
        print()
        
        # LSTM
        print(f"{'🔄 LSTM (Recurrent Memory)':<40} {params['lstm']:>12,} ({params['lstm']/grand_total*100:>5.1f}%)")
        print()
        
        # POLICY HEAD
        print(f"{'🎮 POLICY HEAD (Actor)':<40} {params['policy_head']:>12,} ({params['policy_head']/grand_total*100:>5.1f}%)")
        print()
        
        # VALUE HEAD
        print(f"{'💎 VALUE HEAD (Critic)':<40} {params['value_head']:>12,} ({params['value_head']/grand_total*100:>5.1f}%)")
        print()
        
        # OTHER
        if params['other'] > 0:
            print(f"{'🔧 OTHER (Latent layers, etc.)':<40} {params['other']:>12,} ({params['other']/grand_total*100:>5.1f}%)")
            print()
        
        # TOTAL
        print(f"{'─'*80}")
        print(f"{'✅ GRAND TOTAL':<40} {grand_total:>12,} (100.0%)")
        print(f"{'='*80}\n")
        
        # Memory estimate
        memory_mb = grand_total * 4 / (1024**2)  # float32 = 4 bytes
        print(f"💾 Estimated Memory (FP32): {memory_mb:.1f} MB")
        print(f"💾 Estimated Memory (FP16): {memory_mb/2:.1f} MB")
        print(f"{'='*80}\n")
    
    return params, grand_total


def main():
    """
    Główna funkcja - wczytaj model i policz parametry
    """
    import sys
    import os
    
    # Dodaj ścieżkę do projektu
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    # Ustal katalog snake (dwa poziomy wyżej względem tego pliku)
    snake_dir = os.path.abspath(os.path.join(base_dir, '..', '..'))
    
    # 🔧 KLUCZOWA ZMIANA: Dodaj ścieżkę do katalogu snake, żeby Python mógł znaleźć moduł 'cnn'
    if snake_dir not in sys.path:
        sys.path.insert(0, snake_dir)
    
    # Możesz też spróbować zaimportować moduły przed wczytaniem modelu
    try:
        import cnn  # Upewnij się, że moduł cnn jest dostępny
        print("✅ Moduł 'cnn' znaleziony")
    except ImportError as e:
        print(f"⚠️  Ostrzeżenie: Nie można zaimportować modułu 'cnn': {e}")
        print(f"   Sprawdzam ścieżkę: {snake_dir}")
        # Spróbuj znaleźć plik cnn.py
        cnn_path = os.path.join(snake_dir, 'cnn.py')
        if os.path.exists(cnn_path):
            print(f"   Znaleziono cnn.py w: {cnn_path}")
        else:
            print(f"   ❌ Nie znaleziono cnn.py")
            # Szukaj w podkatalogach
            for root, dirs, files in os.walk(snake_dir):
                if 'cnn.py' in files:
                    print(f"   💡 Znaleziono cnn.py w: {root}")
                    if root not in sys.path:
                        sys.path.insert(0, root)
                    break

    try:
        from sb3_contrib import RecurrentPPO

        # Wczytaj config z katalogu snake/config/config.yaml
        config_path = os.path.join(snake_dir, 'config', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_path = config['paths']['model_path']
        # Jeśli ścieżka jest względna, szukaj w snake/models
        if not os.path.isabs(model_path):
            candidate_path = os.path.join(snake_dir, 'models', model_path) if not os.path.exists(os.path.join(snake_dir, model_path)) else os.path.join(snake_dir, model_path)
            if os.path.exists(candidate_path):
                model_path = candidate_path

        if not os.path.exists(model_path):
            print(f"❌ Model nie istnieje: {model_path}")
            print("💡 Najpierw wytrenuj model lub podaj ścieżkę do istniejącego modelu")
            return

        print(f"📥 Wczytuję model z: {model_path}")
        model = RecurrentPPO.load(model_path)

        print("🔢 Liczę parametry...\n")
        params, total = count_parameters(model, print_details=True)

        # Zapisz wyniki do pliku (do katalogu logs w snake)
        output_path = os.path.join(snake_dir, 'logs', 'model_parameters.txt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Parameter Count\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"Total parameters: {total:,}\n\n")

            f.write(f"Breakdown:\n")
            f.write(f"  Feature Extractor: {sum(params['extractor'].values()):,}\n")
            for key, val in params['extractor'].items():
                if val > 0:
                    f.write(f"    - {key}: {val:,}\n")
            f.write(f"  LSTM: {params['lstm']:,}\n")
            f.write(f"  Policy Head: {params['policy_head']:,}\n")
            f.write(f"  Value Head: {params['value_head']:,}\n")
            if params['other'] > 0:
                f.write(f"  Other: {params['other']:,}\n")

        print(f"✅ Zapisano szczegóły do: {output_path}")

    except FileNotFoundError as e:
        print(f"❌ Błąd: {e}")
        print("💡 Upewnij się, że model został wytrenowany i zapisany")
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()