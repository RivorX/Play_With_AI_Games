"""
Narzędzia pomocnicze
"""


def print_banner():
    """Wyświetla banner aplikacji"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║         CARS-GA v1.0                  ║
    ║  Algorytm Genetyczny dla Samochodów  ║
    ╚═══════════════════════════════════════╝
    """
    print(banner)


def format_time(seconds):
    """
    Formatuje czas w sekundach do czytelnej formy
    
    Args:
        seconds: Liczba sekund
        
    Returns:
        String z sformatowanym czasem
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_statistics(stats):
    """
    Wyświetla statystyki generacji
    
    Args:
        stats: Słownik ze statystykami
    """
    print("\n" + "="*50)
    print(f"Generacja: {stats['generation']}")
    print("-"*50)
    print(f"Fitness:")
    print(f"  Najlepszy:  {stats['best_fitness']:8.2f}")
    print(f"  Średni:     {stats['avg_fitness']:8.2f}")
    print(f"  Najgorszy:  {stats['worst_fitness']:8.2f}")
    print(f"Checkpointy:")
    print(f"  Najlepiej:  {stats['best_checkpoints']:8d}")
    print(f"  Średnio:    {stats['avg_checkpoints']:8.2f}")
    if 'best_architecture' in stats:
        print(f"Architektura (Best): {stats['best_architecture']}")
    print("="*50)


def save_model_info(network, filepath, generation, fitness):
    """
    Zapisuje informacje o modelu
    
    Args:
        network: Obiekt NeuralNetwork
        filepath: Ścieżka do pliku info
        generation: Numer generacji
        fitness: Wartość fitness
    """
    import json
    
    info = {
        'generation': generation,
        'fitness': fitness,
        'num_parameters': network.get_num_parameters(),
        'architecture': {
            'input_size': network.input_size,
            'hidden_layers': network.hidden_layers,
            'output_size': network.output_size
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
