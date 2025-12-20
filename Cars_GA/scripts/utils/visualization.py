"""
Wizualizacja statystyk treningu
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_fitness_history(best_fitness, avg_fitness, save_path='logs/fitness.png'):
    """
    Rysuje wykres historii fitness
    
    Args:
        best_fitness: Lista najlepszych fitness w każdej generacji
        avg_fitness: Lista średnich fitness w każdej generacji
        save_path: Ścieżka do zapisu wykresu
    """
    plt.figure(figsize=(12, 6))
    
    generations = range(1, len(best_fitness) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Najlepszy')
    plt.plot(generations, avg_fitness, 'g--', linewidth=2, label='Średni')
    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.title('Historia Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    improvement = np.array(best_fitness) - best_fitness[0]
    plt.plot(generations, improvement, 'r-', linewidth=2)
    plt.xlabel('Generacja')
    plt.ylabel('Poprawa od startu')
    plt.title('Poprawa Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Wykres zapisany: {save_path}")
    plt.close()


def plot_population_diversity(population_fitness, save_path='logs/diversity.png'):
    """
    Rysuje wykres różnorodności populacji
    
    Args:
        population_fitness: Lista list fitness dla każdej generacji
        save_path: Ścieżka do zapisu wykresu
    """
    plt.figure(figsize=(10, 6))
    
    generations = range(1, len(population_fitness) + 1)
    
    # Oblicz statystyki
    best = [max(gen) for gen in population_fitness]
    worst = [min(gen) for gen in population_fitness]
    avg = [np.mean(gen) for gen in population_fitness]
    std = [np.std(gen) for gen in population_fitness]
    
    plt.fill_between(generations, worst, best, alpha=0.3, color='blue', label='Zakres')
    plt.plot(generations, avg, 'g-', linewidth=2, label='Średnia')
    plt.plot(generations, best, 'b-', linewidth=1, label='Najlepszy')
    plt.plot(generations, worst, 'r-', linewidth=1, label='Najgorszy')
    
    plt.xlabel('Generacja')
    plt.ylabel('Fitness')
    plt.title('Różnorodność Populacji')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150)
    print(f"Wykres zapisany: {save_path}")
    plt.close()


def save_training_log(ga, filepath='logs/training_log.txt'):
    """
    Zapisuje log z treningu
    
    Args:
        ga: Obiekt GeneticAlgorithm
        filepath: Ścieżka do pliku
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== LOG TRENINGU CARS-GA ===\n\n")
        
        f.write("Parametry algorytmu genetycznego:\n")
        f.write(f"  Rozmiar populacji: {ga.population_size}\n")
        f.write(f"  Współczynnik mutacji: {ga.mutation_rate}\n")
        f.write(f"  Siła mutacji: {ga.mutation_strength}\n")
        f.write(f"  Współczynnik krzyżowania: {ga.crossover_rate}\n")
        f.write(f"  Rozmiar elity: {ga.elite_size}\n\n")
        
        f.write("Wyniki treningu:\n")
        f.write(f"  Liczba generacji: {ga.generation}\n")
        f.write(f"  Najlepszy fitness: {ga.best_fitness:.2f}\n")
        
        if ga.best_fitness_history:
            f.write(f"  Początkowy fitness: {ga.best_fitness_history[0]:.2f}\n")
            f.write(f"  Poprawa: {ga.best_fitness - ga.best_fitness_history[0]:.2f}\n")
        
        f.write("\nHistoria najlepszych fitness:\n")
        for i, fitness in enumerate(ga.best_fitness_history, 1):
            f.write(f"  Gen {i:3d}: {fitness:.2f}\n")
    
    print(f"Log zapisany: {filepath}")
