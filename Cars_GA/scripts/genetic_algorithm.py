"""
Klasa GeneticAlgorithm - implementacja algorytmu genetycznego
"""
import numpy as np
from neural_network import NeuralNetwork
from car import Car
import random
import multiprocessing as mp
from functools import partial
import csv
import os
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, config, track):
        """
        Inicjalizacja algorytmu genetycznego
        
        Args:
            config: Słownik z konfiguracją
            track: Obiekt toru wyścigowego
        """
        self.config = config
        self.track = track
        
        ga_config = config['genetic_algorithm']
        self.population_size = ga_config['population_size']
        self.mutation_rate = ga_config['mutation_rate']
        self.mutation_strength = ga_config['mutation_strength']
        self.crossover_rate = ga_config['crossover_rate']
        self.elite_size = ga_config['elite_size']
        self.tournament_size = ga_config['tournament_size']
        self.use_multiprocessing = ga_config.get('use_multiprocessing', True)
        num_workers_config = ga_config.get('num_workers', 0)
        self.num_workers = mp.cpu_count() if num_workers_config == 0 else num_workers_config
        
        # Fitness
        self.max_time = config['fitness']['max_time']
        self.checkpoint_time_bonus = config['fitness'].get('checkpoint_time_bonus', 10)
        self.absolute_max_time = config['fitness'].get('absolute_max_time', 120)
        
        # Stan
        self.generation = 0
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_checkpoints_history = 0
        
        # Statystyki
        self.best_car = None
        self.best_fitness = 0
        
        # Logowanie - ścieżka względem katalogu Cars_GA
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = os.path.join(self.log_dir, "training_log.csv")
        self._initialize_csv()
        
        # Inicjalizuj populację
        self._initialize_population()
    
    def _initialize_csv(self):
        """Inicjalizuje plik CSV z nagłówkami"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Generacja', 'Najlepszy_Fitness', 'Średni_Fitness', 'Najlepsze_Checkpointy'])
        print(f"Utworzono plik logowania: {self.csv_file}")
    
    def _initialize_population(self):
        """Tworzy początkową populację"""
        self.population = []
        
        for i in range(self.population_size):
            # Utwórz samochód
            car = Car(self.track.start_position[0], 
                     self.track.start_position[1],
                     self.track.start_angle,
                     self.config)
            
            # Przypisz losową sieć neuronową
            nn = NeuralNetwork(self.config)
            
            self.population.append({
                'car': car,
                'network': nn,
                'fitness': 0,
                'checkpoint_index': 0,
                'frames_remaining': int(self.max_time * self.config['window']['fps'])
            })
    
    def evaluate_population(self, dt=1.0):
        """
        Ewaluuje całą populację przez symulację
        
        Args:
            dt: Delta time dla symulacji
            
        Returns:
            True jeśli symulacja zakończona, False jeśli trwa
        """
        if self.use_multiprocessing:
            return self._evaluate_population_parallel(dt)
        else:
            return self._evaluate_population_sequential(dt)
    
    def _evaluate_population_sequential(self, dt=1.0):
        """Sekwencyjna ewaluacja (oryginalna wersja)"""
        fps = self.config['window']['fps']
        absolute_max_frames = int(self.absolute_max_time * fps)
        
        frames = 0
        alive_count = self.population_size
        
        # Symulacja aż wszystkie samochody zginą lub skończy się czas
        while alive_count > 0 and frames < absolute_max_frames:
            alive_count = 0
            
            for individual in self.population:
                car = individual['car']
                network = individual['network']
                
                if not car.alive:
                    continue
                
                # Sprawdź czy ma jeszcze czas
                if individual['frames_remaining'] <= 0:
                    car.alive = False
                    continue
                
                alive_count += 1
                
                # Aktualizuj czujniki
                car.update_sensors(self.track.walls)
                
                # Aktualizuj kierunek do najbliższego checkpointu
                checkpoint_idx = individual['checkpoint_index']
                if checkpoint_idx < len(self.track.checkpoints):
                    car.update_checkpoint_direction(self.track.checkpoints[checkpoint_idx])
                
                # Sieć neuronowa decyduje o akcjach (użyj wszystkich sensorów)
                outputs = network.forward(car.get_sensor_inputs())
                
                # Aktualizuj samochód
                car.update(outputs, dt)
                
                # Sprawdź kolizje
                if car.check_collision(self.track.walls):
                    car.alive = False
                    continue
                
                # Sprawdź checkpointy (cyklicznie)
                checkpoint_idx = individual['checkpoint_index']
                if len(self.track.checkpoints) > 0:
                    checkpoint = self.track.checkpoints[checkpoint_idx % len(self.track.checkpoints)]
                    if car.check_checkpoint(checkpoint):
                        car.checkpoints_passed += 1
                        individual['checkpoint_index'] = (individual['checkpoint_index'] + 1) % len(self.track.checkpoints)
                        # Dodaj dodatkowy czas za checkpoint (max do absolute_max_time)
                        bonus_frames = int(self.checkpoint_time_bonus * fps)
                        individual['frames_remaining'] = min(
                            individual['frames_remaining'] + bonus_frames,
                            absolute_max_frames - frames
                        )
                
                # Zmniejsz pozostały czas
                individual['frames_remaining'] -= 1
            
            frames += 1
        
        # Oblicz fitness dla wszystkich (najpierw bez kary, by policzyć średnią)
        fitnesses_tmp = []
        for individual in self.population:
            fitness = individual['car'].calculate_fitness(self.config)
            individual['fitness'] = fitness
            fitnesses_tmp.append(fitness)

        avg_fitness = sum(fitnesses_tmp) / len(fitnesses_tmp)
        # Teraz policz fitness z adaptacyjną karą
        for individual in self.population:
            fitness = individual['car'].calculate_fitness(self.config, avg_fitness=avg_fitness, generation=self.generation)
            individual['fitness'] = fitness
        
        # Pobierz statystyki PRZED resetowaniem w evolve()
        fitnesses = [ind['fitness'] for ind in self.population]
        checkpoints = [ind['car'].checkpoints_passed for ind in self.population]
        self.best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_checkpoints = max(checkpoints) if checkpoints else 0
        
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.best_checkpoints_history = best_checkpoints
        
        # Zapisz do CSV
        self._log_to_csv()
        
        return True
    
    def _evaluate_population_parallel(self, dt=1.0):
        """
        Równoległa ewaluacja populacji używając multiprocessing
        
        Args:
            dt: Delta time dla symulacji
            
        Returns:
            True jeśli symulacja zakończona
        """
        # Przygotuj dane dla każdego osobnika
        eval_args = []
        for individual in self.population:
            eval_args.append({
                'network_weights': individual['network'].get_weights(),
                'network_config': {
                    'input_size': individual['network'].input_size,
                    'hidden_layers': individual['network'].hidden_layers,
                    'output_size': individual['network'].output_size
                },
                'config': self.config,
                'track_data': self.track.to_dict(),
                'max_time': self.max_time,
                'checkpoint_time_bonus': self.checkpoint_time_bonus,
                'absolute_max_time': self.absolute_max_time,
                'dt': dt
            })
        
        # Uruchom równolegle
        pool = None
        try:
            pool = mp.Pool(processes=self.num_workers)
            results = pool.map(_evaluate_individual_wrapper, eval_args)
        except KeyboardInterrupt:
            import sys
            import io
            # Natychmiast zamknij pool i nie czekaj na worker'y
            if pool:
                pool.terminate()
                # Nie czekaj na pool.join() - to powoduje zawieszenie
            sys.stdout.flush()
            sys.stderr.flush()
            print("\n\nPrzerwano trening (Ctrl+C)", flush=True)
            sys.exit(0)
        finally:
            if pool:
                pool.close()
                # Czekaj na zamknięcie tylko jeśli nie było KeyboardInterrupt
                try:
                    pool.join()
                except KeyboardInterrupt:
                    pass
        
        # Zaktualizuj populację wynikami
        for i, result in enumerate(results):
            individual = self.population[i]
            individual['car'].fitness = result['fitness']
            individual['car'].checkpoints_passed = result['checkpoints_passed']
            individual['car'].distance_traveled = result['distance_traveled']
            individual['car'].time_alive = result['time_alive']
            individual['fitness'] = result['fitness']
            
            # Odejmij penalty za złożoność sieci
            network = individual['network']
            num_params = network.get_num_parameters()
            complexity_penalty = num_params * self.config['fitness'].get('network_complexity_penalty', 0.001)
            individual['fitness'] -= complexity_penalty
        
        # Pobierz statystyki
        fitnesses = [ind['fitness'] for ind in self.population]
        checkpoints = [ind['car'].checkpoints_passed for ind in self.population]
        self.best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_checkpoints = max(checkpoints) if checkpoints else 0
        
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.best_checkpoints_history = best_checkpoints
        
        # Zapisz do CSV
        self._log_to_csv()
        
        return True
    
    def evolve(self):
        """Ewoluuje populację do następnej generacji"""
        # Sortuj według fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Zapisz statystyki
        fitnesses = [ind['fitness'] for ind in self.population]
        self.best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Zapisz najlepszego
        self.best_car = self.population[0]['car']
        
        # Utwórz nową populację
        new_population = []
        
        # Elityzm - zachowaj najlepszych
        for i in range(self.elite_size):
            elite = self.population[i].copy()
            elite['car'].reset()
            elite['checkpoint_index'] = 0
            new_population.append(elite)
        
        # Generuj resztę populacji
        while len(new_population) < self.population_size:
            # Selekcja turniejowa
            parent1_net = self._tournament_selection()
            parent2_net = self._tournament_selection()
            
            # Krzyżowanie
            if random.random() < self.crossover_rate:
                child_net = NeuralNetwork.crossover(parent1_net, parent2_net)
            else:
                child_net = parent1_net.copy()
            
            # Mutacja
            child_net.mutate(self.mutation_rate, self.mutation_strength)
            
            # Utwórz samochód dla dziecka
            car = Car(self.track.start_position[0], 
                     self.track.start_position[1],
                     self.track.start_angle,
                     self.config)
            
            new_population.append({
                'car': car,
                'network': child_net,
                'fitness': 0,
                'checkpoint_index': 0,
                'frames_remaining': int(self.max_time * self.config['window']['fps'])
            })
        
        self.population = new_population
        self.generation += 1
    
    def _tournament_selection(self):
        """
        Selekcja turniejowa
        
        Returns:
            Sieć neuronowa zwycięzcy turnieju
        """
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda x: x['fitness'])
        return winner['network']
    
    def get_best_network(self):
        """
        Zwraca najlepszą sieć neuronową
        
        Returns:
            Obiekt NeuralNetwork
        """
        if self.population:
            best = max(self.population, key=lambda x: x['fitness'])
            return best['network']
        return None
    
    def save_best(self, filepath):
        """
        Zapisuje najlepszą sieć do pliku
        
        Args:
            filepath: Ścieżka do pliku
        """
        best_net = self.get_best_network()
        if best_net:
            best_net.save(filepath)
            print(f"Zapisano najlepszą sieć: {filepath}")
    
    def _log_to_csv(self):
        """Zapisuje statystyki do pliku CSV"""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.generation,
                self.best_fitness_history[-1] if self.best_fitness_history else 0,
                self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
                self.best_checkpoints_history
            ])
    
    def save_training_plots(self, filename_prefix="training_plot"):
        """Generuje i zapisuje wykresy postępu treningu"""
        if not self.best_fitness_history:
            print("Brak danych do wygenerowania wykresów")
            return
        
        generations = list(range(1, len(self.best_fitness_history) + 1))
        
        # Utwórz figurę z 2 wykresami
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Wykres 1: Fitness
        ax1.plot(generations, self.best_fitness_history, 'b-', label='Najlepszy Fitness', linewidth=2)
        ax1.plot(generations, self.avg_fitness_history, 'g--', label='Średni Fitness', linewidth=2)
        ax1.set_xlabel('Generacja')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Postęp Treningu - Fitness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wykres 2: Checkpointy (zbierz historię)
        # Wczytaj z CSV aby mieć pełną historię checkpointów
        checkpoint_history = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    checkpoint_history.append(int(row['Najlepsze_Checkpointy']))
        except:
            # Jeśli nie można wczytać, użyj aktualnej wartości
            checkpoint_history = [self.best_checkpoints_history] * len(generations)
        
        if checkpoint_history:
            ax2.plot(range(1, len(checkpoint_history) + 1), checkpoint_history, 'r-', 
                    label='Najlepsze Checkpointy', linewidth=2)
            ax2.set_xlabel('Generacja')
            ax2.set_ylabel('Liczba Checkpointów')
            ax2.set_title('Postęp Treningu - Checkpointy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Zapisz wykres
        plot_path = os.path.join(self.log_dir, f"{filename_prefix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisany: {plot_path}")
        plt.close()
    
    def get_statistics(self):
        """
        Zwraca statystyki obecnej generacji
        
        Returns:
            Słownik ze statystykami
        """
        if not self.population:
            return {}
        
        fitnesses = [ind['fitness'] for ind in self.population]
        checkpoints = [ind['car'].checkpoints_passed for ind in self.population]
        
        # Znajdź najlepszego osobnika
        best_ind = max(self.population, key=lambda x: x['fitness'])
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'best_checkpoints': self.best_checkpoints_history,
            'avg_checkpoints': sum(checkpoints) / len(checkpoints),
            'population_size': len(self.population),
            'best_architecture': best_ind['network'].hidden_layers
        }


# === FUNKCJE POMOCNICZE DLA MULTIPROCESSINGU ===

def _evaluate_individual_wrapper(args):
    """
    Wrapper dla ewaluacji pojedynczego osobnika w osobnym procesie
    
    Args:
        args: Słownik z parametrami
        
    Returns:
        Słownik z wynikami symulacji
    """
    try:
        # Ukryj komunikaty pygame w worker procesach
        import os
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        
        from track import Track
        from neural_network import NeuralNetwork
        from car import Car
        
        # Rozpakuj argumenty
        network_weights = args['network_weights']
        network_config = args['network_config']
        config = args['config']
        track_data = args['track_data']
        max_time = args['max_time']
        checkpoint_time_bonus = args['checkpoint_time_bonus']
        absolute_max_time = args['absolute_max_time']
        dt = args['dt']
        
        # Utwórz tor
        track = Track.from_dict(track_data)
        
        # Utwórz sieć neuronową
        network = NeuralNetwork(config)
        network.input_size = network_config['input_size']
        network.hidden_layers = network_config['hidden_layers']
        network.output_size = network_config['output_size']
        network.set_weights(network_weights)
        
        # Utwórz samochód
        car = Car(track.start_position[0], 
                 track.start_position[1],
                 track.start_angle,
                 config)
        
        # Symulacja
        fps = config['window']['fps']
        absolute_max_frames = int(absolute_max_time * fps)
        frames_remaining = int(max_time * fps)
        checkpoint_index = 0
        frames = 0
        
        while car.alive and frames_remaining > 0 and frames < absolute_max_frames:
            # Aktualizuj czujniki
            car.update_sensors(track.walls)
            
            # Aktualizuj kierunek do najbliższego checkpointu (cyklicznie)
            if len(track.checkpoints) > 0:
                car.update_checkpoint_direction(track.checkpoints[checkpoint_index % len(track.checkpoints)])
            
            # Sieć neuronowa decyduje o akcjach (użyj wszystkich sensorów)
            outputs = network.forward(car.get_sensor_inputs())
            
            # Aktualizuj samochód
            car.update(outputs, dt)
            
            # Sprawdź kolizje
            if car.check_collision(track.walls):
                car.alive = False
                break
            
            # Sprawdź checkpointy (cyklicznie)
            if len(track.checkpoints) > 0:
                checkpoint = track.checkpoints[checkpoint_index % len(track.checkpoints)]
                if car.check_checkpoint(checkpoint):
                    car.checkpoints_passed += 1
                    checkpoint_index = (checkpoint_index + 1) % len(track.checkpoints)
                    # Dodaj dodatkowy czas za checkpoint
                    bonus_frames = int(checkpoint_time_bonus * fps)
                    frames_remaining = min(
                        frames_remaining + bonus_frames,
                        absolute_max_frames - frames
                    )
            
            frames_remaining -= 1
            frames += 1
        
        # Oblicz fitness
        fitness = car.calculate_fitness(config)
        
        return {
            'fitness': fitness,
            'checkpoints_passed': car.checkpoints_passed,
            'distance_traveled': car.distance_traveled,
            'time_alive': car.time_alive
        }
    
    except KeyboardInterrupt:
        # Wyjdź natychmiast bez drukowania błędu
        import sys
        sys.exit(130)  # Standard exit code dla Ctrl+C
    except Exception:
        # W razie błędu zwróć zerowe wyniki zamiast drukowania
        return {
            'fitness': 0.0,
            'checkpoints_passed': 0,
            'distance_traveled': 0.0,
            'time_alive': 0.0
        }
