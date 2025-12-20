"""
Klasa GeneticAlgorithm - implementacja algorytmu genetycznego
"""
import numpy as np
from neural_network import NeuralNetwork
from car import Car
import random


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
        
        # Fitness
        self.max_time = config['fitness']['max_time']
        
        # Stan
        self.generation = 0
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_checkpoints_history = 0
        
        # Statystyki
        self.best_car = None
        self.best_fitness = 0
        
        # Inicjalizuj populację
        self._initialize_population()
    
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
                'checkpoint_index': 0
            })
    
    def evaluate_population(self, dt=1.0):
        """
        Ewaluuje całą populację przez symulację
        
        Args:
            dt: Delta time dla symulacji
            
        Returns:
            True jeśli symulacja zakończona, False jeśli trwa
        """
        fps = self.config['window']['fps']
        max_frames = int(self.max_time * fps)
        
        frames = 0
        alive_count = self.population_size
        
        # Symulacja aż wszystkie samochody zginą lub skończy się czas
        while alive_count > 0 and frames < max_frames:
            alive_count = 0
            
            for individual in self.population:
                car = individual['car']
                network = individual['network']
                
                if not car.alive:
                    continue
                
                alive_count += 1
                
                # Aktualizuj czujniki
                car.update_sensors(self.track.walls)
                
                # Sieć neuronowa decyduje o akcjach
                outputs = network.forward(car.sensor_readings)
                
                # Aktualizuj samochód
                car.update(outputs, dt)
                
                # Sprawdź kolizje
                if car.check_collision(self.track.walls):
                    car.alive = False
                    continue
                
                # Sprawdź checkpointy
                checkpoint_idx = individual['checkpoint_index']
                if checkpoint_idx < len(self.track.checkpoints):
                    checkpoint = self.track.checkpoints[checkpoint_idx]
                    if car.check_checkpoint(checkpoint):
                        car.checkpoints_passed += 1
                        individual['checkpoint_index'] += 1
            
            frames += 1
        
        # Oblicz fitness dla wszystkich
        for individual in self.population:
            individual['fitness'] = individual['car'].calculate_fitness(self.config)
            
            # Odejmij penalty za złożoność sieci
            network = individual['network']
            num_params = network.get_num_parameters()
            complexity_penalty = num_params * self.config['fitness'].get('network_complexity_penalty', 0.001)
            individual['fitness'] -= complexity_penalty
        
        # Pobierz statystyki PRZED resetowaniem w evolve()
        fitnesses = [ind['fitness'] for ind in self.population]
        checkpoints = [ind['car'].checkpoints_passed for ind in self.population]
        self.best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_checkpoints = max(checkpoints) if checkpoints else 0
        
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.best_checkpoints_history = best_checkpoints
        
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
                'checkpoint_index': 0
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
