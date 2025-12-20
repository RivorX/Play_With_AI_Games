"""
Główna aplikacja Cars-GA z menu i różnymi trybami
"""
import pygame
import sys
import yaml
import os
from track import Track
from car import Car
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np


class CarsGA:
    def __init__(self):
        """Inicjalizacja głównej aplikacji"""
        # Ustal bazowy katalog projektu
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Wczytaj konfigurację
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Pygame
        pygame.init()
        self.width = self.config['window']['width']
        self.height = self.config['window']['height']
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Cars-GA - Algorytm Genetyczny")
        self.clock = pygame.time.Clock()
        self.fps = self.config['window']['fps']
        
        # Fonty
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
        
        # Stan aplikacji
        self.state = "menu"  # menu, train, test, editor
        self.running = True
        
        # Kolory
        self.bg_color = (20, 20, 30)
        self.button_color = (60, 60, 80)
        self.button_hover_color = (80, 80, 120)
        self.text_color = (255, 255, 255)
        
        # Menu
        self.menu_buttons = []
        self._create_menu()
    
    def _create_menu(self):
        """Tworzy przyciski menu"""
        button_width = 400
        button_height = 60
        button_x = (self.width - button_width) // 2
        start_y = 250
        spacing = 80
        
        self.menu_buttons = [
            {
                'rect': pygame.Rect(button_x, start_y, button_width, button_height),
                'text': 'TRENUJ AI',
                'action': 'train'
            },
            {
                'rect': pygame.Rect(button_x, start_y + spacing, button_width, button_height),
                'text': 'TESTUJ MODEL',
                'action': 'test'
            },
            {
                'rect': pygame.Rect(button_x, start_y + spacing * 2, button_width, button_height),
                'text': 'EDYTOR TORÓW',
                'action': 'editor'
            },
            {
                'rect': pygame.Rect(button_x, start_y + spacing * 3, button_width, button_height),
                'text': 'WYJŚCIE',
                'action': 'quit'
            }
        ]
    
    def run(self):
        """Główna pętla aplikacji"""
        while self.running:
            if self.state == "menu":
                self._run_menu()
            elif self.state == "train":
                self._run_training()
                self.state = "menu"
            elif self.state == "test":
                self._run_testing()
                self.state = "menu"
            elif self.state == "editor":
                self._run_editor()
                self.state = "menu"
        
        pygame.quit()
    
    def _run_menu(self):
        """Pętla menu głównego"""
        while self.state == "menu" and self.running:
            dt = self.clock.tick(self.fps) / 1000.0
            mouse_pos = pygame.mouse.get_pos()
            
            # Obsługa zdarzeń
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.state = None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.state = None
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for button in self.menu_buttons:
                            if button['rect'].collidepoint(mouse_pos):
                                if button['action'] == 'quit':
                                    self.running = False
                                    self.state = None
                                else:
                                    self.state = button['action']
            
            # Rysowanie
            self.screen.fill(self.bg_color)
            
            # Tytuł
            title = self.large_font.render("CARS-GA", True, (100, 200, 255))
            title_rect = title.get_rect(center=(self.width // 2, 100))
            self.screen.blit(title, title_rect)
            
            subtitle = self.font.render("Algorytm Genetyczny dla Samochodów", True, (150, 150, 150))
            subtitle_rect = subtitle.get_rect(center=(self.width // 2, 150))
            self.screen.blit(subtitle, subtitle_rect)
            
            # Przyciski
            for button in self.menu_buttons:
                color = self.button_hover_color if button['rect'].collidepoint(mouse_pos) else self.button_color
                pygame.draw.rect(self.screen, color, button['rect'], border_radius=10)
                pygame.draw.rect(self.screen, (100, 100, 100), button['rect'], 2, border_radius=10)
                
                text = self.font.render(button['text'], True, self.text_color)
                text_rect = text.get_rect(center=button['rect'].center)
                self.screen.blit(text, text_rect)
            
            pygame.display.flip()
    
    def _run_training(self):
        """Tryb trenowania AI z graficznym wyborem mapy"""
        # Przygotuj listę torów: bazowy + z katalogu
        tracks_dir = os.path.join(self.base_dir, 'tracks')
        tracks = Track.list_tracks(tracks_dir)
        map_options = [('Bazowa mapa', 'simple')] + [(name, name) for name in tracks]

        # Graficzny wybór mapy
        selected = 0
        choosing = True
        while choosing:
            self.screen.fill(self.bg_color)
            title = self.large_font.render("WYBÓR MAPY DO TRENINGU", True, (100, 200, 255))
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 60))

            for i, (label, _) in enumerate(map_options):
                color = (200, 200, 80) if i == selected else self.button_color
                rect = pygame.Rect(self.width//2 - 200, 180 + i*70, 400, 50)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 2, border_radius=8)
                text = self.font.render(label, True, self.text_color)
                self.screen.blit(text, (rect.x + 20, rect.y + 10))

            info = self.small_font.render("Strzałki: wybierz, Enter: zatwierdź, ESC: anuluj", True, (180,180,180))
            self.screen.blit(info, (self.width//2 - info.get_width()//2, 180 + len(map_options)*70))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_UP:
                        selected = (selected - 1) % len(map_options)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(map_options)
                    elif event.key == pygame.K_RETURN:
                        choosing = False
                        break

        # Załaduj wybrany tor
        label, track_name = map_options[selected]
        if track_name == 'simple':
            track = Track.create_simple_track()
            # Zapisz jeśli nie istnieje
            if 'simple' not in tracks:
                track.save(tracks_dir)
        else:
            track = Track()
            track.load(track_name, tracks_dir)

        # Inicjalizuj algorytm genetyczny
        ga = GeneticAlgorithm(self.config, track)

        # Parametry treningu
        max_generations = self.config['genetic_algorithm']['generations']
        save_interval = self.config['training']['save_interval']
        show_best = self.config['training']['show_best']
        show_sensors = self.config['training']['show_sensors']

        print(f"\nRozpoczynanie treningu na {max_generations} generacji...")
        print(f"Rozmiar populacji: {ga.population_size}")

        # Pętla treningowa
        running = True
        generation = 0

        while running and generation < max_generations:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if generation % 10 == 0 or generation == max_generations - 1:
                self._visualize_generation(ga, track, show_sensors)
            else:
                ga.evaluate_population()

            ga.evolve()
            generation = ga.generation

            stats = ga.get_statistics()
            print(f"\nGeneracja {stats['generation']}")
            print(f"  Najlepszy fitness: {stats['best_fitness']:.2f}")
            print(f"  Średni fitness: {stats['avg_fitness']:.2f}")
            print(f"  Najlepiej checkpointów: {stats['best_checkpoints']}")

            if generation % save_interval == 0:
                models_dir = os.path.join(self.base_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, f"generation_{generation}.json")
                ga.save_best(model_path)

        if running:
            models_dir = os.path.join(self.base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'best_model.json')
            ga.save_best(model_path)
            print("\n=== Trening zakończony! ===")
            print(f"Najlepszy fitness: {ga.best_fitness:.2f}")
            self._plot_training_progress(ga)

        # Czekaj na powrót do menu
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            info = self.font.render("Trening zakończony. Naciśnij dowolny klawisz...", True, (180,180,180))
            self.screen.fill(self.bg_color)
            self.screen.blit(info, (self.width//2 - info.get_width()//2, self.height//2))
            pygame.display.flip()
    
    def _visualize_generation(self, ga, track, show_sensors):
        """
        Wizualizuje symulację generacji
        
        Args:
            ga: Obiekt GeneticAlgorithm
            track: Obiekt Track
            show_sensors: Czy pokazywać czujniki
        """
        dt = 1.0
        max_frames = int(ga.max_time * self.fps)
        frames = 0
        
        while frames < max_frames:
            self.clock.tick(self.fps)
            
            # Obsługa zdarzeń
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            # Aktualizuj populację
            alive_count = 0
            
            for individual in ga.population:
                car = individual['car']
                network = individual['network']
                
                if not car.alive:
                    continue
                
                alive_count += 1
                
                # Czujniki
                car.update_sensors(track.walls)
                
                # Decyzja sieci
                outputs = network.forward(car.sensor_readings)
                
                # Aktualizuj
                car.update(outputs, dt)
                
                # Kolizje
                if car.check_collision(track.walls):
                    car.alive = False
                    continue
                
                # Checkpointy
                checkpoint_idx = individual['checkpoint_index']
                if checkpoint_idx < len(track.checkpoints):
                    if car.check_checkpoint(track.checkpoints[checkpoint_idx]):
                        car.checkpoints_passed += 1
                        individual['checkpoint_index'] += 1
            
            # Rysuj
            self.screen.fill(self.bg_color)
            track.draw(self.screen)
            
            # Rysuj samochody
            for individual in ga.population:
                car = individual['car']
                if car.alive:
                    car.draw(self.screen, show_sensors)
            
            # UI
            self._draw_training_ui(ga, alive_count, frames, max_frames)
            
            pygame.display.flip()
            
            frames += 1
            
            if alive_count == 0:
                break
        
        # Oblicz fitness
        for individual in ga.population:
            individual['fitness'] = individual['car'].calculate_fitness(self.config)
    
    def _draw_training_ui(self, ga, alive_count, frames, max_frames):
        """Rysuje UI podczas treningu"""
        y = 10
        
        # Generacja
        text = self.font.render(f"Generacja: {ga.generation}", True, (255, 255, 255))
        self.screen.blit(text, (10, y))
        y += 35
        
        # Żywe samochody
        text = self.small_font.render(f"Żywe: {alive_count}/{ga.population_size}", True, (0, 255, 0))
        self.screen.blit(text, (10, y))
        y += 30
        
        # Czas
        time_percent = (frames / max_frames) * 100
        text = self.small_font.render(f"Czas: {time_percent:.1f}%", True, (200, 200, 200))
        self.screen.blit(text, (10, y))
        y += 30
        
        # Najlepszy fitness
        if ga.best_fitness_history:
            text = self.small_font.render(f"Najlepszy: {ga.best_fitness_history[-1]:.2f}", 
                                        True, (255, 200, 0))
            self.screen.blit(text, (10, y))
    
    def _run_testing(self):
        """Tryb testowania wytrenowanego modelu"""
        print("\n=== TESTOWANIE MODELU ===")
        
        # Sprawdź czy istnieje model
        model_path = os.path.join(self.base_dir, 'models', 'best_model.json')
        if not os.path.exists(model_path):
            print("Brak wytrenowanego modelu! Najpierw wytrenuj AI.")
            input("Naciśnij Enter...")
            return
        
        # Wczytaj model
        network = NeuralNetwork(self.config)
        if not network.load(model_path):
            print("Błąd wczytywania modelu!")
            input("Naciśnij Enter...")
            return
        
        # Wybierz tor
        tracks_dir = os.path.join(self.base_dir, 'tracks')
        tracks = Track.list_tracks(tracks_dir)
        if not tracks:
            track = Track.create_simple_track()
        else:
            print("Dostępne tory:")
            for i, track_name in enumerate(tracks):
                print(f"{i + 1}. {track_name}")
            
            choice = input("Wybierz numer toru (Enter = 1): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(tracks):
                track = Track()
                track.load(tracks[int(choice) - 1], tracks_dir)
            else:
                track = Track()
                track.load(tracks[0], tracks_dir)
        
        # Utwórz samochód
        car = Car(track.start_position[0], track.start_position[1], track.start_angle, self.config)
        
        # Symulacja
        print("\nTestowanie modelu... (ESC aby zakończyć)")
        running = True
        
        while running:
            self.clock.tick(self.fps)
            dt = 1.0
            
            # Obsługa zdarzeń
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset
                        car.reset()
            
            if car.alive:
                # Aktualizuj czujniki
                car.update_sensors(track.walls)
                
                # Decyzja sieci
                outputs = network.forward(car.sensor_readings)
                
                # Aktualizuj samochód
                car.update(outputs, dt)
                
                # Kolizje
                car.check_collision(track.walls)
            
            # Rysuj
            self.screen.fill(self.bg_color)
            track.draw(self.screen)
            car.draw(self.screen, True)
            
            # UI
            y = 10
            text = self.font.render("TESTOWANIE", True, (0, 255, 0))
            self.screen.blit(text, (10, y))
            y += 40
            
            status_color = (0, 255, 0) if car.alive else (255, 0, 0)
            status_text = "Żywy" if car.alive else "Crash!"
            text = self.small_font.render(f"Status: {status_text}", True, status_color)
            self.screen.blit(text, (10, y))
            y += 30
            
            text = self.small_font.render(f"Prędkość: {car.speed:.2f}", True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 30
            
            text = self.small_font.render(f"Dystans: {car.distance_traveled:.2f}", True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 40
            
            text = self.small_font.render("R - Reset", True, (150, 150, 150))
            self.screen.blit(text, (10, y))
            
            pygame.display.flip()
    
    def _run_editor(self):
        """Uruchamia edytor torów"""
        from track_editor import TrackEditor
        
        editor = TrackEditor(self.config)
        editor.run()
    
    def _plot_training_progress(self, ga):
        """Rysuje wykres postępu treningu"""
        if not ga.best_fitness_history:
            return
        
        plt.figure(figsize=(10, 6))
        
        generations = range(1, len(ga.best_fitness_history) + 1)
        
        plt.plot(generations, ga.best_fitness_history, 'b-', label='Najlepszy', linewidth=2)
        plt.plot(generations, ga.avg_fitness_history, 'g--', label='Średni', linewidth=2)
        
        plt.xlabel('Generacja')
        plt.ylabel('Fitness')
        plt.title('Postęp Treningu - Algorytm Genetyczny')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('logs/training_progress.png')
        print("Wykres zapisany: logs/training_progress.png")
        plt.show()


def main():
    """Funkcja główna"""
    # Ustal bazowy katalog
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Upewnij się że katalogi istnieją
    os.makedirs(os.path.join(base_dir, 'tracks'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
    
    # Uruchom aplikację
    app = CarsGA()
    app.run()


if __name__ == "__main__":
    main()
