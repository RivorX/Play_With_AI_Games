"""
Główna aplikacja Cars-GA z menu i różnymi trybami
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import sys
import yaml
import signal
from track import Track
from car import Car
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np


# Globalny handler dla Ctrl+C - natychmiast wychodzi
def _exit_handler(signum, frame):
    print("\n\n=== Program przerwany (Ctrl+C) ===", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, _exit_handler)


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
        """Tryb trenowania AI z graficznym wyborem mapy (z kategoriami)"""
        tracks_dir = os.path.join(self.base_dir, 'tracks')
        categories = {}
        # Zbierz mapy z podkatalogów jako kategorie
        for entry in os.listdir(tracks_dir):
            full_path = os.path.join(tracks_dir, entry)
            if os.path.isdir(full_path):
                maps = Track.list_tracks(full_path)
                if maps:
                    categories[entry] = maps
        # Mapy z głównego katalogu (bez podkategorii)
        root_maps = [name for name in Track.list_tracks(tracks_dir) if name not in categories]
        if root_maps:
            categories['Inne'] = root_maps
        # Przekaż kategorie do wyboru
        selected_category, selected_map = self._show_track_selection_with_categories(categories, "WYBÓR MAPY DO TRENINGU")
        if selected_category is None or selected_map is None:
            return
        # Załaduj wybrany tor
        if selected_category == 'Inne':
            track = Track()
            track.load(selected_map, tracks_dir)
        else:
            track = Track()
            track.load(selected_map, os.path.join(tracks_dir, selected_category))

        # Inicjalizuj algorytm genetyczny
        ga = GeneticAlgorithm(self.config, track)

        # Parametry treningu
        max_generations = self.config['genetic_algorithm']['generations']
        save_interval = self.config['training']['save_interval']
        show_best = self.config['training']['show_best']
        show_sensors = self.config['training']['show_sensors']
        visualize_interval = self.config['training'].get('visualize_interval', 10)
        fast_mode = self.config['training'].get('fast_mode', False)

        print(f"\nRozpoczynanie treningu na {max_generations} generacji...")
        print(f"Rozmiar populacji: {ga.population_size}")
        if fast_mode:
            print("TRYB SZYBKI - wizualizacja wyłączona")
        else:
            print(f"Wizualizacja co {visualize_interval} generacji")

        # Pętla treningowa
        running = True
        generation = 0

        try:
            while running and generation < max_generations:
                # Sprawdź zdarzenia tylko jeśli nie w trybie szybkim
                if not fast_mode:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            elif event.key == pygame.K_v:
                                # Przełącz wizualizację
                                fast_mode = not fast_mode
                                print(f"\nTryb szybki: {fast_mode}")

                # Wizualizacja lub szybka symulacja
                if not fast_mode and visualize_interval > 0 and generation > 0 and (generation % visualize_interval == 0 or generation == max_generations - 1):
                    self._visualize_generation(ga, track, show_sensors)
                else:
                    ga.evaluate_population()

                ga.evolve()
                generation = ga.generation

                stats = ga.get_statistics()
                print(f"\nGeneracja {stats['generation']}")
                print(f"  Najlepszy fitness: {stats['best_fitness']:.2f}")
                print(f"  Średni fitness: {stats['avg_fitness']:.2f}")
                print(f"  Najlepiej checkpointów: {stats['best_checkpoints']} (średnio: {stats['avg_checkpoints']:.2f})")

                if generation % save_interval == 0:
                    models_dir = os.path.join(self.base_dir, 'models')
                    os.makedirs(models_dir, exist_ok=True)
                    model_path = os.path.join(models_dir, f"generation_{generation}.json")
                    ga.save_best(model_path)
                    # Zapisz wykresy
                    ga.save_training_plots(f"training_plot_gen_{generation}")
                    print(f"  Model i wykresy zapisane (generacja {generation})")

        except KeyboardInterrupt:
            print("\n\n=== Trening przerwany przez użytkownika (Ctrl+C) ===")
            import sys
            sys.exit(0)

        if running:
            models_dir = os.path.join(self.base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'best_model.json')
            ga.save_best(model_path)
            # Zapisz końcowe wykresy
            ga.save_training_plots("training_plot_final")
            print("\n=== Trening zakończony! ===")
            print(f"Najlepszy fitness: {ga.best_fitness:.2f}")
            print(f"Wykresy zapisane w katalogu logs/")
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
    
    def _show_track_selection_with_categories(self, categories, title_text):
        """Graficzny wybór mapy z podziałem na kategorie (katalogi) + obsługa myszy"""
        category_names = list(categories.keys())
        selected_cat = 0
        selected_map = 0
        choosing = True
        # Bufory prostokątów do obsługi kliknięć
        cat_rects = []
        map_rects = []
        while choosing:
            self.screen.fill(self.bg_color)
            cat_rects.clear()
            map_rects.clear()
            # Tytuł
            title = self.large_font.render(title_text, True, (100, 200, 255))
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 40))
            # Kategorie
            cat_y = 120
            for i, cat in enumerate(category_names):
                color = (200, 200, 80) if i == selected_cat else (80, 80, 100)
                rect = pygame.Rect(self.width//2 - 350, cat_y + i*45, 200, 40)
                cat_rects.append(rect)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 2, border_radius=8)
                text = self.font.render(cat, True, (255, 255, 255))
                self.screen.blit(text, (rect.x + 20, rect.y + 7))
            # Mapy w wybranej kategorii
            maps = categories[category_names[selected_cat]]
            map_y = 120
            for j, map_name in enumerate(maps):
                color = (200, 200, 80) if j == selected_map else (80, 80, 100)
                rect = pygame.Rect(self.width//2 - 100, map_y + j*45, 400, 40)
                map_rects.append(rect)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 2, border_radius=8)
                text = self.font.render(map_name, True, (255, 255, 255))
                self.screen.blit(text, (rect.x + 20, rect.y + 7))
            # Instrukcje
            instructions = self.small_font.render("↑/↓: kategoria, ←/→: mapa, Enter: wybierz, ESC: anuluj, MYSZ: kliknij", True, (180, 180, 180))
            self.screen.blit(instructions, (self.width//2 - instructions.get_width()//2, self.height - 60))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None, None
                    elif event.key == pygame.K_UP:
                        selected_cat = (selected_cat - 1) % len(category_names)
                        selected_map = 0
                    elif event.key == pygame.K_DOWN:
                        selected_cat = (selected_cat + 1) % len(category_names)
                        selected_map = 0
                    elif event.key == pygame.K_LEFT:
                        selected_map = (selected_map - 1) % len(categories[category_names[selected_cat]])
                    elif event.key == pygame.K_RIGHT:
                        selected_map = (selected_map + 1) % len(categories[category_names[selected_cat]])
                    elif event.key == pygame.K_RETURN:
                        return category_names[selected_cat], categories[category_names[selected_cat]][selected_map]
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    # Klik na kategorię
                    for i, rect in enumerate(cat_rects):
                        if rect.collidepoint(mouse_pos):
                            selected_cat = i
                            selected_map = 0
                            break
                    # Klik na mapę
                    for j, rect in enumerate(map_rects):
                        if rect.collidepoint(mouse_pos):
                            selected_map = j
                            return category_names[selected_cat], categories[category_names[selected_cat]][selected_map]
        return None, None

        # Inicjalizuj algorytm genetyczny
        ga = GeneticAlgorithm(self.config, track)

        # Parametry treningu
        max_generations = self.config['genetic_algorithm']['generations']
        save_interval = self.config['training']['save_interval']
        show_best = self.config['training']['show_best']
        show_sensors = self.config['training']['show_sensors']
        visualize_interval = self.config['training'].get('visualize_interval', 10)
        fast_mode = self.config['training'].get('fast_mode', False)

        print(f"\nRozpoczynanie treningu na {max_generations} generacji...")
        print(f"Rozmiar populacji: {ga.population_size}")
        if fast_mode:
            print("TRYB SZYBKI - wizualizacja wyłączona")
        else:
            print(f"Wizualizacja co {visualize_interval} generacji")

        # Pętla treningowa
        running = True
        generation = 0

        try:
            while running and generation < max_generations:
                # Sprawdź zdarzenia tylko jeśli nie w trybie szybkim
                if not fast_mode:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            elif event.key == pygame.K_v:
                                # Przełącz wizualizację
                                fast_mode = not fast_mode
                                print(f"\nTryb szybki: {fast_mode}")

                # Wizualizacja lub szybka symulacja
                if not fast_mode and visualize_interval > 0 and generation > 0 and (generation % visualize_interval == 0 or generation == max_generations - 1):
                    self._visualize_generation(ga, track, show_sensors)
                else:
                    ga.evaluate_population()

                ga.evolve()
                generation = ga.generation

                stats = ga.get_statistics()
                print(f"\nGeneracja {stats['generation']}")
                print(f"  Najlepszy fitness: {stats['best_fitness']:.2f}")
                print(f"  Średni fitness: {stats['avg_fitness']:.2f}")
                print(f"  Najlepiej checkpointów: {stats['best_checkpoints']} (średnio: {stats['avg_checkpoints']:.2f})")

                if generation % save_interval == 0:
                    models_dir = os.path.join(self.base_dir, 'models')
                    os.makedirs(models_dir, exist_ok=True)
                    model_path = os.path.join(models_dir, f"generation_{generation}.json")
                    ga.save_best(model_path)
                    # Zapisz wykresy
                    ga.save_training_plots(f"training_plot_gen_{generation}")
                    print(f"  Model i wykresy zapisane (generacja {generation})")

        except KeyboardInterrupt:
            print("\n\n=== Trening przerwany przez użytkownika (Ctrl+C) ===")
            running = False

        if running:
            models_dir = os.path.join(self.base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'best_model.json')
            ga.save_best(model_path)
            # Zapisz końcowe wykresy
            ga.save_training_plots("training_plot_final")
            print("\n=== Trening zakończony! ===")
            print(f"Najlepszy fitness: {ga.best_fitness:.2f}")
            print(f"Wykresy zapisane w katalogu logs/")
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
                
                # Aktualizuj kierunek do checkpointu
                checkpoint_idx = individual['checkpoint_index']
                if checkpoint_idx < len(track.checkpoints):
                    car.update_checkpoint_direction(track.checkpoints[checkpoint_idx])
                
                # Decyzja sieci (użyj wszystkich sensorów)
                outputs = network.forward(car.get_sensor_inputs())
                
                # Aktualizuj
                car.update(outputs, dt)
                
                # Kolizje
                if car.check_collision(track.walls):
                    car.alive = False
                    continue
                
                # Sprawdź czy samochód jest w pitstopie
                if track.pitstop and 'zone' in track.pitstop:
                    if track.pitstop['zone'].collidepoint(car.x, car.y):
                        car.in_pitstop = True
                        car.service_in_pitstop(dt)
                    else:
                        car.in_pitstop = False
                        car.pitstop_time = 0
                
                # Checkpointy - checkpoint 0 liczy się tylko po przejściu 1,2,3
                checkpoint_idx = individual['checkpoint_index']
                
                # Normalny checkpoint
                if checkpoint_idx < len(track.checkpoints):
                    current_checkpoint = track.checkpoints[checkpoint_idx]
                    
                    # Jeśli to checkpoint 0, sprawdź czy przejechał już 1,2,3
                    if checkpoint_idx == 0:
                        if car.completed_lap:
                            if car.check_checkpoint(current_checkpoint):
                                car.checkpoints_passed += 1
                                individual['checkpoint_index'] += 1
                                car.completed_lap = False  # Reset dla kolejnego okrążenia
                    else:
                        if car.check_checkpoint(current_checkpoint):
                            car.checkpoints_passed += 1
                            individual['checkpoint_index'] += 1
                            # Jeśli przeszedł checkpoint 3 (ostatni przed powrotem do 0), oznacz jako completed_lap
                            if checkpoint_idx == len(track.checkpoints) - 1:
                                car.completed_lap = True
                                individual['checkpoint_index'] = 0  # Wróć do pierwszego checkpointu
            
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
        
        # Znajdź wszystkie dostępne modele
        models_dir = os.path.join(self.base_dir, 'models')
        if not os.path.exists(models_dir):
            print("Brak katalogu models! Najpierw wytrenuj AI.")
            self._wait_for_key("Naciśnij dowolny klawisz...")
            return
        
        # Lista wszystkich plików .json w katalogu models
        model_files = [f[:-5] for f in os.listdir(models_dir) if f.endswith('.json')]
        
        if not model_files:
            print("Brak wytrenowanych modeli! Najpierw wytrenuj AI.")
            self._wait_for_key("Naciśnij dowolny klawisz...")
            return
        
        # Sortuj modele (best_model na początku, potem generation_X)
        def sort_key(name):
            if name == 'best_model':
                return (0, 0)
            elif name.startswith('generation_'):
                try:
                    gen_num = int(name.split('_')[1])
                    return (1, gen_num)
                except:
                    return (2, name)
            return (2, name)
        
        model_files.sort(key=sort_key)
        
        # Graficzny wybór modelu
        selected_model = self._show_model_selection(model_files)
        if not selected_model:
            return
        
        model_path = os.path.join(models_dir, f"{selected_model}.json")
        
        # Wczytaj model
        network = NeuralNetwork(self.config)
        if not network.load(model_path):
            print("Błąd wczytywania modelu!")
            self._wait_for_key("Naciśnij dowolny klawisz...")
            return
        
        # Wybierz tor (graficznie, z kategoriami)
        tracks_dir = os.path.join(self.base_dir, 'tracks')
        categories = {}
        for entry in os.listdir(tracks_dir):
            full_path = os.path.join(tracks_dir, entry)
            if os.path.isdir(full_path):
                maps = Track.list_tracks(full_path)
                if maps:
                    categories[entry] = maps
        root_maps = [name for name in Track.list_tracks(tracks_dir) if name not in categories]
        if root_maps:
            categories['Inne'] = root_maps
        selected_category, selected_map = self._show_track_selection_with_categories(categories, "WYBÓR MAPY DO TESTOWANIA")
        if selected_category is None or selected_map is None:
            return
        if selected_category == 'Inne':
            track = Track()
            track.load(selected_map, tracks_dir)
        else:
            track = Track()
            track.load(selected_map, os.path.join(tracks_dir, selected_category))
        
        # Utwórz samochód
        car = Car(track.start_position[0], track.start_position[1], track.start_angle, self.config)
        
        # Symulacja
        print("Testowanie modelu... (ESC aby zakończyć)")
        running = True
        checkpoint_index = 0
        dt = 1.0  # Użyj tego samego dt co podczas treningu!
        
        try:
            while running:
                self.clock.tick(self.fps)
                
                # Obsługa zdarzeń
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            # Reset
                            car.x = track.start_position[0]
                            car.y = track.start_position[1]
                            car.angle = track.start_angle
                            car.alive = True
                            car.checkpoints_passed = 0
                            checkpoint_index = 0
                
                if car.alive:
                    # Aktualizuj czujniki
                    car.update_sensors(track.walls)
                    
                    # Aktualizuj kierunek do checkpointu
                    if len(track.checkpoints) > 0:
                        car.update_checkpoint_direction(track.checkpoints[checkpoint_index % len(track.checkpoints)])
                    
                    # Decyzja sieci (użyj wszystkich sensorów)
                    outputs = network.forward(car.get_sensor_inputs())
                    
                    # Konwertuj outputs na listę jeśli potrzeba
                    if hasattr(outputs, 'tolist'):
                        outputs = outputs.tolist()
                    
                    # Aktualizuj samochód
                    car.update(outputs, dt)
                    
                    # Kolizje
                    if car.check_collision(track.walls):
                        car.alive = False
                    
                    # Sprawdź checkpointy (cyklicznie)
                    if len(track.checkpoints) > 0:
                        checkpoint = track.checkpoints[checkpoint_index % len(track.checkpoints)]
                        if car.check_checkpoint(checkpoint):
                            car.checkpoints_passed += 1
                            checkpoint_index = (checkpoint_index + 1) % len(track.checkpoints)
                
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
                y += 30
                
                text = self.small_font.render(f"Checkpointy: {car.checkpoints_passed}", True, (200, 200, 200))
                self.screen.blit(text, (10, y))
                y += 30
                
                # Kierunek do checkpointu
                checkpoint_dir_text = "Lewo" if car.checkpoint_direction < -0.2 else "Prawo" if car.checkpoint_direction > 0.2 else "Prosto"
                text = self.small_font.render(f"Checkpoint: {checkpoint_dir_text} ({car.checkpoint_direction:.2f})", True, (0, 200, 255))
                self.screen.blit(text, (10, y))
                y += 40
                
                text = self.small_font.render("R - Reset | ESC - Wyjście", True, (150, 150, 150))
                self.screen.blit(text, (10, y))
                
                pygame.display.flip()
        
        except KeyboardInterrupt:
            print("\n\n=== Testowanie przerwane przez użytkownika (Ctrl+C) ===")
            import sys
            sys.exit(0)
    
    def _show_model_selection(self, models):
        """
        Pokazuje graficzne menu wyboru modelu z obsługą myszy
        
        Args:
            models: Lista nazw modeli (bez .json)
            
        Returns:
            Wybrana nazwa modelu lub None
        """
        selected = 0
        choosing = True
        model_rects = {}  # Mapowanie prostokątów na indeksy
        
        while choosing:
            self.screen.fill(self.bg_color)
            model_rects.clear()  # Wyczyść mapy prostokątów
            
            # Tytuł
            title = self.large_font.render("WYBIERZ MODEL DO TESTOWANIA", True, (100, 200, 255))
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 40))
            
            # Informacja
            info_text = self.small_font.render(f"Dostępne modele: {len(models)}", True, (180, 180, 180))
            self.screen.blit(info_text, (self.width//2 - info_text.get_width()//2, 90))
            
            # Lista modeli (max 12 na raz, z scrollowaniem)
            visible_start = max(0, selected - 5)
            visible_end = min(len(models), visible_start + 12)
            
            for i in range(visible_start, visible_end):
                model_name = models[i]
                display_idx = i - visible_start
                color = (200, 200, 80) if i == selected else (80, 80, 100)
                rect = pygame.Rect(self.width//2 - 350, 130 + display_idx*50, 700, 45)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 2, border_radius=8)
                model_rects[id(rect)] = i  # Mapuj rect na indeks
                
                # Nazwa modelu
                text = self.font.render(model_name, True, (255, 255, 255))
                self.screen.blit(text, (rect.x + 20, rect.y + 10))
                
                # Przechowaj rect dla obsługi klików
                model_rects[i] = rect
            
            # Instrukcje
            instructions = self.small_font.render("↑/↓: wybierz, Enter: zatwierdź, Myszka: kliknij, ESC: anuluj", True, (180, 180, 180))
            self.screen.blit(instructions, (self.width//2 - instructions.get_width()//2, self.height - 60))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP:
                        selected = (selected - 1) % len(models)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(models)
                    elif event.key == pygame.K_RETURN:
                        return models[selected]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Obsługa kliknięć myszy
                    if event.button == 1:  # Lewy przycisk
                        mouse_pos = pygame.mouse.get_pos()
                        for idx, rect in model_rects.items():
                            if isinstance(rect, pygame.Rect) and rect.collidepoint(mouse_pos):
                                return models[idx]
        
        return None
    
    def _show_track_selection_simple(self, tracks):
        """
        Pokazuje graficzne menu wyboru toru (wersja uproszczona)
        
        Args:
            tracks: Lista nazw torów
            
        Returns:
            Wybrana nazwa toru lub None
        """
        selected = 0
        choosing = True
        
        while choosing:
            self.screen.fill(self.bg_color)
            
            # Tytuł
            title = self.large_font.render("WYBIERZ TOR", True, (100, 200, 255))
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 60))
            
            # Lista torów
            for i, track_name in enumerate(tracks):
                color = (200, 200, 80) if i == selected else (80, 80, 100)
                rect = pygame.Rect(self.width//2 - 300, 150 + i*60, 600, 50)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 2, border_radius=8)
                
                text = self.font.render(track_name, True, (255, 255, 255))
                self.screen.blit(text, (rect.x + 20, rect.y + 13))
            
            # Instrukcje
            instructions = self.small_font.render("↑/↓: wybierz, Enter: zatwierdź, ESC: anuluj", True, (180, 180, 180))
            self.screen.blit(instructions, (self.width//2 - instructions.get_width()//2, 150 + len(tracks)*60 + 20))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP:
                        selected = (selected - 1) % len(tracks)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(tracks)
                    elif event.key == pygame.K_RETURN:
                        return tracks[selected]
        
        return None
    
    def _wait_for_key(self, message):
        """Czeka na naciśnięcie klawisza z komunikatem"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            
            self.screen.fill(self.bg_color)
            text = self.font.render(message, True, (180, 180, 180))
            self.screen.blit(text, (self.width//2 - text.get_width()//2, self.height//2))
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
