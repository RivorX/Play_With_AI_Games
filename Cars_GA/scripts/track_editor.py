"""
Track Editor - Edytor torów wyścigowych
"""
import pygame
import sys
import os
from track import Track


class TrackEditor:
    def __init__(self, config):
        """
        Inicjalizacja edytora torów
        
        Args:
            config: Słownik z konfiguracją
        """
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Okno
        self.width = config['window']['width']
        self.height = config['window']['height']
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Edytor Torów - Cars GA")
        self.clock = pygame.time.Clock()
        self.fps = config['window']['fps']
        
        # Edytor
        self.track = Track("custom")
        self.mode = "wall"  # "wall", "checkpoint", "start"
        self.drawing = False
        self.start_point = None
        self.temp_line = None
        
        # Siatka
        self.grid_size = config['track_editor']['grid_size']
        self.show_grid = True
        
        # Kolory
        self.bg_color = (30, 30, 30)
        self.grid_color = (50, 50, 50)
        self.temp_line_color = (200, 200, 0)
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
    
    def snap_to_grid(self, pos):
        """
        Przyciąga punkt do siatki
        
        Args:
            pos: (x, y)
            
        Returns:
            Przyciągnięty punkt (x, y)
        """
        if self.show_grid:
            x = round(pos[0] / self.grid_size) * self.grid_size
            y = round(pos[1] / self.grid_size) * self.grid_size
            return (x, y)
        return pos
    
    def run(self):
        """Główna pętla edytora"""
        running = True
        
        while running:
            dt = self.clock.tick(self.fps) / 1000.0
            
            # Obsługa zdarzeń
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    elif event.key == pygame.K_g:
                        # Przełącz siatkę
                        self.show_grid = not self.show_grid
                    
                    elif event.key == pygame.K_c:
                        # Wyczyść tor
                        self.track.clear()
                    
                    elif event.key == pygame.K_s:
                        # Zapisz tor
                        name = input("Nazwa toru: ")
                        if name:
                            self.track.name = name
                            tracks_dir = os.path.join(self.base_dir, 'tracks')
                            os.makedirs(tracks_dir, exist_ok=True)
                            self.track.save(tracks_dir)
                    
                    elif event.key == pygame.K_l:
                        # Wczytaj tor
                        tracks_dir = os.path.join(self.base_dir, 'tracks')
                        tracks = Track.list_tracks(tracks_dir)
                        if tracks:
                            print("Dostępne tory:", tracks)
                            name = input("Nazwa toru do wczytania: ")
                            if name in tracks:
                                self.track.load(name, tracks_dir)
                    
                    elif event.key == pygame.K_1:
                        self.mode = "wall"
                    elif event.key == pygame.K_2:
                        self.mode = "checkpoint"
                    elif event.key == pygame.K_3:
                        self.mode = "start"
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # LPM
                        pos = self.snap_to_grid(event.pos)
                        
                        if self.mode == "start":
                            # Ustaw pozycję startową
                            self.track.set_start_position(pos[0], pos[1], 0)
                        else:
                            # Rozpocznij rysowanie linii
                            self.drawing = True
                            self.start_point = pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.drawing:
                        # Zakończ rysowanie linii
                        end_point = self.snap_to_grid(event.pos)
                        
                        if self.mode == "wall":
                            self.track.add_wall(
                                self.start_point[0], self.start_point[1],
                                end_point[0], end_point[1]
                            )
                        elif self.mode == "checkpoint":
                            self.track.add_checkpoint(
                                self.start_point[0], self.start_point[1],
                                end_point[0], end_point[1]
                            )
                        
                        self.drawing = False
                        self.start_point = None
                        self.temp_line = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        # Aktualizuj tymczasową linię
                        self.temp_line = self.snap_to_grid(event.pos)
            
            # Rysowanie
            self.draw()
        
        pygame.quit()
    
    def draw(self):
        """Rysuje edytor"""
        # Tło
        self.screen.fill(self.bg_color)
        
        # Siatka
        if self.show_grid:
            self.draw_grid()
        
        # Tor
        self.track.draw(self.screen)
        
        # Tymczasowa linia
        if self.drawing and self.temp_line:
            color = self.temp_line_color if self.mode == "wall" else (0, 255, 255)
            pygame.draw.line(self.screen, color,
                           self.start_point, self.temp_line, 2)
        
        # UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_grid(self):
        """Rysuje siatkę"""
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, self.grid_color,
                           (x, 0), (x, self.height), 1)
        
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, self.grid_color,
                           (0, y), (self.width, y), 1)
    
    def draw_ui(self):
        """Rysuje interfejs użytkownika"""
        y = 10
        
        # Tytuł
        title = self.large_font.render("EDYTOR TORÓW", True, (255, 255, 255))
        self.screen.blit(title, (10, y))
        y += 50
        
        # Tryb
        mode_text = {
            "wall": "Ściany (1)",
            "checkpoint": "Checkpointy (2)",
            "start": "Start (3)"
        }
        
        mode_color = (0, 255, 0) if self.mode == "wall" else (150, 150, 150)
        text = self.font.render(f"• {mode_text['wall']}", True, mode_color)
        self.screen.blit(text, (10, y))
        y += 30
        
        mode_color = (0, 255, 0) if self.mode == "checkpoint" else (150, 150, 150)
        text = self.font.render(f"• {mode_text['checkpoint']}", True, mode_color)
        self.screen.blit(text, (10, y))
        y += 30
        
        mode_color = (0, 255, 0) if self.mode == "start" else (150, 150, 150)
        text = self.font.render(f"• {mode_text['start']}", True, mode_color)
        self.screen.blit(text, (10, y))
        y += 50
        
        # Instrukcje
        instructions = [
            "G - Siatka",
            "C - Wyczyść",
            "S - Zapisz",
            "L - Wczytaj",
            "ESC - Wyjdź"
        ]
        
        for instruction in instructions:
            text = self.font.render(instruction, True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 25
        
        # Statystyki toru
        y = self.height - 100
        stats = [
            f"Ścian: {len(self.track.walls)}",
            f"Checkpointów: {len(self.track.checkpoints)}",
            f"Start: {self.track.start_position}"
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, (150, 150, 255))
            self.screen.blit(text, (10, y))
            y += 25


def main():
    """Funkcja główna edytora"""
    import yaml
    
    # Ustal bazowy katalog projektu
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Wczytaj konfigurację
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Uruchom edytor
    editor = TrackEditor(config)
    editor.run()


if __name__ == "__main__":
    main()
