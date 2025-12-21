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
        self.mode = "wall"  # "wall", "checkpoint", "start", "pitstop"
        self.drawing = False
        self.start_point = None
        self.temp_line = None
        self.pitstop_rect_start = None  # Do rysowania prostokąta pitstopa
        self.hover_element = None  # Element pod kursorem (do podświetlenia przed usunięciem)
        
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
                        # Wczytaj tor - graficzny wybór
                        tracks_dir = os.path.join(self.base_dir, 'tracks')
                        tracks = Track.list_tracks(tracks_dir)
                        if tracks:
                            selected = self.show_track_selection(tracks)
                            if selected:
                                self.track = Track()
                                self.track.load(selected, tracks_dir)
                        else:
                            print("Brak zapisanych torów!")
                    
                    elif event.key == pygame.K_1:
                        self.mode = "wall"
                    elif event.key == pygame.K_2:
                        self.mode = "checkpoint"
                    elif event.key == pygame.K_3:
                        self.mode = "start"
                    elif event.key == pygame.K_4:
                        self.mode = "pitstop"
                    
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        # Usuń ostatni element
                        if self.mode == "wall" and self.track.walls:
                            self.track.walls.pop()
                        elif self.mode == "checkpoint" and self.track.checkpoints:
                            self.track.checkpoints.pop()
                        elif self.mode == "pitstop":
                            self.track.pitstop = None
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # LPM
                        pos = self.snap_to_grid(event.pos)
                        
                        if self.mode == "start":
                            # Ustaw pozycję startową
                            self.track.set_start_position(pos[0], pos[1], 0)
                        elif self.mode == "pitstop":
                            # Rozpocznij rysowanie prostokąta pitstopa
                            self.drawing = True
                            self.pitstop_rect_start = pos
                        else:
                            # Rozpocznij rysowanie linii
                            self.drawing = True
                            self.start_point = pos
                    
                    elif event.button == 3:  # PPM - usuń element
                        pos = event.pos
                        self.delete_element_at(pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.drawing:
                        # Zakończ rysowanie
                        end_point = self.snap_to_grid(event.pos)
                        
                        if self.mode == "pitstop":
                            # Utwórz prostokąt pitstopa
                            x1, y1 = self.pitstop_rect_start
                            x2, y2 = end_point
                            x = min(x1, x2)
                            y = min(y1, y2)
                            w = abs(x2 - x1)
                            h = abs(y2 - y1)
                            
                            if w > 0 and h > 0:
                                self.track.pitstop = {
                                    'zone': pygame.Rect(x, y, w, h),
                                    'checkpoint': None,  # Użytkownik doda osobno
                                    'refuel_time': 2.0
                                }
                            self.pitstop_rect_start = None
                        elif self.mode == "wall":
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
                
                    else:
                        # Podświetl element pod kursorem
                        self.hover_element = self.find_element_at(event.pos)
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
        
        # Podświetl element pod kursorem
        if self.hover_element and not self.drawing:
            self.draw_hover_highlight()
        
        # Tor
        self.track.draw(self.screen)
        
        # Tymczasowa linia lub prostokąt
        if self.drawing and self.temp_line:
            if self.mode == "pitstop" and self.pitstop_rect_start:
                # Rysuj tymczasowy prostokąt pitstopa
                x1, y1 = self.pitstop_rect_start
                x2, y2 = self.temp_line
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                pygame.draw.rect(self.screen, (255, 140, 0), pygame.Rect(x, y, w, h))
                pygame.draw.rect(self.screen, (255, 200, 100), pygame.Rect(x, y, w, h), 2)
            else:
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
            "start": "Start (3)",
            "pitstop": "Pitstop (4)"
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
        y += 30
        
        mode_color = (0, 255, 0) if self.mode == "pitstop" else (150, 150, 150)
        text = self.font.render(f"• {mode_text['pitstop']}", True, mode_color)
        self.screen.blit(text, (10, y))
        y += 50
        
        # Instrukcje
        instructions = [
            "G - Siatka",
            "C - Wyczyść",
            "S - Zapisz",
            "L - Wczytaj",
            "DEL - Usuń ostatni",
            "PPM - Usuń element",
            "ESC - Wyjdź"
        ]
        
        for instruction in instructions:
            text = self.font.render(instruction, True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 25
        
        # Statystyki toru
        y = self.height - 120
        stats = [
            f"Ścian: {len(self.track.walls)}",
            f"Checkpointów: {len(self.track.checkpoints)}",
            f"Start: {self.track.start_position}",
            f"Pitstop: {'TAK' if self.track.pitstop else 'NIE'}"
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, (150, 150, 255))
            self.screen.blit(text, (10, y))
            y += 25
    
    def find_element_at(self, pos):
        """
        Znajduje element najbliższy pozycji kursora
        
        Args:
            pos: Pozycja myszy (x, y)
            
        Returns:
            Tuple ('type', index) lub None
        """
        x, y = pos
        threshold = 10  # Maksymalna odległość w pikselach
        
        # Sprawdź ściany
        for i, wall in enumerate(self.track.walls):
            if self.point_to_line_distance(x, y, wall[0], wall[1], wall[2], wall[3]) < threshold:
                return ('wall', i)
        
        # Sprawdź checkpointy
        for i, checkpoint in enumerate(self.track.checkpoints):
            if self.point_to_line_distance(x, y, checkpoint[0], checkpoint[1], checkpoint[2], checkpoint[3]) < threshold:
                return ('checkpoint', i)
        
        # Sprawdź pitstop
        if self.track.pitstop and 'zone' in self.track.pitstop:
            if self.track.pitstop['zone'].collidepoint(pos):
                return ('pitstop', 0)
        
        return None
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """
        Oblicza odległość punktu od odcinka
        
        Args:
            px, py: Współrzędne punktu
            x1, y1, x2, y2: Współrzędne odcinka
            
        Returns:
            Odległość w pikselach
        """
        import math
        
        line_mag = math.hypot(x2 - x1, y2 - y1)
        if line_mag < 1e-6:
            return math.hypot(px - x1, py - y1)
        
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        u = max(0, min(1, u))
        
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        
        return math.hypot(px - ix, py - iy)
    
    def delete_element_at(self, pos):
        """
        Usuwa element w danej pozycji
        
        Args:
            pos: Pozycja myszy (x, y)
        """
        element = self.find_element_at(pos)
        
        if element:
            elem_type, index = element
            
            if elem_type == 'wall':
                self.track.walls.pop(index)
                print(f"Usunięto ścianę #{index}")
            elif elem_type == 'checkpoint':
                self.track.checkpoints.pop(index)
                print(f"Usunięto checkpoint #{index}")
            elif elem_type == 'pitstop':
                self.track.pitstop = None
                print("Usunięto pitstop")
            
            self.hover_element = None
    
    def draw_hover_highlight(self):
        """Podświetla element pod kursorem"""
        if not self.hover_element:
            return
        
        elem_type, index = self.hover_element
        
        if elem_type == 'wall':
            wall = self.track.walls[index]
            pygame.draw.line(self.screen, (255, 0, 0),
                           (wall[0], wall[1]), (wall[2], wall[3]), 5)
        elif elem_type == 'checkpoint':
            checkpoint = self.track.checkpoints[index]
            pygame.draw.line(self.screen, (255, 0, 0),
                           (checkpoint[0], checkpoint[1]), (checkpoint[2], checkpoint[3]), 4)
        elif elem_type == 'pitstop':
            if self.track.pitstop and 'zone' in self.track.pitstop:
                pygame.draw.rect(self.screen, (255, 0, 0), self.track.pitstop['zone'], 3)
    
    def show_track_selection(self, tracks):
        """
        Pokazuje graficzne menu wyboru toru
        
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
            title = self.large_font.render("WYBIERZ TOR DO WCZYTANIA", True, (100, 200, 255))
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 60))
            
            # Lista torów
            for i, track_name in enumerate(tracks):
                color = (200, 200, 80) if i == selected else (100, 100, 100)
                rect = pygame.Rect(self.width//2 - 300, 150 + i*60, 600, 50)
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 2, border_radius=8)
                
                text = self.font.render(track_name, True, (255, 255, 255))
                self.screen.blit(text, (rect.x + 20, rect.y + 13))
            
            # Instrukcje
            info = self.font.render("↑/↓: wybierz, Enter: zatwierdź, ESC: anuluj", True, (180, 180, 180))
            self.screen.blit(info, (self.width//2 - info.get_width()//2, 150 + len(tracks)*60 + 20))
            
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
