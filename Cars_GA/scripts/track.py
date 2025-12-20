"""
Klasa Track - reprezentuje tor wyścigowy z checkpointami
"""
import pygame
import json
import os


class Track:
    def __init__(self, name="default"):
        """
        Inicjalizacja toru
        
        Args:
            name: Nazwa toru
        """
        self.name = name
        self.walls = []  # Lista linii (x1, y1, x2, y2)
        self.checkpoints = []  # Lista checkpointów (x1, y1, x2, y2)
        self.start_position = (100, 100)  # Pozycja startowa
        self.start_angle = 0  # Kąt startowy
        
        # Kolory
        self.wall_color = (100, 100, 100)
        self.checkpoint_color = (0, 255, 0)
        self.start_color = (255, 255, 0)
    
    def add_wall(self, x1, y1, x2, y2):
        """Dodaje ścianę do toru"""
        self.walls.append((x1, y1, x2, y2))
    
    def add_checkpoint(self, x1, y1, x2, y2):
        """Dodaje checkpoint do toru"""
        self.checkpoints.append((x1, y1, x2, y2))
    
    def set_start_position(self, x, y, angle=0):
        """Ustawia pozycję startową"""
        self.start_position = (x, y)
        self.start_angle = angle
    
    def clear(self):
        """Czyści tor"""
        self.walls = []
        self.checkpoints = []
    
    def draw(self, screen):
        """
        Rysuje tor na ekranie
        
        Args:
            screen: Powierzchnia pygame
        """
        # Rysuj ściany
        for wall in self.walls:
            pygame.draw.line(screen, self.wall_color, 
                           (wall[0], wall[1]), (wall[2], wall[3]), 3)
        
        # Rysuj checkpointy
        for i, checkpoint in enumerate(self.checkpoints):
            pygame.draw.line(screen, self.checkpoint_color, 
                           (checkpoint[0], checkpoint[1]), 
                           (checkpoint[2], checkpoint[3]), 2)
            
            # Numer checkpointu
            font = pygame.font.Font(None, 24)
            text = font.render(str(i + 1), True, self.checkpoint_color)
            center_x = (checkpoint[0] + checkpoint[2]) / 2
            center_y = (checkpoint[1] + checkpoint[3]) / 2
            screen.blit(text, (center_x - 10, center_y - 10))
        
        # Rysuj pozycję startową
        pygame.draw.circle(screen, self.start_color, 
                         (int(self.start_position[0]), int(self.start_position[1])), 10)
        
        # Strzałka kierunku startowego
        import math
        angle_rad = math.radians(self.start_angle)
        end_x = self.start_position[0] + math.sin(angle_rad) * 30
        end_y = self.start_position[1] - math.cos(angle_rad) * 30
        pygame.draw.line(screen, self.start_color,
                       self.start_position, (int(end_x), int(end_y)), 3)
    
    def save(self, directory="tracks"):
        """
        Zapisuje tor do pliku JSON
        
        Args:
            directory: Katalog do zapisu
        """
        data = {
            'name': self.name,
            'walls': self.walls,
            'checkpoints': self.checkpoints,
            'start_position': self.start_position,
            'start_angle': self.start_angle
        }
        
        filepath = os.path.join(directory, f"{self.name}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tor zapisany: {filepath}")
    
    def load(self, name, directory="tracks"):
        """
        Wczytuje tor z pliku JSON
        
        Args:
            name: Nazwa toru
            directory: Katalog z torami
            
        Returns:
            True jeśli sukces, False w przeciwnym razie
        """
        filepath = os.path.join(directory, f"{name}.json")
        
        if not os.path.exists(filepath):
            print(f"Nie znaleziono toru: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.name = data.get('name', name)
            self.walls = [tuple(w) for w in data.get('walls', [])]
            self.checkpoints = [tuple(c) for c in data.get('checkpoints', [])]
            self.start_position = tuple(data.get('start_position', (100, 100)))
            self.start_angle = data.get('start_angle', 0)
            
            print(f"Wczytano tor: {self.name}")
            return True
        
        except Exception as e:
            print(f"Błąd wczytywania toru: {e}")
            return False
    
    @staticmethod
    def list_tracks(directory="tracks"):
        """
        Zwraca listę dostępnych torów
        
        Args:
            directory: Katalog z torami
            
        Returns:
            Lista nazw torów
        """
        if not os.path.exists(directory):
            return []
        
        tracks = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                tracks.append(filename[:-5])  # Usuń .json
        
        return sorted(tracks)
    
    @staticmethod
    def create_simple_track():
        """
        Tworzy prosty przykładowy tor
        
        Returns:
            Obiekt Track
        """
        track = Track("simple")
        
        # Prostokątny tor
        # Zewnętrzne ściany
        track.add_wall(100, 100, 700, 100)  # Góra
        track.add_wall(700, 100, 700, 500)  # Prawo
        track.add_wall(700, 500, 100, 500)  # Dół
        track.add_wall(100, 500, 100, 100)  # Lewo
        
        # Wewnętrzne ściany
        track.add_wall(200, 200, 600, 200)  # Góra wewnętrzna
        track.add_wall(600, 200, 600, 400)  # Prawo wewnętrzne
        track.add_wall(600, 400, 200, 400)  # Dół wewnętrzny
        track.add_wall(200, 400, 200, 200)  # Lewo wewnętrzne
        
        # Checkpointy
        track.add_checkpoint(100, 300, 200, 300)  # 1
        track.add_checkpoint(400, 100, 400, 200)  # 2
        track.add_checkpoint(600, 300, 700, 300)  # 3
        track.add_checkpoint(400, 400, 400, 500)  # 4
        
        # Pozycja startowa
        track.set_start_position(150, 300, 0)
        
        return track
    
    @staticmethod
    def create_oval_track():
        """
        Tworzy owalny tor
        
        Returns:
            Obiekt Track
        """
        track = Track("oval")
        
        import math
        
        # Zewnętrzny owal
        center_x, center_y = 400, 300
        outer_radius_x, outer_radius_y = 250, 150
        inner_radius_x, inner_radius_y = 150, 80
        
        num_segments = 32
        
        # Generuj zewnętrzne ściany
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi
            
            x1 = center_x + math.cos(angle1) * outer_radius_x
            y1 = center_y + math.sin(angle1) * outer_radius_y
            x2 = center_x + math.cos(angle2) * outer_radius_x
            y2 = center_y + math.sin(angle2) * outer_radius_y
            
            track.add_wall(x1, y1, x2, y2)
        
        # Generuj wewnętrzne ściany
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi
            
            x1 = center_x + math.cos(angle1) * inner_radius_x
            y1 = center_y + math.sin(angle1) * inner_radius_y
            x2 = center_x + math.cos(angle2) * inner_radius_x
            y2 = center_y + math.sin(angle2) * inner_radius_y
            
            track.add_wall(x1, y1, x2, y2)
        
        # Checkpointy
        checkpoint_angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        checkpoint_radius = (outer_radius_x + inner_radius_x) / 2
        
        for angle in checkpoint_angles:
            # Punkt na torze
            cx = center_x + math.cos(angle) * checkpoint_radius
            cy = center_y + math.sin(angle) * checkpoint_radius
            
            # Prostopadła linia
            perp_angle = angle + math.pi / 2
            dx = math.cos(perp_angle) * 30
            dy = math.sin(perp_angle) * 30
            
            track.add_checkpoint(cx - dx, cy - dy, cx + dx, cy + dy)
        
        # Pozycja startowa
        start_x = center_x - checkpoint_radius
        start_y = center_y
        track.set_start_position(start_x, start_y, 90)
        
        return track
    
    @staticmethod
    def create_zigzag_track():
        """
        Tworzy zygzakowaty tor
        
        Returns:
            Obiekt Track
        """
        track = Track("zigzag")
        
        # Parametry
        width = 100
        height = 600
        segment_height = 100
        offset = 150
        start_x = 200
        start_y = 100
        
        # Generuj zygzak
        points_left = []
        points_right = []
        
        y = start_y
        x_left = start_x
        x_right = start_x + width
        direction = 1
        
        for i in range(7):
            points_left.append((x_left, y))
            points_right.append((x_right, y))
            
            y += segment_height
            
            if i < 6:  # Nie przesuwaj ostatniego
                x_left += offset * direction
                x_right += offset * direction
                direction *= -1
        
        # Dodaj ściany
        for i in range(len(points_left) - 1):
            track.add_wall(points_left[i][0], points_left[i][1],
                         points_left[i+1][0], points_left[i+1][1])
            track.add_wall(points_right[i][0], points_right[i][1],
                         points_right[i+1][0], points_right[i+1][1])
        
        # Zamknij górę i dół
        track.add_wall(points_left[0][0], points_left[0][1],
                     points_right[0][0], points_right[0][1])
        track.add_wall(points_left[-1][0], points_left[-1][1],
                     points_right[-1][0], points_right[-1][1])
        
        # Checkpointy
        for i in range(1, len(points_left)):
            x = (points_left[i][0] + points_right[i][0]) / 2
            y = points_left[i][1]
            track.add_checkpoint(points_left[i][0], y, points_right[i][0], y)
        
        # Pozycja startowa
        track.set_start_position(start_x + width/2, start_y + 20, 0)
        
        return track
