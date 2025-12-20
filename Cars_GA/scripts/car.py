"""
Klasa Car - reprezentuje samochód z fizyką i czujnikami
"""
import pygame
import math
import numpy as np


class Car:
    def __init__(self, x, y, angle, config):
        """
        Inicjalizacja samochodu
        
        Args:
            x: Pozycja startowa X
            y: Pozycja startowa Y
            angle: Kąt startowy
            config: Słownik z konfiguracją
        """
        self.start_x = x
        self.start_y = y
        self.start_angle = angle
        self.x = x
        self.y = y
        self.angle = angle  # Kąt w stopniach
        
        # Parametry fizyczne
        self.width = config['car']['width']
        self.height = config['car']['height']
        self.max_speed = config['car']['max_speed']
        self.acceleration = config['car']['acceleration']
        self.friction = config['car']['friction']
        self.rotation_speed = config['car']['rotation_speed']
        
        # Próg aktywacji akcji
        self.action_threshold = config['car'].get('action_threshold', 0.3)
        
        # Stan ruchu (minimalna prędkość do przodu ułatwia start)
        self.initial_speed = config['car'].get('initial_speed', 1.0)
        self.speed = self.initial_speed
        self.velocity_x = 0
        self.velocity_y = 0
        
        # Czujniki
        self.sensor_range = config['car']['sensor_range']
        self.num_sensors = config['car']['num_sensors']
        self.sensor_readings = [1.0] * self.num_sensors  # 1.0 = nic nie wykryto
        
        # Status
        self.alive = True
        self.fitness = 0
        self.distance_traveled = 0
        self.checkpoints_passed = 0
        self.time_alive = 0
        
        # Kolory
        self.color = (0, 150, 255)
        self.sensor_color = (255, 100, 100, 100)
    
    def reset(self):
        """Resetuje samochód do pozycji startowej"""
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = self.initial_speed  # Minimalna prędkość startowa
        self.velocity_x = 0
        self.velocity_y = 0
        self.alive = True
        self.fitness = 0
        self.distance_traveled = 0
        self.checkpoints_passed = 0
        self.time_alive = 0
        self.sensor_readings = [1.0] * self.num_sensors
    
    def update(self, action, dt=1.0):
        """
        Aktualizuje stan samochodu na podstawie akcji
        
        Args:
            action: Lista [left, right, accelerate, brake] z wartościami 0-1
            dt: Delta time
        """
        if not self.alive:
            return
        
        self.time_alive += dt
        
        # Rozpakowanie akcji
        left, right, accelerate, brake = action
        
        # Kierowanie (zawsze możliwe, nawet przy małej prędkości)
        if left > self.action_threshold:
            self.angle -= self.rotation_speed * dt
        if right > self.action_threshold:
            self.angle += self.rotation_speed * dt
        
        # Przyspieszanie/hamowanie (obniżony próg dla łatwiejszej aktywacji)
        if accelerate > self.action_threshold:
            self.speed += self.acceleration * dt
        if brake > self.action_threshold:
            self.speed -= self.acceleration * dt * 1.5
        
        # Ograniczenie prędkości
        self.speed = max(-self.max_speed/2, min(self.max_speed, self.speed))
        
        # Tarcie
        self.speed *= self.friction
        
        # Oblicz prędkość w osiach X i Y
        angle_rad = math.radians(self.angle)
        self.velocity_x = math.sin(angle_rad) * self.speed
        self.velocity_y = -math.cos(angle_rad) * self.speed
        
        # Zapisz poprzednią pozycję
        old_x, old_y = self.x, self.y
        
        # Aktualizuj pozycję
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        
        # Oblicz przebytą odległość
        distance = math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        self.distance_traveled += distance
    
    def update_sensors(self, track_walls):
        """
        Aktualizuje odczyty czujników odległości
        
        Args:
            track_walls: Lista ścian toru (linii)
        """
        if not self.alive:
            return
        
        sensor_angles = self._get_sensor_angles()
        
        for i, sensor_angle in enumerate(sensor_angles):
            angle_rad = math.radians(sensor_angle)
            end_x = self.x + math.sin(angle_rad) * self.sensor_range
            end_y = self.y - math.cos(angle_rad) * self.sensor_range
            min_distance = self.sensor_range
            for wall in track_walls:
                intersection = self._line_intersection(
                    self.x, self.y, end_x, end_y,
                    wall[0], wall[1], wall[2], wall[3]
                )
                if intersection:
                    dist = math.sqrt(
                        (intersection[0] - self.x)**2 + 
                        (intersection[1] - self.y)**2
                    )
                    min_distance = min(min_distance, dist)
            self.sensor_readings[i] = min_distance / self.sensor_range

    def _get_sensor_angles(self):
        """
        Zwraca listę kątów (w stopniach) dla wszystkich czujników względem samochodu
        """
        angles = []
        if self.num_sensors >= 2:
            # Zawsze jeden do przodu i jeden do tyłu
            angles.append(self.angle)      # Przód
            angles.append(self.angle + 180) # Tył
            remaining = self.num_sensors - 2
            if remaining > 0:
                for i in range(remaining):
                    if remaining == 1:
                        angle_offset = -90
                    else:
                        angle_offset = (i / (remaining - 1) - 0.5) * 180
                    angles.append(self.angle + angle_offset)
        else:
            angles.append(self.angle)
        return angles
    
    def _line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Oblicza punkt przecięcia dwóch linii
        
        Returns:
            (x, y) jeśli linie się przecinają, None w przeciwnym razie
        """
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # t musi być > 0 (nie od startu lasera) i <= 1 (do końca lasera)
        # u musi być między 0 a 1 (na długości ściany)
        if 0 < t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def check_collision(self, track_walls):
        """
        Sprawdza kolizję samochodu ze ścianami
        
        Args:
            track_walls: Lista ścian toru
            
        Returns:
            True jeśli kolizja, False w przeciwnym razie
        """
        if not self.alive:
            return False
        
        # Punkty narożników samochodu
        corners = self._get_corners()
        
        # Sprawdź każdą krawędź samochodu
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            
            # Sprawdź kolizję z każdą ścianą
            for wall in track_walls:
                if self._line_intersection(x1, y1, x2, y2, wall[0], wall[1], wall[2], wall[3]):
                    self.alive = False
                    return True
        
        return False
    
    def check_checkpoint(self, checkpoint):
        """
        Sprawdza przejście przez checkpoint (zalicza, jeśli dowolny bok samochodu przecina checkpoint
        LUB dowolny róg samochodu jest w odległości <= 10px od linii checkpointu)
        
        Args:
            checkpoint: (x1, y1, x2, y2) linia checkpointu
        Returns:
            True jeśli przeszedł, False w przeciwnym razie
        """
        if not self.alive:
            return False
        
        # Jeśli checkpoint jest punktem (x1==x2 i y1==y2), sprawdź odległość
        if abs(checkpoint[0] - checkpoint[2]) < 1 and abs(checkpoint[1] - checkpoint[3]) < 1:
            dist = math.sqrt((self.x - checkpoint[0])**2 + (self.y - checkpoint[1])**2)
            return dist < 60  # Próg zaliczenia punktu
        
        # Sprawdź przecięcie checkpointu z dowolnym bokiem samochodu
        corners = self._get_corners()
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            if self._line_intersection(x1, y1, x2, y2, checkpoint[0], checkpoint[1], checkpoint[2], checkpoint[3]):
                return True
        # Sprawdź czy dowolny róg samochodu jest blisko linii checkpointu (<= 10px)
        def point_line_distance(px, py, x1, y1, x2, y2):
            # Odległość punktu od odcinka
            line_mag = math.hypot(x2 - x1, y2 - y1)
            if line_mag < 1e-6:
                return math.hypot(px - x1, py - y1)
            u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
            u = max(0, min(1, u))
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            return math.hypot(px - ix, py - iy)
        for cx, cy in corners:
            if point_line_distance(cx, cy, checkpoint[0], checkpoint[1], checkpoint[2], checkpoint[3]) <= 10:
                return True
        return False
    
    def _get_corners(self):
        """
        Oblicza pozycje narożników samochodu
        
        Returns:
            Lista 4 punktów (x, y)
        """
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Połowa wymiarów
        hw = self.width / 2
        hh = self.height / 2
        
        # 4 narożniki względem środka
        corners = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh)
        ]
        
        # Obróć i przesuń do pozycji samochodu
        rotated = []
        for dx, dy in corners:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rotated.append((self.x + rx, self.y + ry))
        
        return rotated
    
    def draw(self, screen, show_sensors=True):
        """
        Rysuje samochód na ekranie
        
        Args:
            screen: Powierzchnia pygame
            show_sensors: Czy rysować czujniki
        """
        if not self.alive:
            return
        
        # Rysuj czujniki
        if show_sensors:
            self._draw_sensors(screen)
        
        # Rysuj samochód
        corners = self._get_corners()
        pygame.draw.polygon(screen, self.color, corners)
        
        # Rysuj kierunek (mały trójkąt z przodu)
        angle_rad = math.radians(self.angle)
        front_x = self.x + math.sin(angle_rad) * self.height / 2
        front_y = self.y - math.cos(angle_rad) * self.height / 2
        
        # Mały znacznik kierunku
        pygame.draw.circle(screen, (255, 255, 255), (int(front_x), int(front_y)), 3)
    
    def _draw_sensors(self, screen):
        """Rysuje czujniki odległości"""
        sensor_angles = self._get_sensor_angles()
        for i, sensor_angle in enumerate(sensor_angles):
            angle_rad = math.radians(sensor_angle)
            distance = self.sensor_readings[i] * self.sensor_range
            end_x = self.x + math.sin(angle_rad) * distance
            end_y = self.y - math.cos(angle_rad) * distance
            color_intensity = int(self.sensor_readings[i] * 200)
            color = (200 - color_intensity, color_intensity, 0)
            pygame.draw.line(screen, color, 
                           (int(self.x), int(self.y)), 
                           (int(end_x), int(end_y)), 1)
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)
    
    def calculate_fitness(self, config):
        """
        Oblicza fitness samochodu
        
        Args:
            config: Słownik z konfiguracją
            
        Returns:
            Wartość fitness
        """
        fitness_config = config['fitness']
        
        fitness = 0
        
        # Punkty za checkpointy
        fitness += self.checkpoints_passed * fitness_config['checkpoint_reward']
        
        # Punkty za dystans
        fitness += self.distance_traveled * fitness_config['distance_weight']
        
        # Kara za czas
        fitness -= self.time_alive * fitness_config['time_penalty']
        
        # Kara za jazdę do tyłu
        reverse_penalty = fitness_config.get('reverse_penalty', 2.0)
        if self.speed < 0:
            fitness -= abs(self.speed) * reverse_penalty * self.time_alive
        
        # Kara za crash (jeśli nie żyje)
        if not self.alive:
            fitness += fitness_config['crash_penalty']
        
        self.fitness = max(0, fitness)  # Fitness nie może być ujemne
        return self.fitness
