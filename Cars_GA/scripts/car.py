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
        self.checkpoint_direction = 0.0  # Kierunek do najbliższego checkpointu (-1 do 1)
        
        # Wiatr
        self.wind_strength = config['car'].get('wind_strength', 0.3)
        self.wind_change_rate = config['car'].get('wind_change_rate', 0.05)
        self.wind_side_strength = config['car'].get('wind_side_strength', 0.2)
        self.wind_x = 0.0  # Siła wiatru w osi X
        self.wind_y = 0.0  # Siła wiatru w osi Y
        self.wind_angle = 0.0  # Aktualny kąt wiatru
        
        # Degradacja opon
        self.tire_degradation_rate = config['car'].get('tire_degradation_rate', 0.0008)
        self.base_tire_grip = config['car'].get('base_tire_grip', 1.0)
        self.min_tire_grip = config['car'].get('min_tire_grip', 0.5)
        self.tire_grip = self.base_tire_grip  # Aktualna przyczepność
        
        # Paliwo
        self.fuel_capacity = config['car'].get('fuel_capacity', 100)
        self.fuel_base_consumption = config['car'].get('fuel_base_consumption', 0.03)
        self.fuel_speed_factor = config['car'].get('fuel_speed_factor', 0.008)
        self.fuel_effect_on_speed = config['car'].get('fuel_effect_on_speed', 0.8)
        self.fuel = self.fuel_capacity  # Paliwo
        
        # Status
        self.alive = True
        self.fitness = 0
        self.distance_traveled = 0
        self.checkpoints_passed = 0
        self.time_alive = 0
        self.completed_lap = False  # Czy przejechał pełne okrążenie (wszystkie checkpointy poza pierwszym)
        self.in_pitstop = False  # Czy jest w pitstopie
        self.pitstop_time = 0  # Czas spędzony w pitstopie
        
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
        self.completed_lap = False
        self.in_pitstop = False
        self.pitstop_time = 0
        self.sensor_readings = [1.0] * self.num_sensors
        self.checkpoint_direction = 0.0
        
        # Resetuj wiatr, opony i paliwo
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.wind_angle = 0.0
        self.tire_grip = self.base_tire_grip
        self.fuel = self.fuel_capacity
    
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
        
        # === WIATR ===
        # Losowo zmień kierunek wiatru
        self.wind_angle += (np.random.randn() * self.wind_change_rate)
        wind_magnitude = np.random.uniform(0, self.wind_strength)
        self.wind_x = np.cos(self.wind_angle) * wind_magnitude
        
        # Dodaj boczny wiatr (prostopadle do kierunku jazdy)
        side_wind = np.random.randn() * self.wind_side_strength
        angle_rad = math.radians(self.angle)
        self.wind_y = np.sin(angle_rad) * side_wind
        
        # === DEGRADACJA OPON ===
        # Opony degradują się w zależności od prędkości i obrotu
        tire_stress = abs(self.speed) * 0.1 + abs(self.velocity_x + self.velocity_y) * 0.05
        self.tire_grip = max(self.min_tire_grip, 
                            self.tire_grip - self.tire_degradation_rate * tire_stress)
        
        # === PALIWO ===
        # Zużywaj paliwo: stały pobór + pobór zależny od prędkości
        # Stały pobór (silnik włączony) + pobór proporcjonalny do prędkości
        fuel_burn = (self.fuel_base_consumption + abs(self.speed) * self.fuel_speed_factor) * dt
        self.fuel = max(0, self.fuel - fuel_burn)
        
        # Jeśli nie ma paliwa, samochód nie żyje
        if self.fuel <= 0:
            self.alive = False
            return
        
        # Wpływ paliwa na maksymalną prędkość
        fuel_ratio = self.fuel / self.fuel_capacity
        effective_max_speed = self.max_speed * (
            self.fuel_effect_on_speed + (1 - self.fuel_effect_on_speed) * fuel_ratio
        )
        
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
        
        # Ograniczenie prędkości (uwzględniaj paliwo)
        self.speed = max(-effective_max_speed/2, min(effective_max_speed, self.speed))
        
        # Tarcie (wpływ degradacji opon)
        self.speed *= (self.friction * self.tire_grip)
        
        # Oblicz prędkość w osiach X i Y
        angle_rad = math.radians(self.angle)
        self.velocity_x = math.sin(angle_rad) * self.speed
        self.velocity_y = -math.cos(angle_rad) * self.speed
        
        # Dodaj wpływ wiatru
        self.velocity_x += self.wind_x * (1 - self.tire_grip)  # Wiatr bardziej wpływa na zmęczone opony
        self.velocity_y += self.wind_y * (1 - self.tire_grip)
        
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
        
        # Zmień kolor samochodu w zależności od stanu
        if self.in_pitstop:
            car_color = (255, 140, 0)  # Pomarańczowy - w pitstopie
        elif self.fuel > self.fuel_capacity * 0.5:
            car_color = (0, 150, 255)  # Niebieski - pełne paliwo
        elif self.fuel > self.fuel_capacity * 0.2:
            car_color = (255, 200, 0)  # Żółty - mało paliwa
        else:
            car_color = (255, 50, 50)  # Czerwony - krytycznie mało paliwa
        
        # Zmień kolor w zależności od degradacji opon
        if self.tire_grip < 0.7 and not self.in_pitstop:
            car_color = tuple(max(0, c - 50) for c in car_color)  # Ciemniej
        
        # Rysuj samochód
        corners = self._get_corners()
        pygame.draw.polygon(screen, car_color, corners)
        
        # Rysuj kierunek (mały trójkąt z przodu)
        angle_rad = math.radians(self.angle)
        front_x = self.x + math.sin(angle_rad) * self.height / 2
        front_y = self.y - math.cos(angle_rad) * self.height / 2
        
        # Mały znacznik kierunku
        pygame.draw.circle(screen, (255, 255, 255), (int(front_x), int(front_y)), 3)
        
        # Rysuj pasek paliwa nad samochodem
        self._draw_fuel_bar(screen)
        
        # Rysuj wskaźnik degradacji opon
        self._draw_tire_indicator(screen)
    
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
    
    def _draw_fuel_bar(self, screen):
        """Rysuje pasek paliwa nad samochodem"""
        bar_width = 30
        bar_height = 5
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.height / 2 - 15
        
        # Tło paska
        pygame.draw.rect(screen, (50, 50, 50), 
                        (int(bar_x), int(bar_y), int(bar_width), int(bar_height)))
        
        # Pasek paliwa
        fuel_ratio = max(0, self.fuel / self.fuel_capacity)
        if fuel_ratio > 0.5:
            color = (0, 200, 0)  # Zielony
        elif fuel_ratio > 0.2:
            color = (255, 200, 0)  # Żółty
        else:
            color = (255, 50, 50)  # Czerwony
        
        pygame.draw.rect(screen, color, 
                        (int(bar_x), int(bar_y), int(bar_width * fuel_ratio), int(bar_height)))
    
    def _draw_tire_indicator(self, screen):
        """Rysuje wskaźnik degradacji opon"""
        indicator_x = self.x - 8
        indicator_y = self.y - self.height / 2 - 25
        indicator_size = 4
        
        # Kolor wskaźnika w zależności od stanu opon
        if self.tire_grip > 0.9:
            color = (0, 255, 0)  # Zielony
        elif self.tire_grip > 0.7:
            color = (255, 200, 0)  # Żółty
        else:
            color = (255, 50, 50)  # Czerwony
        
        pygame.draw.circle(screen, color, (int(indicator_x), int(indicator_y)), indicator_size)
    
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
        
        # Kara za brak paliwa (przedwczesna śmierć)
        if not self.alive and self.fuel <= 0:
            fitness -= fitness_config.get('crash_penalty', -50) * 0.5  # Mniejsza kara niż przy zderzeniu
        
        # Kara za crash (jeśli nie żyje i to nie z powodu paliwa)
        if not self.alive and self.fuel > 0:
            fitness += fitness_config['crash_penalty']
        
        # Bonus za konserwację pojazdów (dobre opony = lepsze wyniki)
        tire_efficiency = (self.tire_grip - self.min_tire_grip) / (self.base_tire_grip - self.min_tire_grip)
        fitness += tire_efficiency * self.distance_traveled * 0.05  # Bonus za dobre opony
        
        # Bonus za oszczędzanie paliwa
        fuel_efficiency = (self.fuel / self.fuel_capacity)
        fitness += fuel_efficiency * self.distance_traveled * 0.03  # Bonus za pozostałe paliwo
        
        self.fitness = max(0, fitness)  # Fitness nie może być ujemne
        return self.fitness
    
    def update_checkpoint_direction(self, checkpoint):
        """
        Aktualizuje kierunek do najbliższego checkpointu
        
        Args:
            checkpoint: Współrzędne checkpointu (x1, y1, x2, y2)
        """
        if not self.alive or not checkpoint:
            self.checkpoint_direction = 0.0
            return
        
        # Środek checkpointu
        checkpoint_x = (checkpoint[0] + checkpoint[2]) / 2
        checkpoint_y = (checkpoint[1] + checkpoint[3]) / 2
        
        # Wektor do checkpointu
        dx = checkpoint_x - self.x
        dy = checkpoint_y - self.y
        
        # Kąt do checkpointu w radianach
        angle_to_checkpoint = math.atan2(dy, dx)
        
        # Aktualny kąt samochodu w radianach (0 stopni = góra, więc dodajemy 90 stopni)
        car_angle_rad = math.radians(self.angle - 90)
        
        # Różnica kątów (-π do π)
        angle_diff = angle_to_checkpoint - car_angle_rad
        
        # Normalizuj do zakresu -π do π
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Znormalizuj do zakresu -1 do 1 (gdzie -1 = lewo, 0 = prosto, 1 = prawo)
        self.checkpoint_direction = angle_diff / math.pi
    
    def get_sensor_inputs(self):
        """
        Zwraca wszystkie wejścia dla sieci neuronowej
        
        Returns:
            Lista wartości: czujniki odległości + kierunek do checkpointu
        """
        return self.sensor_readings + [self.checkpoint_direction]
    
    def service_in_pitstop(self, dt):
        """
        Uzupełnia paliwo i regeneruje opony w pitstopie
        
        Args:
            dt: Delta time
        """
        self.pitstop_time += dt
        
        # Szybkie tankowanie i wymiana opon (2 sekundy)
        refuel_rate = self.fuel_capacity / 2.0  # Pełny bak w 2 sekundy
        tire_repair_rate = (self.base_tire_grip - self.min_tire_grip) / 2.0
        
        # Uzupełnij paliwo
        self.fuel = min(self.fuel_capacity, self.fuel + refuel_rate * dt)
        
        # Napraw opony
        self.tire_grip = min(self.base_tire_grip, self.tire_grip + tire_repair_rate * dt)
