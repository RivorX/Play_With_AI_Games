"""
visual_styles.py - 🎨 Snake Visual Rendering System
Zawiera 3 tryby wizualne: Classic, Modern, Realistic
"""
import pygame
import numpy as np
import math


class VisualStyle:
    """Bazowa klasa dla stylów wizualnych"""
    
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
    
    def render(self, canvas, snake, food, direction):
        """Renderuj scenę - do nadpisania w podklasach"""
        raise NotImplementedError


class ClassicStyle(VisualStyle):
    """🟩 Klasyczny styl - proste kwadraty, jasne kolory"""
    
    def __init__(self, grid_size, cell_size):
        super().__init__(grid_size, cell_size)
        self.bg_color = (20, 20, 20)
        self.grid_color = (40, 40, 40)
        self.snake_body_color = (0, 255, 0)
        self.snake_head_color = (150, 255, 150)
        self.food_color = (255, 0, 0)
        self.show_grid = True
    
    def render(self, canvas, snake, food, direction):
        # Tło
        canvas.fill(self.bg_color)
        
        # Siatka
        if self.show_grid:
            for x in range(self.grid_size + 1):
                pygame.draw.line(
                    canvas, 
                    self.grid_color,
                    (x * self.cell_size, 0),
                    (x * self.cell_size, self.window_size),
                    1
                )
            for y in range(self.grid_size + 1):
                pygame.draw.line(
                    canvas,
                    self.grid_color,
                    (0, y * self.cell_size),
                    (self.window_size, y * self.cell_size),
                    1
                )
        
        # Jedzenie
        food_rect = pygame.Rect(
            food[1] * self.cell_size,
            food[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, self.food_color, food_rect)
        
        # Ciało węża
        for segment in snake:
            seg_rect = pygame.Rect(
                segment[1] * self.cell_size,
                segment[0] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(canvas, self.snake_body_color, seg_rect)
        
        # Głowa (jaśniejsza)
        head_rect = pygame.Rect(
            snake[0][1] * self.cell_size,
            snake[0][0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, self.snake_head_color, head_rect)


class ModernStyle(VisualStyle):
    """🎮 Nowoczesny styl - gradientowe kolory, zaokrąglone krawędzie, animacje"""
    
    def __init__(self, grid_size, cell_size):
        super().__init__(grid_size, cell_size)
        self.bg_color = (15, 15, 25)
        self.grid_color = (30, 30, 50)
        self.snake_gradient_start = (50, 200, 100)
        self.snake_gradient_end = (100, 255, 150)
        self.food_color = (255, 80, 80)
        self.food_glow = (255, 120, 120)
        self.animation_frame = 0
        self.corner_radius = max(3, cell_size // 6)
    
    def _draw_gradient_rect(self, canvas, rect, color_start, color_end):
        """Rysuje prostokąt z gradientem pionowym"""
        for y in range(rect.height):
            ratio = y / rect.height
            color = tuple(
                int(color_start[i] + (color_end[i] - color_start[i]) * ratio)
                for i in range(3)
            )
            pygame.draw.line(
                canvas,
                color,
                (rect.x, rect.y + y),
                (rect.x + rect.width, rect.y + y)
            )
    
    def _draw_rounded_rect(self, canvas, rect, color, radius):
        """Rysuje zaokrąglony prostokąt"""
        pygame.draw.rect(canvas, color, rect, border_radius=radius)
    
    def render(self, canvas, snake, food, direction):
        self.animation_frame += 1
        
        # Tło z delikatnym gradientem
        for y in range(self.window_size):
            ratio = y / self.window_size
            bg_r = int(15 + 10 * ratio)
            bg_g = int(15 + 10 * ratio)
            bg_b = int(25 + 15 * ratio)
            pygame.draw.line(canvas, (bg_r, bg_g, bg_b), (0, y), (self.window_size, y))
        
        # Subtelna siatka
        for x in range(self.grid_size + 1):
            alpha_line = pygame.Surface((1, self.window_size))
            alpha_line.set_alpha(30)
            alpha_line.fill(self.grid_color)
            canvas.blit(alpha_line, (x * self.cell_size, 0))
        
        for y in range(self.grid_size + 1):
            alpha_line = pygame.Surface((self.window_size, 1))
            alpha_line.set_alpha(30)
            alpha_line.fill(self.grid_color)
            canvas.blit(alpha_line, (0, y * self.cell_size))
        
        # Jedzenie z pulsującą poświatą
        pulse = abs(math.sin(self.animation_frame * 0.1)) * 0.3 + 0.7
        glow_size = int(self.cell_size * 1.4 * pulse)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        
        glow_center = glow_size // 2
        for radius in range(glow_size // 2, 0, -2):
            alpha = int(40 * (radius / (glow_size / 2)))
            pygame.draw.circle(glow_surface, (*self.food_glow, alpha), (glow_center, glow_center), radius)
        
        glow_x = food[1] * self.cell_size + self.cell_size // 2 - glow_size // 2
        glow_y = food[0] * self.cell_size + self.cell_size // 2 - glow_size // 2
        canvas.blit(glow_surface, (glow_x, glow_y))
        
        # Jedzenie
        food_rect = pygame.Rect(
            food[1] * self.cell_size + 2,
            food[0] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        self._draw_rounded_rect(canvas, food_rect, self.food_color, self.corner_radius)
        
        # Ciało węża z gradientem
        for i, segment in enumerate(snake):
            ratio = i / max(len(snake) - 1, 1)
            color = tuple(
                int(self.snake_gradient_start[j] + 
                    (self.snake_gradient_end[j] - self.snake_gradient_start[j]) * ratio)
                for j in range(3)
            )
            
            seg_rect = pygame.Rect(
                segment[1] * self.cell_size + 1,
                segment[0] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            self._draw_rounded_rect(canvas, seg_rect, color, self.corner_radius)
        
        # Głowa z efektem 3D
        head_rect = pygame.Rect(
            snake[0][1] * self.cell_size + 1,
            snake[0][0] * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        
        # Cień
        shadow_rect = head_rect.copy()
        shadow_rect.y += 2
        self._draw_rounded_rect(canvas, shadow_rect, (30, 80, 40), self.corner_radius)
        
        # Główny kolor
        self._draw_rounded_rect(canvas, head_rect, self.snake_gradient_end, self.corner_radius)
        
        # Highlight
        highlight_rect = pygame.Rect(
            head_rect.x + 3,
            head_rect.y + 3,
            head_rect.width - 6,
            head_rect.height // 3
        )
        highlight_surface = pygame.Surface((highlight_rect.width, highlight_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(highlight_surface, (255, 255, 255, 80), highlight_surface.get_rect(), border_radius=self.corner_radius // 2)
        canvas.blit(highlight_surface, (highlight_rect.x, highlight_rect.y))


class RealisticStyle(VisualStyle):
    """🐍 Realistyczny styl - tekstury, cienie, 3D efekty"""
    
    def __init__(self, grid_size, cell_size):
        super().__init__(grid_size, cell_size)
        self.bg_color = (34, 45, 35)
        self.grass_colors = [
            (40, 52, 42),
            (38, 50, 40),
            (42, 54, 44),
            (36, 48, 38)
        ]
        self.snake_base_color = (60, 120, 60)
        self.snake_scale_color = (80, 140, 80)
        self.snake_belly_color = (180, 200, 150)
        self.food_color = (200, 50, 50)
        self.shadow_color = (0, 0, 0, 80)
        
        # Generuj wzór trawy (raz, żeby było szybko)
        self.grass_pattern = self._generate_grass_pattern()
        
        # Cache dla tekstur segmentów
        self.segment_cache = {}
    
    def _generate_grass_pattern(self):
        """Generuje wzór trawy dla tła"""
        pattern = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pattern[y, x] = (x + y) % len(self.grass_colors)
        return pattern
    
    def _draw_textured_rect(self, canvas, rect, base_color, is_head=False):
        """Rysuje prostokąt z teksturą łusek"""
        # Użyj cache dla lepszej wydajności
        cache_key = (rect.width, rect.height, base_color, is_head)
        
        if cache_key not in self.segment_cache:
            surface = pygame.Surface((rect.width, rect.height))
            surface.fill(base_color)
            
            # Łuski (hexagonalny wzór)
            scale_size = max(3, rect.width // 4)
            for sy in range(0, rect.height, scale_size):
                for sx in range(0, rect.width, scale_size):
                    offset_x = (scale_size // 2) if (sy // scale_size) % 2 else 0
                    
                    # Kolor łuski z wariacją
                    variation = np.random.randint(-10, 10)
                    scale_color = tuple(max(0, min(255, c + variation)) for c in self.snake_scale_color)
                    
                    # Hexagon (przybliżony kółkiem)
                    center_x = sx + offset_x + scale_size // 2
                    center_y = sy + scale_size // 2
                    
                    if 0 <= center_x < rect.width and 0 <= center_y < rect.height:
                        pygame.draw.circle(surface, scale_color, (center_x, center_y), scale_size // 3)
                        
                        # Highlight na łusce
                        highlight_x = center_x - scale_size // 6
                        highlight_y = center_y - scale_size // 6
                        pygame.draw.circle(surface, (min(255, scale_color[0] + 30),
                                                     min(255, scale_color[1] + 30),
                                                     min(255, scale_color[2] + 30)),
                                         (highlight_x, highlight_y), scale_size // 6)
            
            # Dodaj oczy jeśli to głowa
            if is_head:
                eye_radius = max(2, rect.width // 6)
                eye_y = rect.height // 3
                
                # Lewe oko
                pygame.draw.circle(surface, (255, 255, 200), (rect.width // 3, eye_y), eye_radius)
                pygame.draw.circle(surface, (20, 20, 20), (rect.width // 3, eye_y), eye_radius // 2)
                
                # Prawe oko
                pygame.draw.circle(surface, (255, 255, 200), (2 * rect.width // 3, eye_y), eye_radius)
                pygame.draw.circle(surface, (20, 20, 20), (2 * rect.width // 3, eye_y), eye_radius // 2)
            
            self.segment_cache[cache_key] = surface
        
        canvas.blit(self.segment_cache[cache_key], (rect.x, rect.y))
    
    def render(self, canvas, snake, food, direction):
        # Tło z wzorem trawy
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grass_color = self.grass_colors[self.grass_pattern[y, x]]
                grass_rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(canvas, grass_color, grass_rect)
                
                # Dodaj subtelne linie (szczegół trawy)
                if np.random.random() < 0.1:
                    line_color = tuple(max(0, c - 10) for c in grass_color)
                    start_x = x * self.cell_size + np.random.randint(0, self.cell_size)
                    start_y = y * self.cell_size + np.random.randint(0, self.cell_size)
                    end_x = start_x + np.random.randint(-5, 5)
                    end_y = start_y + np.random.randint(2, 8)
                    pygame.draw.line(canvas, line_color, (start_x, start_y), (end_x, end_y), 1)
        
        # Jedzenie (jabłko) z cieniem
        shadow_offset = 3
        shadow_rect = pygame.Rect(
            food[1] * self.cell_size + shadow_offset,
            food[0] * self.cell_size + shadow_offset,
            self.cell_size,
            self.cell_size
        )
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.shadow_color, shadow_surface.get_rect())
        canvas.blit(shadow_surface, (shadow_rect.x, shadow_rect.y))
        
        # Jabłko
        food_rect = pygame.Rect(
            food[1] * self.cell_size + 2,
            food[0] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.ellipse(canvas, self.food_color, food_rect)
        
        # Highlight na jabłku
        highlight_rect = pygame.Rect(
            food_rect.x + food_rect.width // 4,
            food_rect.y + food_rect.height // 6,
            food_rect.width // 3,
            food_rect.height // 4
        )
        pygame.draw.ellipse(canvas, (255, 150, 150), highlight_rect)
        
        # Łodyga
        stem_x = food_rect.centerx
        stem_y = food_rect.top
        pygame.draw.line(canvas, (100, 70, 40), (stem_x, stem_y - 4), (stem_x, stem_y), 2)
        
        # Ciało węża z cieniami i teksturą
        for i, segment in enumerate(snake):
            # Cień
            shadow_rect = pygame.Rect(
                segment[1] * self.cell_size + shadow_offset,
                segment[0] * self.cell_size + shadow_offset,
                self.cell_size - 1,
                self.cell_size - 1
            )
            shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, self.shadow_color, shadow_surface.get_rect())
            canvas.blit(shadow_surface, (shadow_rect.x, shadow_rect.y))
            
            # Segment z teksturą
            seg_rect = pygame.Rect(
                segment[1] * self.cell_size + 1,
                segment[0] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            
            is_head = (i == 0)
            self._draw_textured_rect(canvas, seg_rect, self.snake_base_color, is_head)
            
            # Dodaj 3D efekt (gradient z boku)
            gradient_surface = pygame.Surface((seg_rect.width, seg_rect.height), pygame.SRCALPHA)
            for x in range(seg_rect.width // 3):
                alpha = int(40 * (1 - x / (seg_rect.width / 3)))
                pygame.draw.line(gradient_surface, (0, 0, 0, alpha), (x, 0), (x, seg_rect.height))
            canvas.blit(gradient_surface, (seg_rect.x, seg_rect.y))


def create_renderer(style_name, grid_size, cell_size):
    """
    Factory function do tworzenia rendererów
    
    Args:
        style_name: 'classic', 'modern', lub 'realistic'
        grid_size: rozmiar siatki
        cell_size: rozmiar komórki w pikselach
    
    Returns:
        VisualStyle: Odpowiedni renderer
    """
    styles = {
        'classic': ClassicStyle,
        'modern': ModernStyle,
        'realistic': RealisticStyle
    }
    
    style_class = styles.get(style_name.lower(), ClassicStyle)
    return style_class(grid_size, cell_size)