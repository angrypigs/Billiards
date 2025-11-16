import pygame
from math import sin, cos, radians

from src.utils import IMAGES, CUE_RADIUS, WIDTH, HEIGHT


class Cue:
    
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.surface_cue = None
        self.pos_cue = [0, 0]
        self.flag_trail = False
        self.pos_trail_start = pygame.math.Vector2(0, 0)
        self.pos_trail_end = pygame.math.Vector2(0, 0)

    def update(self, cue_ball_pos, angle, power) -> None:
        self.surface_cue = pygame.transform.rotate(IMAGES["cue"].copy(), -angle)
        self.pos_cue = [
            cue_ball_pos.x + (CUE_RADIUS + power * 4) * cos(radians(angle + 180)),
            cue_ball_pos.y + (CUE_RADIUS + power * 4) * sin(radians(angle + 180)),
        ]
        w, h = self.surface_cue.get_size()
        self.pos_cue[0] -= w // 2
        self.pos_cue[1] -= h // 2
        self.flag_trail = True
        self.pos_trail_start.x = cue_ball_pos.x
        self.pos_trail_start.y = cue_ball_pos.y
        self.pos_trail_end.x = cue_ball_pos.x + WIDTH * cos(radians(angle))
        self.pos_trail_end.y = cue_ball_pos.y + WIDTH * sin(radians(angle))
        
    def disable(self) -> None:
        self.surface_cue = None
        self.flag_trail = False

    def draw(self) -> None:
        if self.surface_cue:
            self.screen.blit(self.surface_cue, self.pos_cue)
        if self.flag_trail:
            pygame.draw.line(self.screen, (150, 20, 20), 
                             self.pos_trail_start, self.pos_trail_end, 2)