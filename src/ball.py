from pygame.math import Vector2
import pygame

from src.utils import *


class Ball:

    def __init__(self, 
                 screen: pygame.Surface,
                 coords: tuple[int, int],
                 color: tuple[int, int, int],
                 index: int) -> None:
        self.screen = screen
        self.coords = Vector2(coords)
        self.color = color
        self.velocity = Vector2(0, 0)
        self.moving = False
        self.index = index
        self.surf = pygame.Surface((RADIUS * 2, RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.surf, self.color, (RADIUS, RADIUS), RADIUS)
        number = FONTS[24].render(str(index), True, (255, 255, 0))
        number_rect = number.get_rect()
        number_rect.center = (RADIUS, RADIUS)
        self.surf.blit(number, number_rect)
        self.active = True
        self.mask = pygame.mask.from_surface(self.surf)

    def __str__(self) -> str:
        return f"Ball {self.index}"

    def punch(self, angle: int, power: int) -> None:
        self.velocity = Vector2(power, 0).rotate(angle)
        self.moving = True

    def draw(self) -> None:
        self.screen.blit(self.surf, (self.coords.x - RADIUS, self.coords.y - RADIUS))
        # pygame.draw.circle(self.screen, self.color, self.coords, RADIUS)
