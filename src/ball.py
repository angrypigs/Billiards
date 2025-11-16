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
        surf = pygame.Surface((RADIUS * 2, RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 255, 255, 255), (RADIUS, RADIUS), RADIUS)
        self.mask = pygame.mask.from_surface(surf)

    def __str__(self) -> str:
        return f"Ball {self.index}"

    def punch(self, angle: int, power: int) -> None:
        self.velocity = Vector2(power, 0).rotate(angle)
        self.moving = True

    def draw(self) -> None:
        pygame.draw.circle(self.screen, self.color, self.coords, RADIUS)
