from pygame.math import Vector2
import pygame

from src.utils import *


class Ball:

    def __init__(self, 
                 screen: pygame.Surface,
                 coords: tuple[int, int]) -> None:
        self.screen = screen
        self.coords = Vector2(coords)
        self.velocity = Vector2(0, 0)
        self.moving = False

    def punch(self, angle: int, power: int) -> None:
        self.velocity = Vector2(power, 0).rotate(angle)
        self.moving = True

    def draw(self) -> None:
        if self.moving:
            if self.coords[0] <= LIMIT or self.coords[0] >= WIDTH - LIMIT:
                self.velocity[0] *= -1
            elif self.coords[1] <= LIMIT or self.coords[1] >= HEIGHT - LIMIT:
                self.velocity[1] *= -1
            self.coords += self.velocity
            self.velocity *= 0.97
            if self.velocity.magnitude() < 0.1:
                self.moving = False
        pygame.draw.circle(self.screen, (255, 255, 255), self.coords, RADIUS)
