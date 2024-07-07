import pygame
from random import randint

from src.utils import *
from src.ball import Ball

class Game:

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.balls: list[Ball] = []
        self.power = 0
        self.balls.append(Ball(self.screen, (WIDTH // 2, HEIGHT // 2)))
        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((20, 120, 20))
    
    def draw(self) -> None:
        self.screen.blit(self.bg, (0, 0))
        for ball in self.balls:
            ball.draw()

    def release(self) -> None:
        if not self.balls[0].moving:
            self.balls[0].punch(randint(0, 360), self.power)
            self.power = 0
        
    def load(self) -> None:
        if not self.balls[0].moving:
            if self.power < MAX_POWER:
                self.power += 0.1
            else:
                self.power = MAX_POWER