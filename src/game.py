import pygame
from pygame.math import Vector2
from random import randint, choice
from math import atan2, degrees, sqrt

from src.utils import *
from src.ball import Ball

class Game:

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.balls: list[Ball] = []
        self.power = 0
        self.angle = 0
        self.balls.append(Ball(self.screen, (WIDTH // 2, HEIGHT // 2), (255, 255, 255), 0))
        for i, (x, y) in enumerate(START_POS):
            self.balls.append(Ball(self.screen, (WIDTH * 3 // 4 + RADIUS * x, HEIGHT // 2 + RADIUS * y), choice(COLORS), i + 1))
        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((20, 120, 20))
    
    def draw(self) -> None:
        self.screen.blit(self.bg, (0, 0))
        for ball in self.balls:
            if ball.moving:
                ball.coords += ball.velocity
                if ball.coords[0] <= LIMIT or ball.coords[0] >= WIDTH - LIMIT:
                    ball.velocity[0] *= -1
                    ball.coords += ball.velocity
                elif ball.coords[1] <= LIMIT or ball.coords[1] >= HEIGHT - LIMIT:
                    ball.velocity[1] *= -1
                    ball.coords += ball.velocity
                ball.velocity *= 0.98
                if ball.velocity.magnitude() < 0.1:
                    ball.moving = False
                    break
                for ball2 in self.balls:
                    if (ball != ball2 and 
                        sqrt((ball2.coords[0] - ball.coords[0])**2 + (ball2.coords[1] - ball.coords[1])**2) <= 2 * RADIUS and
                        ball.velocity.magnitude() >= ball2.velocity.magnitude()):
                        angle = degrees(atan2(ball.coords.y - ball2.coords.y, ball.coords.x - ball2.coords.x))
                        power = ball.velocity.magnitude()
                        ball2.moving = True
                        ball2.velocity = Vector2(power, 0).rotate(angle)
                        ball2.velocity.rotate_ip(-90)
                        ball2.coords += ball2.velocity
                        ball.coords += ball.velocity
            ball.draw()

    def release(self) -> None:
        if not self.balls[0].moving:
            self.balls[0].punch(self.angle, self.power)
            self.power = 0
        
    def load(self, pos: tuple[int, int]) -> None:
        if not self.balls[0].moving:
            ball_pos = self.balls[0].coords
            self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
            if self.power < MAX_POWER:
                self.power += 0.1
            else:
                self.power = MAX_POWER
