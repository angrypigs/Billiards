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
        self.balls.append(Ball(self.screen, (WIDTH // 2, (HEIGHT - 100) // 2), (255, 255, 255), 0))
        for i, (x, y) in enumerate(START_POS):
            self.balls.append(Ball(self.screen, (WIDTH * 3 // 4 + RADIUS * x, (HEIGHT - 100) // 2 + RADIUS * y), choice(COLORS), i + 1))
        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((0, 128, 0))
        self.bg.blit(IMAGES["table_bg"], (0, 0))
        self.bg.blit(IMAGES["table"], (0, 0))
        self.mask = pygame.mask.from_surface(IMAGES["table"])
    
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
                ball.velocity *= 0.99
                if ball.velocity.magnitude() < 0.1:
                    ball.moving = False
                    continue
                for ball2 in self.balls:
                    if ball != ball2:
                        distance = ball.coords.distance_to(ball2.coords)
                        if distance <= 2 * RADIUS:

                            collision_vector = ball2.coords - ball.coords
                            collision_angle = atan2(collision_vector.y, collision_vector.x)

                            v1 = ball.velocity.rotate(-degrees(collision_angle))
                            v2 = ball2.velocity.rotate(-degrees(collision_angle))

                            v1_final = Vector2(v2.x, v1.y)
                            v2_final = Vector2(v1.x, v2.y)

                            ball.velocity = v1_final.rotate(degrees(collision_angle))
                            ball2.velocity = v2_final.rotate(degrees(collision_angle))

                            overlap = 2 * RADIUS - distance
                            correction = collision_vector.normalize() * (overlap / 2)
                            ball.coords -= correction
                            ball2.coords += correction
                            
                            ball2.moving = True
        for ball in self.balls:
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
