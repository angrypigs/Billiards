import pygame
from pygame.math import Vector2
from random import randint, choice
from math import atan2, degrees, sqrt, sin, cos, radians

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
            self.balls.append(Ball(self.screen, 
                (WIDTH * 3 // 4 + RADIUS * x, (HEIGHT - 100) // 2 + RADIUS * y), COLORS[i], i + 1))
        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((0, 128, 0))
        self.bg.blit(IMAGES["table_bg"], (0, 0))
        self.bg.blit(IMAGES["table"], (0, 0))
        for c in HOLES:
            pygame.draw.circle(self.bg, (0, 0, 0), c, POCKET_RADIUS)
        self.mask = pygame.mask.from_surface(IMAGES["table"])
        self.mask_surf = self.mask.to_surface(setcolor=(255,0,0,255), unsetcolor=(0,0,0,0))
        
        self.cue = None
        self.cue_pos = [0, 0]
        
        self.player_flag = None
        self.shoot_counter = 0
        
        
    def draw(self) -> None:
        self.screen.blit(self.bg, (0, 0))
        # self.screen.blit(self.mask_surf, (0,0))
        for ball in self.balls:
            if ball.moving:
                ball.coords += ball.velocity
                self.__ball_collision_single(ball)
                ball.velocity *= 0.99
                if ball.velocity.magnitude() < 0.1:
                    ball.moving = False
                    continue
                for ball2 in self.balls:
                    if ball != ball2:
                        self.__ball_collision_double(ball, ball2)
        for ball in self.balls:
            ball.draw()
        if self.cue is not None:
            self.screen.blit(self.cue, self.cue_pos)
        if not any([b.moving for b in self.balls]) and self.player_flag is None:
            self.player_flag = 0
            
    def __ball_collision_single(self, ball: Ball) -> None:
        offset = (int(ball.coords[0] - RADIUS), int(ball.coords[1] - RADIUS))
        overlap = self.mask.overlap(ball.mask, offset)
        if overlap:
            nx, ny = estimate_normal(self.mask, overlap[0], overlap[1])
            dot = ball.velocity[0] * nx + ball.velocity[1] * ny
            ball.velocity[0] -= 2 * dot * nx
            ball.velocity[1] -= 2 * dot * ny
            ball.coords += ball.velocity
        for (x, y) in HOLES:
            dx = x - ball.coords.x
            dy = y - ball.coords.y
            d = sqrt(dx**2 + dy**2)
            if d < 20:
                print(ball, d)
            if d < POCKET_RADIUS:
                print(ball.index)
                if ball.index == 0:
                    self.shoot_counter = -1
                    ball.velocity.x, ball.velocity.y = 0, 0
                    ball.coords.x, ball.coords.y = WIDTH // 2, (HEIGHT - 100) // 2
                else:
                    self.shoot_counter += 1
                    self.balls.remove(ball)
    
    def __ball_collision_double(self, ball: Ball, ball2: Ball) -> None:
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
            
    def cue_handle(self, pos: tuple[int, int]) -> None:
        if self.player_flag is not None:
            self.cue = pygame.transform.rotate(IMAGES["cue"].copy(), -self.angle)
            self.cue_pos = [
                self.balls[0].coords.x + (CUE_RADIUS + self.power * 4) * cos(radians(self.angle + 180)),
                self.balls[0].coords.y + (CUE_RADIUS + self.power * 4) * sin(radians(self.angle + 180))
            ]
            s = self.cue.get_size()
            self.cue_pos[0] -= s[0] // 2
            self.cue_pos[1] -= s[1] // 2

    def release(self) -> None:
        if self.player_flag is not None:
            self.balls[0].punch(self.angle, self.power)
            self.power = 0
            self.player_flag = None
            self.cue = None
        
    def load(self, pos: tuple[int, int]) -> None:
        if self.player_flag is not None:
            ball_pos = self.balls[0].coords
            self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
            if self.power < MAX_POWER:
                self.power += 0.1
            else:
                self.power = MAX_POWER
        self.cue_handle(pos)
        
    def move(self, pos: tuple[int, int]) -> None:
        ball_pos = self.balls[0].coords
        self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
        self.cue_handle(pos)
