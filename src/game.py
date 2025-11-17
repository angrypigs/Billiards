import pygame
from pygame.math import Vector2
from random import randint, choice, uniform
from math import atan2, degrees, sqrt, sin, cos, radians
import csv

from src.utils import *
from src.ball import Ball
from src.cue import Cue

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
        
        self.cue = Cue(self.screen)
        self.player_flag = 0
        
        self._history = []
        self.shoot_counter = 0
        self._white_shooted = False
        self._save = {}
        self.flag_won = None
        
    def draw(self) -> None:
        self.__game_frame()
        self.screen.blit(self.bg, (0, 0))
        # self.screen.blit(self.mask_surf, (0,0))
        self.cue.draw()
        for ball in self.balls:
            ball.draw()
            
    def simulate(self, angle: float, power: float = MAX_POWER) -> None:
        self.__shoot(angle, power)
        while self.player_flag is None:
            self.__game_frame()
        return self.balls_pos, self.shoot_counter
    
    def save_history(self) -> None:
        if self._history:
            headers = self._history[0].keys()
            for i in self._history:
                print(i)
            with open(CSV_TRAINING_DATA_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerows(self._history)
            self._history = []
    
    def __shoot(self, angle: float, power: float) -> None:
        rand_power = uniform(power - 0.1, power + 0.1)
        self.balls[0].punch(angle, rand_power)
        self.power = 0
        self.player_flag = None
        self.cue.disable()
        self.shoot_counter = 0
        balls_pos = {"angle": angle, "power": rand_power}
        for i in range(16):
            found = next((b for b in self.balls if b.index == i), None)
            balls_pos[f"{i}_x"] = -1 if found is None else round(found.coords.x, 4)
            balls_pos[f"{i}_y"] = -1 if found is None else round(found.coords.y, 4)
        self._save = balls_pos.copy()
    
    def __game_frame(self) -> None:
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
        if not any([b.moving for b in self.balls]) and self.player_flag is None:
            self.player_flag = 0
            self.__calculate_score()
            self._history.append(self._save.copy())
            if len(self.balls) < 2 and self.flag_won is None:
                self.flag_won = 0

    def __calculate_score(self) -> None:
        if self._white_shooted:
            self._save["score"] = -1
            return
        max_simil = 0
        c = self.balls[0].coords
        for ball in self.balls:
            if ball.index != 0:
                a = (ball.coords.x - c.x, ball.coords.y - c.y)
                for (hx, hy) in HOLES:
                    b = (hx - ball.coords.x, hy - ball.coords.y)
                    simil = cosine_similarity(a, b)
                    if simil > max_simil:
                        max_simil = simil
        max_simil = (max_simil + 1) / 4
        self._save["score"] = self.shoot_counter + max_simil
                    
            
    def __ball_collision_single(self, ball: Ball) -> None:
        offset = (int(ball.coords[0] - RADIUS), int(ball.coords[1] - RADIUS))
        overlap = self.mask.overlap(ball.mask, offset)
        if overlap:
            nx, ny = estimate_normal(self.mask, overlap[0], overlap[1])
            vx, vy = ball.velocity
            dot = vx * nx + vy * ny
            ball.velocity[0] = vx - 2 * dot * nx
            ball.velocity[1] = vy - 2 * dot * ny
            px, py = ball.coords
            while self.mask.overlap(ball.mask, (int(px - RADIUS), int(py - RADIUS))):
                # print(f"Overlap: {ball} with normal {nx}, {ny}")
                px -= nx * 0.5
                py -= ny * 0.5
            ball.coords.x = px
            ball.coords.y = py
        for (x, y) in HOLES:
            dx = x - ball.coords.x
            dy = y - ball.coords.y
            d = sqrt(dx**2 + dy**2)
            if d < 20:
                print(ball, d)
            if d < POCKET_RADIUS:
                print(ball.index)
                if ball.index == 0:
                    self._white_shooted = True
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
            self.cue.update(self.balls[0].coords, self.angle, self.power)

    def release(self) -> None:
        if self.player_flag is not None:
            self.__shoot(self.angle, self.power)
        
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
