import pygame
from pygame.math import Vector2
from random import randint, choice, uniform
from math import atan2, degrees, sqrt, sin, cos, radians
import csv

from src.utils import *
from src.ball import Ball
from src.cue import Cue
from src.db import dbHandler

class Game:

    def __init__(self, db: dbHandler, screen: pygame.Surface | None = None, debug: bool = True) -> None:
        if screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            init_assets()
        else:
            self.screen = screen
        self.db = db
        self.balls: list[Ball] = []
        self.power = 0
        self.angle = 0
        self.balls.append(Ball(self.screen, (WIDTH // 2, (HEIGHT - 100) // 2), (255, 255, 255), 0))
        for i in range(BALL_QUANTITY):
            x, y = START_POS[i]
            self.balls.append(Ball(self.screen, 
                (WIDTH * 3 // 4 + RADIUS * x, (HEIGHT - 100) // 2 + RADIUS * y), COLORS[i], i + 1))
        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((0, 128, 0))
        self.bg.blit(IMAGES["table_bg"], (0, 0))
        self.bg.blit(IMAGES["table"], (0, 0))
        for c in HOLES:
            pygame.draw.circle(self.bg, (0, 0, 0), c, POCKET_RADIUS)
        self.mask = pygame.mask.from_surface(IMAGES["table"])
        self.mask_surf = self.mask.to_surface(setcolor=(255,0,0,120), unsetcolor=(0,0,0,0))
        
        self.cue = Cue(self.screen)
        self.player_flag = 0
        
        self._history = []
        self.shoot_counter = 0
        self._white_shooted = False
        self._save = {}
        self.flag_won = None
        
        self.debug = debug
        self.debug_score = None
        
    def draw(self) -> None:
        self.__game_frame()
        self.screen.blit(self.bg, (0, 0))
        # self.screen.blit(self.mask_surf, (0,0))
        self.cue.draw()
        for ball in self.balls:
            if ball.active:
                ball.draw()
        if self.debug and self.debug_score is not None:
            try:
                pygame.draw.line(self.screen, (255, 0, 0), self.balls[0].coords, 
                                 self.balls[self.debug_score[0]].coords)
                pygame.draw.line(self.screen, (255, 0, 0), self.balls[self.debug_score[0]].coords, 
                                 HOLES[self.debug_score[1]])
            except Exception:
                pass
            
    def simulate(self, angle: float, power: float = MAX_POWER, backtrack: bool = False) -> float:
        self.shoot_counter = 0
        self._white_shooted = False
        saved_state = [(ball.coords.xy[:], ball.active) for ball in self.balls]
        self.shoot(angle, power)
        while self.player_flag is None:
            self.__game_frame(backtrack=backtrack)
        if backtrack:
            for ball, state in zip(self.balls, saved_state):
                coords, active = state
                ball.coords = Vector2(coords)
                ball.active = active
        return self._save["score"]
    
    def save_history(self) -> None:
        if self._history:
            if self.debug:
                for i in self._history:
                    print(i)
            for h in self._history:
                self.db.insert([h[x] for x in COLUMN_NAMES])
            self._history = []
    
    def shoot(self, angle: float, power: float) -> None:
        rand_power = uniform(power - 0.1, power + 0.1)
        self.balls[0].punch(angle, rand_power)
        self.power = 0
        self.player_flag = None
        self.cue.disable()
        self.shoot_counter = 0
        self._white_shooted = False
        balls_pos = {"angle": angle, "power": rand_power}
        for i in range(BALL_QUANTITY + 1):
            found = next((b for b in self.balls if b.index == i and b.active), None)
            balls_pos[f"x_{i}"] = -1 if found is None else round(found.coords.x, 4)
            balls_pos[f"y_{i}"] = -1 if found is None else round(found.coords.y, 4)
        self._save = balls_pos.copy()
        self.debug_score = None
    
    def __game_frame(self, backtrack: bool = False) -> None:
        for ball in self.balls:
            if not ball.active:
                continue
            if ball.moving:
                ball.coords += ball.velocity
                self.__ball_collision_single(ball)
                ball.velocity *= 0.99
                if ball.velocity.magnitude() < 0.1:
                    ball.moving = False
                    continue
                for ball2 in self.balls:
                    if ball != ball2 and ball2.active:
                        self.__ball_collision_double(ball, ball2)
        if not any([b.moving for b in self.balls]) and self.player_flag is None:
            self.player_flag = 0
            self.__calculate_score()
            if not backtrack: self._history.append(self._save.copy())
            if len([b for b in self.balls if b.active]) < 2 and self.flag_won is None:
                self.flag_won = 0

    def __calculate_score(self) -> None:
        if self._white_shooted:
            self._save["score"] = -1
            return
        max_simil = 0
        debug_data = [-1, -1]
        c = self.balls[0].coords
        for i, ball in enumerate(self.balls):
            flag = False
            if ball.index != 0 and ball.active:
                for ball2 in self.balls:
                    if ball2.index != 0 and ball2.index != ball.index and ball2.active:
                        if (is_point_in_rectangle_buffer(c, ball.coords, ball2.coords, RADIUS * 2) or 
                            line_hits_mask(self.mask, c.x, c.y, ball.coords.x, ball.coords.y)):
                            flag = True
                            break
                if flag: continue
                a = (ball.coords.x - c.x, ball.coords.y - c.y)
                for j, (hx, hy) in enumerate(HOLES):
                    flag2 = False
                    for ball2 in self.balls:
                        if ball2.index != 0 and ball2.index != ball.index and ball2.active:
                            if (is_point_in_rectangle_buffer(ball.coords, (hx, hy), ball2.coords, RADIUS * 2) or
                                line_hits_mask(self.mask, ball.coords.x, ball.coords.y, hx, hy)):
                                flag2 = True
                                break
                    if flag2: continue
                    b = (hx - ball.coords.x, hy - ball.coords.y)
                    dist = sqrt(a[0]**2 + a[1]**2) + sqrt(b[0]**2 + b[1]**2)
                    simil = cosine_similarity(a, b) - dist / DIAMETER / 2
                    if simil > max_simil:
                        max_simil = simil
                        debug_data = [i, j]
        max_simil = max(0.0, max_simil) / 2
        self._save["score"] = self.shoot_counter + max_simil
        self.debug_score = None if debug_data == [-1, -1] else debug_data
        if self.debug and self.debug_score is not None:
            print(self.balls[self.debug_score[0]], max_simil)
                    
            
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
            if d < POCKET_RADIUS:
                if ball.index == 0:
                    self._white_shooted = True
                    ball.velocity.x, ball.velocity.y = 0, 0
                    ball.coords.x, ball.coords.y = WIDTH // 2, (HEIGHT - 100) // 2
                else:
                    print(ball.index)
                    ball.active = False
                    ball.velocity.x, ball.velocity.y = 0, 0
                    ball.coords.x, ball.coords.y = -1000, -1000
                    self.shoot_counter += 1
    
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
            self.shoot(self.angle, self.power)
        
    def load(self, pos: tuple[int, int]) -> None:
        if self.player_flag is not None:
            ball_pos = self.balls[0].coords
            self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
            if self.power < MAX_POWER:
                self.power += 0.15
            else:
                self.power = MAX_POWER
        self.cue_handle(pos)
        
    def move(self, pos: tuple[int, int]) -> None:
        ball_pos = self.balls[0].coords
        self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
        self.cue_handle(pos)
