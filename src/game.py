import pygame
from random import uniform
from math import atan2, degrees, sqrt

from src.utils import *
from src.ball import Ball
from src.cue import Cue
from src.physics import PhysicsEngine, ShotEvents
from src.rules import Rules
from src.computer import Computer

class Game:

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        init_assets()

        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((0, 128, 0))
        self.bg.blit(IMAGES["table_bg"], (0, 0))
        self.bg.blit(IMAGES["table"], (0, 0))
        for c in HOLES:
            pygame.draw.circle(self.bg, (0, 0, 0), c, POCKET_RADIUS)
            
        self.mask = pygame.mask.from_surface(IMAGES["table"])
        self.physics = PhysicsEngine(self.mask)
        self.rules = Rules()
        self.computer = Computer()

        self.cue = Cue(self.screen)
        self.reset()

    def reset(self) -> None:
        self.balls: list[Ball] = []
        self.power = 0
        self.angle = 0
        self.rules.reset()
        self.turn = self.rules.turn
        white_start = (WIDTH // 2, (HEIGHT - 100) // 2)
        white_ball = Ball(self.screen, white_start, (255, 255, 255), 0)
        self.balls.append(white_ball)

        for i in range(BALL_QUANTITY):
            x_pos, y_pos = START_POS[i]
            self.balls.append(Ball(
                self.screen,
                (
                    WIDTH * 3 // 4 + RADIUS * x_pos,
                    (HEIGHT - 100) // 2 + RADIUS * y_pos,
                ),
                COLORS[i],
                i + 1,
            ))
        self.movement_flag = False
        self.flag_won = None
        self.cue.update(self.balls[0].coords, 0, 0)
        
    def draw(self) -> None:
        self.__game_frame()
        self.screen.blit(self.bg, (0, 0))
        self.cue.draw()
        for ball in self.balls:
            if ball.active:
                ball.draw()
            
    def shoot(self, angle: float, power: float) -> None:
        rand_power = uniform(power - 0.05, power + 0.05)
        self.balls[0].punch(angle, rand_power)
        self.power = 0
        self.movement_flag = True
        self.physics.shot_events = ShotEvents()
        self.cue.disable()
    
    def __game_frame(self) -> None:
        if self.turn and not self.movement_flag:
            angle, power = self.computer.shoot()
            self.shoot(angle, power)

        self.physics.update(self.balls)

        if not any([b.moving for b in self.balls]) and self.movement_flag:
            self.movement_flag = False
            self.turn = self.rules.handle_turn(self.physics.shot_events)
            if len([b for b in self.balls if b.active]) < 2 and self.flag_won is None:
                self.flag_won = 0
            
    def cue_handle(self, pos: tuple[int, int]) -> None:
        if not self.movement_flag and not self.turn:
            self.cue.update(self.balls[0].coords, self.angle, self.power)

    def release(self) -> None:
        if not self.movement_flag and not self.turn:
            self.shoot(self.angle, self.power)
        
    def load(self, pos: tuple[int, int]) -> None:
        if not self.movement_flag and not self.turn:
            ball_pos = self.balls[0].coords
            self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
            if self.power < MAX_POWER:
                self.power += 0.15
            else:
                self.power = MAX_POWER
        self.cue_handle(pos)
        
    def move(self, pos: tuple[int, int]) -> None:
        if not self.turn:
            ball_pos = self.balls[0].coords
            self.angle = degrees(atan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0]))
            self.cue_handle(pos)
