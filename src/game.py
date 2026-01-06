import pygame
from pygame.math import Vector2
from random import uniform
from math import atan2, degrees, sqrt, asin

from src.utils import *
from src.ball import Ball
from src.cue import Cue
from src.db import dbHandler

class Game:

    def __init__(self, db: dbHandler, screen: pygame.Surface | None = None, debug: bool = True, special_mode: int = 0, off_screen: bool = False) -> None:
        if screen is not None:
            self.screen = screen
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        init_assets()
        if off_screen:
            pygame.display.quit()
        self.db = db
        self.debug = debug

        self.bg = pygame.Surface((WIDTH, HEIGHT))
        self.bg.fill((0, 128, 0))
        self.bg.blit(IMAGES["table_bg"], (0, 0))
        self.bg.blit(IMAGES["table"], (0, 0))
        for c in HOLES:
            pygame.draw.circle(self.bg, (0, 0, 0), c, POCKET_RADIUS)
            
        self.mask = pygame.mask.from_surface(IMAGES["table"])
        self.mask_surf = self.mask.to_surface(setcolor=(255,0,0,120), unsetcolor=(0,0,0,0))

        dummy_surface = self.screen if self.screen else pygame.Surface((1, 1))
        self.cue = Cue(dummy_surface)
        self.reset(special_mode)

    def reset(self, special_mode: int = 0) -> None: 
        self.balls: list[Ball] = []
        self.power = 0
        self.angle = 0
        white_start = (WIDTH // 2, (HEIGHT - 100) // 2)
        white_ball = Ball(self.screen, white_start, (255, 255, 255), 0)
        self.balls.append(white_ball)
        if special_mode:
            for i in range(special_mode):
                valid_pos = False
                attempts = 0
                while not valid_pos and attempts < 100:
                    x = WIDTH * 4 // 5 - (RADIUS * 3 * i) 
                    y = uniform(150, HEIGHT - 250)
                    dist = sqrt((x - white_ball.coords.x)**2 + (y - white_ball.coords.y)**2)
                    if dist > 2.5 * RADIUS:
                        valid_pos = True
                        self.balls.append(Ball(self.screen, (x, y), COLORS[i], i + 1))
                    attempts += 1
                
                if not valid_pos:
                    print("WARN: Cannot place ball randomly, reseted.")
                    self.balls.append(Ball(self.screen, (WIDTH - 100, HEIGHT - 100), COLORS[i], i + 1))
                    
        else:
            for i in range(BALL_QUANTITY):
                x_pos, y_pos = START_POS[i]
                self.balls.append(Ball(self.screen, 
                    (WIDTH * 3 // 4 + RADIUS * x_pos, (HEIGHT - 100) // 2 + RADIUS * y_pos), COLORS[i], i + 1))
        self.player_flag = 0
        self._history = []
        self.shoot_counter = 0
        self._white_shooted = False
        self._any_hits = False
        self._save = {}
        self.flag_won = None
        self._physics_error = False
        self.debug_score = None
        self.cue.update(self.balls[0].coords, 0, 0)
        
    def draw(self) -> None:
        self.__game_frame(history_save=False)
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
            
    def simulate(self, ball_idx: int, angle: float, power: float = 0.75, backtrack: bool = False) -> float:
        data = {"angle": angle, "ball": ball_idx}
        for i in range(BALL_QUANTITY + 1):
            found = next((b for b in self.balls if b.index == i and b.active), None)
            data[f"x_{i}"] = -1 if found is None else round(found.coords.x, 4)
            data[f"y_{i}"] = -1 if found is None else round(found.coords.y, 4)
        self._save = data.copy()
        self.shoot_counter = 0
        self._white_shooted = False
        saved_state = [(ball.coords.xy[:], ball.active) for ball in self.balls]
        saved_won = self.flag_won
        
        new_angle, new_power = self.agent_data_to_input(ball_idx, angle, power)
        self.shoot(new_angle, new_power)
        while self.player_flag is None:
            self.__game_frame(history_save=True, backtrack=backtrack)
        if backtrack:
            for ball, state in zip(self.balls, saved_state):
                coords, active = state
                ball.coords = Vector2(coords)
                ball.velocity = Vector2(0, 0)
                ball.active = active
                self.flag_won = saved_won
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
        rand_power = uniform(power - 0.05, power + 0.05)
        self.balls[0].punch(angle, rand_power)
        self.power = 0
        self.player_flag = None
        self._physics_error = False
        self.cue.disable()
        self.shoot_counter = 0
        self._white_shooted = False
        self._any_hits = False
        self.debug_score = None
    
    def __game_frame(self, history_save: bool = False, backtrack: bool = False) -> None:
        for ball in self.balls:
            if not ball.active:
                continue
            if ball.moving:
                ball.last_valid_coords = ball.coords.copy()
                ball.coords += ball.velocity
                self.__ball_collision_single(ball, backtrack=backtrack)
                ball.velocity *= 0.99
                if ball.velocity.magnitude() < 0.1:
                    ball.moving = False
                    continue
                for ball2 in self.balls:
                    if ball != ball2 and ball2.active:
                        self.__ball_collision_double(ball, ball2)
                if (ball.coords.x < 10 or ball.coords.x > WIDTH - 10 or
                    ball.coords.y < 10 or ball.coords.y > HEIGHT - 10):
                    print(f"Physics error: ball {ball} on {ball.coords}")
                    ball.coords = ball.last_valid_coords.copy()
                    if ball.velocity.magnitude() > MAX_POWER:
                        ball.velocity.scale_to_length(MAX_POWER * 0.8)
                    if (ball.coords.x < ERROR_THRESHOLD or ball.coords.x > WIDTH - ERROR_THRESHOLD or
                        ball.coords.y < ERROR_THRESHOLD or ball.coords.y > HEIGHT - ERROR_THRESHOLD):
                        print(f"Critical physics error: ball {ball} reseted to table middle")
                        ball.coords = Vector2(WIDTH // 2, (HEIGHT - 100) // 2)
                        ball.velocity = Vector2(0, 0)
                        self._physics_error = True
        if not any([b.moving for b in self.balls]) and self.player_flag is None:
            self.player_flag = 0
            self.__calculate_score()
            if history_save and not backtrack: self._history.append(self._save.copy())
            if len([b for b in self.balls if b.active]) < 2 and self.flag_won is None:
                self.flag_won = 0

    def __calculate_score(self) -> None:
        if self._physics_error:
            self._save["score"] = 0.0
            return
        if self._white_shooted:
            self._save["score"] = -5.0
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
        no_hit_penalty = -1.0 if not self._any_hits else 0.0
        self._save["score"] = self.shoot_counter * 1.5 + max_simil + no_hit_penalty - 0.45
        self.debug_score = None if debug_data == [-1, -1] else debug_data
        if self.debug and self.debug_score is not None:
            print(self.balls[self.debug_score[0]])
            print(self._save["score"])
                    
            
    def __ball_collision_single(self, ball: Ball, backtrack: bool = False) -> None:
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
                    # if not backtrack: print(ball.index)
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
            if ball.index == 0 or ball2.index == 0:
                self._any_hits = True
            
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

    def ball_angle_range(self, target_ball: Ball) -> tuple[float, float] | None:
        cue_pos = self.balls[0].coords
        COLLISION_DISTANCE = 2 * RADIUS 
        if target_ball.index == 0 or not target_ball.active:
            return None
        ball_pos = target_ball.coords
        D = cue_pos.distance_to(ball_pos)
        if D < 0.00001:
            print(f"!!! CRITICAL WARNING: Distance D is practically ZERO ({D}). Balls are merged at same coords!")
            return (0.0, 360.0)
        dx = ball_pos.x - cue_pos.x
        dy = ball_pos.y - cue_pos.y
        theta_center = degrees(atan2(dy, dx))
        if D <= COLLISION_DISTANCE + 0.0001: 
            diff = COLLISION_DISTANCE - D
            print(f"--- PHYSICS WARN: Overlap detected! ---")
            print(f"    Target Ball: {target_ball.index}")
            print(f"    Distance D: {D:.8f}")
            print(f"    Required:   {COLLISION_DISTANCE:.8f}")
            print(f"    Overlap:    {diff:.8f}")
            print(f"    -> Returning fallback push-shot angle.")
            return (theta_center - 89.0, theta_center + 89.0)
        try:
            raw_ratio = COLLISION_DISTANCE / D
            if raw_ratio > 1.0 or raw_ratio < -1.0:
                print(f"--- MATH WARN: Floating point precision error (Clamping needed) ---")
                print(f"    Distance D: {D:.10f}")
                print(f"    Raw Ratio:  {raw_ratio:.10f}")
                print(f"    -> Clamping to [-1, 1]")
            ratio = max(-1.0, min(1.0, raw_ratio))
            alpha = degrees(asin(ratio))
        except ValueError as e:
            print(f"!!! CRITICAL MATH ERROR in asin() !!!")
            print(f"    Error: {e}")
            print(f"    Distance D: {D}")
            print(f"    Ratio: {COLLISION_DISTANCE/D}")
            return (theta_center - 90, theta_center + 90)
        except Exception as e:
            print(f"!!! UNKNOWN ERROR in ball_angle_range !!! {e}")
            return None
            
        theta_min = theta_center - alpha
        theta_max = theta_center + alpha
        
        return (theta_min, theta_max)

    def agent_data_to_input(self, ball_idx: int, angle: float, power: float) -> tuple[float, float]:
        ball = next((b for b in self.balls if b.index == ball_idx and b.active), None)
        bounds = None if ball is None else self.ball_angle_range(ball)
        if bounds is None:
            new_angle = uniform(-180, 180)
            debug_info = "" if ball is None else f"at {ball.coords}"
            print(f"Critical error: ball {ball_idx} not found or numerical error {debug_info}")
        else:
            angle_coefficient = (angle + 1.0) / 2.0
            new_angle = bounds[0] + angle_coefficient * (bounds[1] - bounds[0])
        new_power = power * MAX_POWER
        return new_angle, new_power