import pygame
from pygame.math import Vector2
from math import degrees, atan2, acos
import random
import sys

from src.game import Game
from src.utils import *
from src.db import dbHandler

class SmartTeacher:
    def __init__(self, db_path=TRAINING_DATA_PATH_1PLAYER):
        self.db = dbHandler(db_path)
        self.game = Game(self.db, debug=False, off_screen=True)
        self.width = WIDTH
        self.height = HEIGHT

    def is_path_clear(self, start_pos: Vector2, end_pos: Vector2, ignore_indices: list[int]) -> bool:
        if line_hits_mask(self.game.mask, start_pos.x, start_pos.y, end_pos.x, end_pos.y):
            return False

        safe_margin = RADIUS * 2.2
        
        for ball in self.game.balls:
            if not ball.active or ball.index in ignore_indices:
                continue
            
            if is_point_in_rectangle_buffer(start_pos, end_pos, ball.coords, safe_margin):
                return False
                
        return True

    def calculate_best_shot(self):
        white = self.game.balls[0]
        if not white.active: return None

        best_shot = None
        best_difficulty_score = float('inf')

        targets = [b for b in self.game.balls if b.index != 0 and b.active]
        
        for target in targets:
            for hole in HOLES:
                hole_vec = Vector2(hole)
                
                target_to_hole = hole_vec - target.coords
                dist_target_hole = target_to_hole.length()
                
                aim_dir = target_to_hole.normalize()
                
                ghost_pos = target.coords - (aim_dir * (2 * RADIUS))
                
                white_to_ghost = ghost_pos - white.coords
                dist_white_ghost = white_to_ghost.length()
                shot_dir = white_to_ghost.normalize()

                dot_product = shot_dir.dot(aim_dir)
                cut_angle_rad = acos(max(-1.0, min(1.0, dot_product)))
                cut_angle_deg = degrees(cut_angle_rad)

                if cut_angle_deg > 80:
                    continue

                if not self.is_path_clear(white.coords, ghost_pos, [0, target.index]):
                    continue

                if not self.is_path_clear(target.coords, hole_vec, [0, target.index]):
                    continue

                difficulty = dist_white_ghost + dist_target_hole + (cut_angle_deg * 8)

                if difficulty < best_difficulty_score:
                    required_angle_global = degrees(atan2(shot_dir.y, shot_dir.x))
                    
                    bounds = self.game.ball_angle_range(target)
                    
                    if bounds is None:
                        continue
                        
                    agent_angle_input = map_angle_to_agent_output(required_angle_global, bounds)
                    
                    best_difficulty_score = difficulty
                    best_shot = (target.index, agent_angle_input, 0.75)

        return best_shot

    def run(self, samples_needed=10000):
        samples_collected = 0
        iterations = 0
        
        print(f"Generating {samples_needed} balanced samples...")
        
        while samples_collected < samples_needed:
            iterations += 1

            n_balls = random.randint(1, 15)
            use_standard_triangle = (random.random() > 0.9)
            
            mode = 0 if use_standard_triangle else n_balls
            self.game.reset(special_mode=mode)

            if mode != 0:
                colored_balls = [b for b in self.game.balls if b.index != 0]

                new_indices = random.sample(range(1, 16), len(colored_balls))

                for ball, new_idx in zip(colored_balls, new_indices):
                    ball.index = new_idx
            
            shot_params = self.calculate_best_shot()
            
            if shot_params:
                ball_idx, angle_input, power_input = shot_params

                score = self.game.simulate(ball_idx, angle_input, power_input, backtrack=False)
                
                if score > 0:
                    self.game.save_history()
                    samples_collected += 1
                    
                    if samples_collected % 100 == 0:
                        print(f"Progress: {samples_collected}/{samples_needed}")
                else:
                    self.game._history = []
            else:
                self.game._history = []

        self.game.db.close()