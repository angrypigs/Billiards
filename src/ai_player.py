import torch
import numpy as np
from pygame.math import Vector2

from src.model import AngleRegressorModel
from src.utils import *
from src.game import Game

class AIPlayer:
    def __init__(self, model_path=CHECKPOINT_PATH_1PLAYER):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AngleRegressorModel().to(self.device)
        self.game: Game | None = None
        self.model.eval()
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f"AI: Loaded model from {model_path}")
        except Exception as e:
            print(f"AI CRITICAL ERROR: Could not load model! {e}")

    def get_state_vector(self, game):
        white = game.balls[0]
        wx = white.coords.x
        wy = white.coords.y
        features = [wx / WIDTH, wy / HEIGHT]
        
        for i in range(1, BALL_QUANTITY + 1):
            ball = next((b for b in game.balls if b.index == i and b.active), None)
            if ball is None:
                features.extend([-1.0, 0.0])
            else:
                dx = ball.coords.x - wx
                dy = ball.coords.y - wy
                dist = np.sqrt(dx**2 + dy**2) / DIAGONAL
                angle = np.arctan2(dy, dx) / np.pi 
                features.extend([dist, angle])
        return np.array(features, dtype=np.float32)

    def select_best_ball_heuristic(self, game):
        best_ball_idx = None
        best_score = float('inf')
        
        white_pos = game.balls[0].coords

        for ball in game.balls:
            if not ball.active or ball.index == 0:
                continue

            ball_vec = ball.coords

            for hole in HOLES:
                hole_vec = Vector2(hole)

                vec_to_hole = hole_vec - ball_vec
                dist_to_hole = vec_to_hole.length()

                if dist_to_hole == 0: continue

                dir_to_hole = vec_to_hole.normalize()
                ghost_pos = ball_vec - (dir_to_hole * (2 * RADIUS))
                vec_white_ghost = ghost_pos - white_pos
                dist_white_ghost = vec_white_ghost.length()
                
                if dist_white_ghost == 0: dir_attack = Vector2(0,0)
                else: dir_attack = vec_white_ghost.normalize()

                cos_sim = dir_attack.dot(dir_to_hole)

                penalty_angle = DIAGONAL * (1.0 - cos_sim)
                coll_white_pen = 0

                if line_hits_mask(game.mask, white_pos.x, white_pos.y, ghost_pos.x, ghost_pos.y):
                     coll_white_pen = DIAGONAL * 10

                else:
                    for ball_coll in game.balls:
                        if not ball_coll.active or ball_coll.index == 0 or ball_coll.index == ball.index:
                            continue
                        if is_point_in_rectangle_buffer(white_pos, ghost_pos, ball_coll.coords, RADIUS * 2.1):
                            coll_white_pen = DIAGONAL * 10
                            break

                coll_ball_pen = 0
                
                if line_hits_mask(game.mask, ball_vec.x, ball_vec.y, hole_vec.x, hole_vec.y):
                    coll_ball_pen = DIAGONAL * 10
                else:
                    for ball_coll in game.balls:
                        if not ball_coll.active or ball_coll.index == 0 or ball_coll.index == ball.index:
                            continue
                        if is_point_in_rectangle_buffer(ball_vec, hole_vec, ball_coll.coords, RADIUS * 2.1):
                            coll_ball_pen = DIAGONAL * 10
                            break
                
                current_score = (dist_to_hole + dist_white_ghost) + \
                                penalty_angle + \
                                coll_white_pen + \
                                coll_ball_pen
                
                if current_score < best_score:
                    best_score = current_score
                    best_ball_idx = ball.index

        return best_ball_idx

    def predict(self, game):
        target_idx = self.select_best_ball_mathematically(game)
        
        if target_idx is None:
            return None

        state_np = self.get_state_vector(game)
        state_t = torch.tensor(state_np, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred_angles_norm = self.model(state_t)

        tensor_idx = target_idx - 1
        raw_angle_norm = pred_angles_norm[0, tensor_idx].item()
        raw_angle_norm = max(min(raw_angle_norm, 0.95), -0.95)

        return target_idx, raw_angle_norm, AI_POWER

    select_best_ball_mathematically = select_best_ball_heuristic