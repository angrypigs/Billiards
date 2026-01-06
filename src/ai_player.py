import torch
import numpy as np
from pygame.math import Vector2

from src.agent_rl import AngleRegressorModel
from src.utils import *

class AIPlayer:
    def __init__(self, model_path=CHECKPOINT_PATH_1PLAYER):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AngleRegressorModel().to(self.device)
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
                features.extend([0.0, 0.0])
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

            for hole in HOLES:
                hole_vec = Vector2(hole)
                ball_vec = ball.coords

                vec_to_hole = hole_vec - ball_vec
                dist_to_hole = vec_to_hole.length()

                dir_to_hole = vec_to_hole.normalize()
                ghost_pos = ball_vec - (dir_to_hole * (2 * RADIUS))

                dist_white_ghost = (ghost_pos - white_pos).length()
                score = dist_to_hole + dist_white_ghost
                
                if score < best_score:
                    best_score = score
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
        
        print(f"Math: {target_idx} | AI Norm: {raw_angle_norm:.4f}")

        return target_idx, raw_angle_norm, AI_POWER

    select_best_ball_mathematically = select_best_ball_heuristic