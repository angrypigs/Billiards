import torch
import numpy as np
from src.agent_rl import DualHeadMLPModel
from src.utils import WIDTH, HEIGHT, BALL_QUANTITY, CHECKPOINT_PATH_1PLAYER

class AIPlayer:
    def __init__(self, model_path=CHECKPOINT_PATH_1PLAYER):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualHeadMLPModel().to(self.device)
        self.model.eval()
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f"AI: Loaded brain from {model_path}")
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
                dx = (ball.coords.x - wx) / WIDTH
                dy = (ball.coords.y - wy) / HEIGHT
                features.extend([dx, dy])
                
        return np.array(features, dtype=np.float32)

    def predict(self, game):
        state_np = self.get_state_vector(game)
        state_t = torch.tensor(state_np, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred_class, pred_cont_all = self.model(state_t)

        active_indices = [b.index - 1 for b in game.balls if b.active and b.index != 0]
        if not active_indices:
            return None

        mask = torch.full_like(pred_class, float('-inf'))
        mask[0, active_indices] = 0 
        masked_logits = pred_class + mask

        target_idx_raw = torch.argmax(masked_logits, dim=1).item()
        target_ball_idx = target_idx_raw + 1 

        params = pred_cont_all[0, target_idx_raw]
        raw_angle = params[0].item()
        raw_power = params[1].item()

        final_power = max(0.85, raw_power) 

        return target_ball_idx, raw_angle, final_power