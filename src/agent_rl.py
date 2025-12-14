import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pygame
from collections import deque
from src.game import Game
from src.db import dbHandler
from src.utils import WIDTH, HEIGHT, BALL_QUANTITY, CHECKPOINT_PATH_1PLAYER, TRAINING_DATA_PATH_1PLAYER, MAX_POWER

class DualHeadMLPModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[256, 256, 128]):
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )

        self.actor_discrete = nn.Linear(hidden_dims[2], BALL_QUANTITY)
        self.actor_continuous = nn.Linear(hidden_dims[2], BALL_QUANTITY * 2) 

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

        nn.init.orthogonal_(self.actor_continuous.weight, gain=0.01)

    def forward(self, x):
        features = self.trunk(x)
        discrete_output = self.actor_discrete(features)
        continuous_flat = self.actor_continuous(features)
        continuous_output = continuous_flat.view(-1, BALL_QUANTITY, 2)
        return discrete_output, continuous_output


class RLAgent:
    def __init__(self,
                 model_path: str = CHECKPOINT_PATH_1PLAYER,
                 db: dbHandler | None = None,
                 device: str = None,
                 replay_size: int = 20000,
                 lr: float = 4e-5,
                 epsilon_start: float = 0.5,
                 epsilon_final: float = 0.02,
                 epsilon_decay_steps: int = 20000,
                 gamma: float = 0.99,
                 game: Game | None = None) -> None:
        
        self.model_path = model_path
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.db = db if db is not None else dbHandler(TRAINING_DATA_PATH_1PLAYER)

        self.model = DualHeadMLPModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.criterion_ce = nn.CrossEntropyLoss(reduction='none') 
        self.criterion_mse = nn.MSELoss(reduction='none') 
        
        self.episode_history = [] 
        self.replay = deque(maxlen=replay_size) 
        self.step_count = 0
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay_steps

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            print("No checkpoint found or incompatible. Starting fresh.")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        if game is None:
            self.game = Game(screen=self.screen, db=self.db, debug=False, off_screen=True)
        else:
            self.game = game

    def get_state(self) -> np.ndarray:
        white = self.game.balls[0]
        wx_norm = white.coords.x / WIDTH
        wy_norm = white.coords.y / HEIGHT
        state_features = []
        state_features.extend([wx_norm, wy_norm])
        for i in range(1, BALL_QUANTITY + 1):
            ball = next((b for b in self.game.balls if b.index == i), None)
            if ball is None or not ball.active:
                state_features.extend([0.0, 0.0])
            else:
                bx_norm = ball.coords.x / WIDTH
                by_norm = ball.coords.y / HEIGHT
                dx = bx_norm - wx_norm
                dy = by_norm - wy_norm
                state_features.extend([dx, dy])

        return np.array(state_features, dtype=np.float32)

    def _state_tensor(self, state_np):
        return torch.tensor(state_np, dtype=torch.float32, device=self.device)

    def predict(self, state: np.ndarray, add_noise: bool = True) -> tuple[int, float, float]:
        active_balls_indices = [b.index for b in self.game.balls if b.index != 0 and b.active]

        if not active_balls_indices:
            return 1, 0.0, 0.5

        if add_noise and random.random() < self.epsilon:
            target_idx = random.choice(active_balls_indices)

            delta_angle_norm = random.uniform(-1.0, 1.0)
            power_norm = random.uniform(0.1, 1.0)
            
            return target_idx, delta_angle_norm, power_norm

        self.model.eval()
        with torch.no_grad():
            x = self._state_tensor(state).unsqueeze(0)
            discrete_out, continuous_out = self.model(x)
        
        mask = torch.full(discrete_out.shape, float('-inf'), device=self.device)
        
        for idx in active_balls_indices:
            mask[0, idx - 1] = 0 
            
        masked_logits = discrete_out + mask

        target_idx_tensor = torch.argmax(masked_logits, dim=1)
        target_idx = target_idx_tensor.item() + 1

        if target_idx not in active_balls_indices:
            print("Warning: Agent chose the ball that isn't on the deck; random ball chosen instead")
            target_idx = random.choice(active_balls_indices)

        array_idx = target_idx - 1
        specific_params = continuous_out[0, array_idx, :]
        
        delta_angle_norm = float(torch.tanh(specific_params[0]).cpu().item())
        power_norm = float(torch.sigmoid(specific_params[1]).cpu().item())

        if add_noise:
            noise_scale = self.epsilon * 0.5
            delta_angle_norm = np.clip(delta_angle_norm + np.random.normal(0, noise_scale), -1.0, 1.0)
            power_norm = np.clip(power_norm + np.random.normal(0, noise_scale * 0.5), 0.0, 1.0)

        return target_idx, delta_angle_norm, power_norm

    def update_epsilon(self):
        self.epsilon = max(self.eps_final, self.eps_start - (self.step_count / self.eps_decay) * (self.eps_start - self.eps_final))

    def save_checkpoint(self) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.model_path)

    def store_transition(self, state: np.ndarray, action: tuple, reward: float) -> None:
        target_idx, delta_angle_norm, power_norm = action 
        continuous_action = np.array([delta_angle_norm, power_norm], dtype=np.float32)
        target_class_idx = np.array(target_idx - 1, dtype=np.int64) 
        self.episode_history.append(
            (state.copy(), target_class_idx, continuous_action, float(reward))
        )

    def train_from_episode(self) -> float | None:
        if not self.episode_history:
            return None

        rewards = [r for _, _, _, r in self.episode_history]
        G = 0.0
        discounted_returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_returns.append(G)
        discounted_returns.reverse()

        states = np.stack([b[0] for b in self.episode_history])
        targets_discrete = np.stack([b[1] for b in self.episode_history])
        targets_continuous = np.stack([b[2] for b in self.episode_history])

        returns_t = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)

        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)
        else:
            returns_t = returns_t - returns_t.mean()

        states_t = self._state_tensor(states)
        targets_discrete_t = torch.tensor(targets_discrete, dtype=torch.long, device=self.device) 
        targets_continuous_t = torch.tensor(targets_continuous, dtype=torch.float32, device=self.device)

        self.model.train()
        logits_t, all_preds_continuous_t = self.model(states_t) 
        
        gather_idx = targets_discrete_t.view(-1, 1, 1).expand(-1, 1, 2)
        selected_preds_continuous = torch.gather(all_preds_continuous_t, 1, gather_idx).squeeze(1)

        log_probs = F.log_softmax(logits_t, dim=1)
        probs = F.softmax(logits_t, dim=1)         

        action_log_probs = log_probs.gather(1, targets_discrete_t.unsqueeze(1)).squeeze(1)
        loss_discrete = -(action_log_probs * returns_t).mean()
        
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_coef = 0.01

        mse_per_sample = self.criterion_mse(selected_preds_continuous, targets_continuous_t).mean(dim=1)
        positive_mask = (returns_t > 0).float()
        loss_continuous = (mse_per_sample * returns_t * positive_mask).mean()

        loss = loss_discrete + loss_continuous - (entropy_coef * entropy)

        self.optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
        self.optimizer.step()

        if len(self.episode_history) > 300 or random.random() < 0.1:
            with torch.no_grad():
                max_prob = probs.max(dim=1)[0].mean().item()

                avg_angle_sat = torch.abs(torch.tanh(all_preds_continuous_t[:, :, 0])).mean().item()
                
            print(f"\n--- DIAGNOSTICS (Ep. Length: {len(self.episode_history)}) ---")
            print(f"| Loss: {loss.item():.4f} (Disc: {loss_discrete.item():.3f}, Cont: {loss_continuous.item():.3f})")
            print(f"| Entropy: {entropy.item():.4f} (High = Exploration, Low = Fixation)")
            print(f"| Max Prob: {max_prob:.4f} (If ~1.0 -> Overconfident/Stuck Policy)")
            print(f"| Grad Norm: {total_norm:.4f} (If ~0.0 -> Dead Network/ReLU Death)")
            print(f"| Angle Saturation: {avg_angle_sat:.4f} (If ~1.0 -> Steering Lock/Tanh Saturated)")
            print("---------------------------------------------------\n")

        self.episode_history = []
        return loss.item()
    
    def play_episode(self, 
                     train_during: bool = False, 
                     max_steps: int = 1000,
                     special_mode: int = 0):

        self.game.reset(special_mode=special_mode)
        
        self.step_count = 0
        self.episode_history = []
        episode_reward = 0.0

        while self.game.flag_won is None:
            if self.step_count >= max_steps:
                break

            state = self.get_state()
            target_idx, delta_angle_norm, power_norm = self.predict(state, add_noise=True)
            score = self.game.simulate(target_idx, delta_angle_norm, power_norm) 
            episode_reward += score
            self.store_transition(state, (target_idx, delta_angle_norm, power_norm), score) 
            self.step_count += 1
            self.update_epsilon()

        final_loss = self.train_from_episode()
        print(f"Episode finished total_reward={episode_reward:.3f} final_loss={final_loss} steps={self.step_count}")
        self.save_checkpoint()