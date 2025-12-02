import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
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

        self.actor_continuous = nn.Linear(hidden_dims[2], 2) 

    def forward(self, x):
        features = self.trunk(x)
        discrete_output = self.actor_discrete(features)
        continuous_output = self.actor_continuous(features)
        return discrete_output, continuous_output



class RLAgent:
    def __init__(self,
                 model_path: str = CHECKPOINT_PATH_1PLAYER,
                 db: dbHandler | None = None,
                 device: str = None,
                 replay_size: int = 20000,
                 lr: float = 3e-4,
                 epsilon_start: float = 0.2,
                 epsilon_final: float = 0.02,
                 epsilon_decay_steps: int = 20000,
                 gamma: float = 0.99,
                 game: Game | None = None):
        
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
            pass

        if game is None:
            self.game = Game(screen=None, db=self.db, debug=False)
        else:
            self.game = game

    def get_state(self) -> np.ndarray:
        coords = []
        for i in range(BALL_QUANTITY + 1):
            found = next((b for b in self.game.balls if b.index == i), None)
            x = found.coords.x / WIDTH if found is not None else -1.0
            y = found.coords.y / HEIGHT if found is not None else -1.0
            coords.extend([x, y])
        while len(coords) < (BALL_QUANTITY + 1) * 2:
            coords.extend([-1.0, -1.0])
        return np.array(coords, dtype=np.float32)

    def _state_tensor(self, state_np):
        return torch.tensor(state_np, dtype=torch.float32, device=self.device)

    def predict(self, state: np.ndarray, add_noise: bool = True) -> tuple[int, float, float]:
        """Zwraca (Target_Idx, Delta_Angle_Norm, Power_Norm)."""
        
        self.model.eval()
        with torch.no_grad():
            x = self._state_tensor(state)
            discrete_out, continuous_out = self.model(x)

            delta_angle_norm = float(torch.tanh(continuous_out[0]).cpu().item())
            power_norm = float(torch.sigmoid(continuous_out[1]).cpu().item())

        active_balls_indices = [b.index for b in self.game.balls if b.index != 0 and b.active]
        mask = torch.full(discrete_out.shape, -1e9, device=self.device) 
        for idx in active_balls_indices:
            mask[idx - 1] = 0
        masked_logits = discrete_out + mask

        if add_noise and random.random() < self.epsilon:
            probabilities = F.softmax(masked_logits, dim=0)
            target_idx_tensor = torch.multinomial(probabilities, 1) 
        else:
            target_idx_tensor = torch.argmax(masked_logits)
        target_idx = target_idx_tensor.item() + 1
        if add_noise and random.random() >= self.epsilon:
            delta_angle_norm = np.clip(delta_angle_norm + np.random.normal(0, 0.1), -1.0, 1.0)
            power_norm = np.clip(power_norm + np.random.normal(0, 0.05), 0.0, 1.0)

        return target_idx, delta_angle_norm, power_norm

    def update_epsilon(self):
        self.epsilon = max(self.eps_final, self.eps_start - (self.step_count / self.eps_decay) * (self.eps_start - self.eps_final))

    def save_checkpoint(self) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.model_path)

    def store_transition(self, state: np.ndarray, action: tuple, reward: float) -> None:
        """Przechowuje (stan, Target_Idx, Continuous_Action, nagroda) w historii epizodu."""
        
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
        states_t = self._state_tensor(states)
        targets_discrete_t = torch.tensor(targets_discrete, dtype=torch.long, device=self.device) 
        targets_continuous_t = torch.tensor(targets_continuous, dtype=torch.float32, device=self.device)

        self.model.train()
        logits_t, preds_continuous_t = self.model(states_t) 
        loss_discrete_per_sample = self.criterion_ce(logits_t, targets_discrete_t)
        loss_continuous_per_sample = self.criterion_mse(preds_continuous_t, targets_continuous_t).mean(dim=1) 
        loss_total_per_sample = loss_discrete_per_sample + loss_continuous_per_sample 
        loss = (loss_total_per_sample * returns_t.detach()).mean() 

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.episode_history = []
        return loss.item()
    
    def play_episode(self, train_during: bool = False, max_steps: int = 5000):
        self.game.__init__(screen=None, db=self.db, debug=False)
        self.step_count = 0
        self.episode_history = []
        episode_reward = 0.0

        while self.game.flag_won is None and self.step_count < max_steps:
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