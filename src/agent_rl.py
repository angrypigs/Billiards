import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.game import Game
from src.db import dbHandler
from src.utils import WIDTH, HEIGHT, BALL_QUANTITY, CHECKPOINT_PATH_1PLAYER, TRAINING_DATA_PATH_1PLAYER, MAX_POWER

class MLPModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


import random
from collections import deque

class RLAgent:
    def __init__(self,
                 model_path: str = CHECKPOINT_PATH_1PLAYER,
                 db: dbHandler | None = None,
                 device: str = None,
                 replay_size: int = 20000,
                 batch_size: int = 128,
                 train_every: int = 8,
                 lr: float = 3e-4,
                 epsilon_start: float = 0.2,
                 epsilon_final: float = 0.02,
                 epsilon_decay_steps: int = 20000):
        self.model_path = model_path
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print("device:", self.device)
        self.db = db if db is not None else dbHandler(TRAINING_DATA_PATH_1PLAYER)

        self.model = MLPModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.replay = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.train_every = train_every
        self.step_count = 0

        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay_steps

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Checkpoint loaded")
        except Exception:
            print("No checkpoint found, creating new model")

        self.game = Game(screen=None, db=self.db, debug=False)

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

    def predict(self, state: np.ndarray, add_noise: bool = True) -> tuple[float, float]:
        """Return denormalized (angle_deg, power) - angle in degrees, power in game units."""
        self.model.eval()
        with torch.no_grad():
            x = self._state_tensor(state)
            out = self.model(x)
            angle_norm = float(torch.tanh(out[0]).cpu().item())
            power_norm = float(torch.sigmoid(out[1]).cpu().item())

        if add_noise:
            if random.random() < self.epsilon:
                angle_norm = random.uniform(-1.0, 1.0)
                power_norm = random.uniform(0.0, 1.0)
            else:
                # gaussian noise
                angle_norm = np.clip(angle_norm + np.random.normal(0, 0.08), -1.0, 1.0)
                power_norm = np.clip(power_norm + np.random.normal(0, 0.05), 0.0, 1.0)

        angle = angle_norm * 180.0
        power = power_norm * MAX_POWER
        return angle, power

    def store_transition(self, state: np.ndarray, action: tuple, reward: float) -> None:
        angle_norm = float(np.clip(action[0] / 180.0, -1.0, 1.0))
        power_norm = float(np.clip(action[1] / MAX_POWER, 0.0, 1.0))
        self.replay.append((state.copy(), np.array([angle_norm, power_norm], dtype=np.float32), float(reward)))

    def sample_batch(self):
        batch = random.sample(self.replay, min(len(self.replay), self.batch_size))
        states = np.stack([b[0] for b in batch])
        actions = np.stack([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        return states, actions, rewards

    def train_from_replay(self, epochs: int = 1):
        if len(self.replay) < max(256, self.batch_size):
            return None
        self.model.train()
        losses = []
        for _ in range(epochs):
            states, actions, rewards = self.sample_batch()
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            preds = self.model(states_t)
            per_sample_loss = self.criterion(preds, actions_t).mean(dim=1)
            weights = 1.0 + torch.clamp(rewards_t, min=0.0, max=1.0)
            loss = (per_sample_loss * weights).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def update_epsilon(self):
        self.epsilon = max(self.eps_final, self.eps_start - (self.step_count / self.eps_decay) * (self.eps_start - self.eps_final))

    def play_episode(self, train_during: bool = True, max_steps: int = 5000):
        self.game.__init__(screen=None, db=self.db, debug=False)
        self.step_count = 0
        episode_reward = 0.0

        while self.game.flag_won is None and self.step_count < max_steps:
            state = self.get_state()
            angle, power = self.predict(state, add_noise=True)
            score = self.game.simulate(angle, power)
            episode_reward += score

            self.store_transition(state, (angle, power), score)

            if train_during and (self.step_count % self.train_every == 0):
                avg_loss = self.train_from_replay(epochs=1)
                if avg_loss is not None:
                    print(f"[train] step={self.step_count} loss={avg_loss:.6f} replay={len(self.replay)} eps={self.epsilon:.4f}")

            self.step_count += 1
            self.update_epsilon()

            print(f"step={self.step_count} score={score:.4f} angle={angle:.3f} power={power:.3f} cue={self.game.balls[0].coords}")

        final_loss = self.train_from_replay(epochs=3)
        print(f"episode finished total_reward={episode_reward:.3f} final_loss={final_loss}")
        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.model_path)
        print(f"Checkpoint saved to {self.model_path}")