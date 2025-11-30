import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import CHECKPOINT_PATH_1PLAYER, TRAINING_DATA_PATH_1PLAYER
from src.db import dbHandler
from src.agent_rl import MLPModel


class SupervisedTrainer:
    def __init__(self, model_path: str = CHECKPOINT_PATH_1PLAYER, db_path: str = TRAINING_DATA_PATH_1PLAYER):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = dbHandler(db_path)

        self.model = MLPModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Existing checkpoint loaded for supervised fine-tuning.")
        except (FileNotFoundError, KeyError):
            print("No checkpoint found — initializing fresh model.")

    def train_from_db(self, epochs: int = 10, batch_size: int = 32):
        X, y, rewards = self.db.get_learning_data()
        if X is None:
            print("No unused training data found.")
            return

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X).to(self.device),
            torch.from_numpy(y).to(self.device),
            torch.from_numpy(rewards).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y, batch_r in dataloader:
                weights = (1.0 + torch.clamp(batch_r, min=0.0)).unsqueeze(1)
                pred = self.model(batch_x)
                loss = torch.mean(weights * (pred - batch_y) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(dataloader):.6f}")

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.model_path)
        print(f"Supervised fine-tuning completed. Checkpoint saved to {self.model_path}")