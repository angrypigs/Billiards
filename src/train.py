import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from src.utils import CHECKPOINT_PATH_1PLAYER, TRAINING_DATA_PATH_1PLAYER
from src.db import dbHandler
from src.agent_rl import AngleRegressorModel

class SupervisedTrainer:
    def __init__(self, model_path=CHECKPOINT_PATH_1PLAYER, db_path=TRAINING_DATA_PATH_1PLAYER):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = dbHandler(db_path)

        self.model = AngleRegressorModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.criterion = nn.MSELoss()

        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print(f"Loaded existing checkpoint from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint ({e}). Starting fresh.")
        else:
            print("No checkpoint found. Starting fresh.")

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def train(self, epochs=50, batch_size=256):
        print("Loading data from database...")
        X_raw, y_raw, _ = self.db.get_learning_data(limit=1000000)

        if X_raw is None: return

        print(f"Training on {len(X_raw)} samples. Device: {self.device}")

        X_t = torch.tensor(X_raw, dtype=torch.float32).to(self.device)
        targets_idx_t = torch.tensor(y_raw[:, 0] - 1, dtype=torch.long).to(self.device)
        targets_angle_t = torch.tensor(y_raw[:, 1], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, targets_idx_t, targets_angle_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_X, batch_idx, batch_angle in dataloader:
                self.optimizer.zero_grad()
                
                pred_angles_all = self.model(batch_X)

                batch_indices = torch.arange(batch_X.size(0), device=self.device)
                pred_angle_selected = pred_angles_all[batch_indices, batch_idx]

                loss = self.criterion(pred_angle_selected, batch_angle)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss (MSE): {avg_loss:.6f}")

        self.save_checkpoint()
        self.db.close()