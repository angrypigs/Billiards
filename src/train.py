import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from src.utils import CHECKPOINT_PATH_1PLAYER, TRAINING_DATA_PATH_1PLAYER
from src.db import dbHandler
from src.agent_rl import DualHeadMLPModel

class SupervisedTrainer:
    
    def __init__(self, model_path=CHECKPOINT_PATH_1PLAYER, db_path=TRAINING_DATA_PATH_1PLAYER):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = dbHandler(db_path)

        self.model = DualHeadMLPModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

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
        targets_class_t = torch.tensor(y_raw[:, 0] - 1, dtype=torch.long).to(self.device)
        targets_cont_t = torch.tensor(y_raw[:, 1:3], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, targets_class_t, targets_cont_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_class_loss = 0.0
            total_reg_loss = 0.0

            for batch_X, batch_y_class, batch_y_cont in dataloader:
                self.optimizer.zero_grad()
                pred_class, pred_cont_all = self.model(batch_X)

                loss_class = self.criterion_class(pred_class, batch_y_class)
                
                batch_indices = torch.arange(batch_X.size(0), device=self.device)
                selected_pred_cont = pred_cont_all[batch_indices, batch_y_class]
                loss_reg = self.criterion_reg(selected_pred_cont, batch_y_cont)

                loss = loss_class + loss_reg

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_class_loss += loss_class.item()
                total_reg_loss += loss_reg.item()

            avg_loss = total_loss / len(dataloader)
            avg_reg = total_reg_loss / len(dataloader)

            scheduler.step(avg_reg)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Total: {avg_loss:.4f} | Class: {total_class_loss/len(dataloader):.4f} | Reg: {avg_reg:.4f}")

        self.save_checkpoint()
        self.db.close()
    