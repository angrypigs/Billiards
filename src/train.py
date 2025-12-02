import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import CHECKPOINT_PATH_1PLAYER, TRAINING_DATA_PATH_1PLAYER
from src.db import dbHandler
from src.agent_rl import DualHeadMLPModel

class SupervisedTrainer:
    
    def __init__(self, model_path: str = CHECKPOINT_PATH_1PLAYER, db_path: str = TRAINING_DATA_PATH_1PLAYER):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = dbHandler(db_path)

        self.model = DualHeadMLPModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.criterion_ce = nn.CrossEntropyLoss(reduction='none')
        self.criterion_mse = nn.MSELoss(reduction='none')

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Existing compatible checkpoint loaded for supervised fine-tuning.")
        except Exception as e:
            print(f"Error loading checkpoint ({e}). Initializing fresh Dual-Head model.")

    def train_from_db(self, epochs: int = 10, batch_size: int = 32):

        X, y_full, rewards = self.db.get_learning_data() 
        if X is None or y_full.shape[1] != 3:
            print(f"Error: Data format incorrect (Y dim: {y_full.shape[1]}). Expected 3 columns.")
            return
        targets_discrete = y_full[:, 0] - 1 
        targets_continuous = y_full[:, 1:3] 
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X).to(self.device).float(),
            torch.from_numpy(targets_discrete).to(self.device).long(),
            torch.from_numpy(targets_continuous).to(self.device).float(),
            torch.from_numpy(rewards).to(self.device).float()
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y_disc, batch_y_cont, batch_r in dataloader:
                weights = (1.0 + batch_r.abs()).detach() 
                logits, pred_cont = self.model(batch_x)
                loss_disc_per_sample = self.criterion_ce(logits, batch_y_disc)
                loss_cont_per_sample = self.criterion_mse(pred_cont, batch_y_cont).mean(dim=1)
                total_loss_per_sample = loss_disc_per_sample + loss_cont_per_sample
                loss = torch.mean(total_loss_per_sample * weights)

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