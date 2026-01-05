import torch
import torch.nn as nn
import numpy as np
from src.utils import BALL_QUANTITY

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
        continuous_reshaped = continuous_flat.view(-1, BALL_QUANTITY, 2)

        raw_angle = continuous_reshaped[:, :, 0]
        raw_power = continuous_reshaped[:, :, 1]

        angle = torch.tanh(raw_angle)
        power = torch.sigmoid(raw_power)

        continuous_output = torch.stack([angle, power], dim=2)
        
        return discrete_output, continuous_output