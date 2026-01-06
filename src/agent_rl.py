import torch
import torch.nn as nn
import numpy as np
from src.utils import BALL_QUANTITY

class AngleRegressorModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[256, 256, 128]):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], BALL_QUANTITY)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        raw_output = self.net(x)
        return torch.tanh(raw_output)