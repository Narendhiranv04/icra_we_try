import torch
import torch.nn as nn

class QueryOnlyBaseline(nn.Module):
    def __init__(self, in_features=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        """
        x: query global feature (B, 768)
        Returns: logits (B, 1)
        """
        return self.net(x)
