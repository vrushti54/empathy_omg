import torch
import torch.nn as nn

class CNN1DRegressor(nn.Module):
    """1D CNN over time on (T, n_mels) sequences."""
    def __init__(self, n_mels, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            # input: (B, n_mels, T)
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=7, padding=3, stride=2), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average over time
        self.head = nn.Sequential(
            nn.Linear(256, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):             # x: (B, T, n_mels)
        x = x.transpose(1, 2)         # -> (B, n_mels, T)
        h = self.net(x)               # -> (B, C, T')
        h = self.pool(h).squeeze(-1)  # -> (B, C)
        y = self.head(h).squeeze(-1)  # -> (B,)
        return y
