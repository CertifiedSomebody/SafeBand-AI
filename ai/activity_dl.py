from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small 1-D CNN for 3-axis accelerometer windows."""
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CNNLSTM(nn.Module):
    """Compact CNN+GRU temporal model; optional comparison model."""
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        _, h = self.gru(x)
        return self.head(h[-1])
