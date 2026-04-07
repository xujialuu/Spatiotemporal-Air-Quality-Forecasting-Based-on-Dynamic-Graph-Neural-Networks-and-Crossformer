"""GRU baseline for multi-step air quality forecasting."""

from __future__ import annotations

import torch
from torch import nn


class GRUBaseline(nn.Module):
    """Sequence baseline over flattened spatiotemporal features.

    Input: x [B, T, N, F]
    Output: y_pred [B, K, N, C]
    """

    def __init__(self, num_nodes: int, num_features: int, hidden_dim: int, num_layers: int, dropout: float, pred_length: int, out_dim: int) -> None:
        super().__init__()
        input_dim = num_nodes * num_features
        self.num_nodes = num_nodes
        self.pred_length = pred_length
        self.out_dim = out_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, pred_length * num_nodes * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, num_features = x.shape
        x = x.view(batch_size, seq_len, num_nodes * num_features)
        output, _ = self.gru(x)
        h_last = output[:, -1, :]
        pred = self.head(h_last)
        pred = pred.view(batch_size, self.pred_length, self.num_nodes, self.out_dim)
        return pred
