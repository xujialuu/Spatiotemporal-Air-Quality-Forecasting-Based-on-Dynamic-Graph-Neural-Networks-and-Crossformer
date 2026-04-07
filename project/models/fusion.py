"""Feature-level fusion head."""

from __future__ import annotations

import torch
from torch import nn


class FusionHead(nn.Module):
    """Fuse spatial and temporal forecasts.

    Inputs:
        y_space: [B, K, N, C]
        y_time: [B, K, N, C]
    Output:
        y_fusion: [B, K, N, C]
    """

    def __init__(self, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, y_space: torch.Tensor, y_time: torch.Tensor) -> torch.Tensor:
        assert y_space.shape == y_time.shape
        fused = torch.cat([y_space, y_time], dim=-1)
        return self.mlp(fused)
