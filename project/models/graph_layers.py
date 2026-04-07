"""Basic graph layers used by the dynamic GCN branch."""

from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    """Simple graph convolution H' = sigma(A_hat @ H @ W)."""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Args:
        x: [B, N, F]
        adj: [B, N, N]
        Returns:
        [B, N, out_dim]
        """

        assert x.ndim == 3 and adj.ndim == 3
        support = self.linear(x)
        out = torch.matmul(adj, support)
        out = self.activation(out)
        return self.dropout(out)
