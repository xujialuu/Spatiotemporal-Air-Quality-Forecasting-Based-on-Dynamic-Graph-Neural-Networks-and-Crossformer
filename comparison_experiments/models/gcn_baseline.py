"""Static GCN baseline for multi-step air quality forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.data.graph_utils import build_static_adj, normalize_adjacency  # noqa: E402
from project.models.graph_layers import GraphConv  # noqa: E402


class GCNBaseline(nn.Module):
    """Two-layer static GCN with last-step temporal aggregation.

    Input: x [B, T, N, F]
    Output: y_pred [B, K, N, C]
    """

    def __init__(self, coords: torch.Tensor, num_features: int, hidden_dim: int, dropout: float, pred_length: int, out_dim: int, sigma: float, top_k: int | None, add_self_loop: bool) -> None:
        super().__init__()
        self.pred_length = pred_length
        self.out_dim = out_dim
        self.num_nodes = coords.size(0)
        static_adj = build_static_adj(coords.float(), sigma=sigma, add_self_loop=add_self_loop, top_k=top_k)
        static_adj = normalize_adjacency(static_adj)
        self.register_buffer('adj', static_adj)
        self.gcn1 = GraphConv(num_features, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.head = nn.Linear(hidden_dim, pred_length * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, _ = x.shape
        adj = self.adj.unsqueeze(0).expand(batch_size, num_nodes, num_nodes)
        h = x[:, -1, :, :]
        h = self.gcn1(h, adj)
        h = self.gcn2(h, adj)
        pred = self.head(h)
        pred = pred.view(batch_size, num_nodes, self.pred_length, self.out_dim)
        return pred.permute(0, 2, 1, 3).contiguous()
