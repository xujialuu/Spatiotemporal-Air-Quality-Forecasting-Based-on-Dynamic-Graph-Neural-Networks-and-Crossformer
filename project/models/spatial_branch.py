"""Dynamic GCN spatial branch."""

from __future__ import annotations

import torch
from torch import nn

from data.graph_utils import build_dynamic_adj, build_static_adj, normalize_adjacency
from .graph_layers import GraphConv


class SpatialBranch(nn.Module):
    """Spatial branch with optional wind-driven dynamic adjacency.

    Input:
        x: [B, T, N, F]
        wind_speed_seq: [B, T, N]
        wind_dir_seq: [B, T, N]
    Output:
        y_space: [B, K, N, C]
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        pred_length: int,
        out_dim: int,
        coords: torch.Tensor,
        sigma: float,
        top_k: int | None,
        add_self_loop: bool,
        dropout: float,
        agg_mode: str = "last",
        use_dynamic_graph: bool = True,
    ) -> None:
        super().__init__()
        self.pred_length = pred_length
        self.out_dim = out_dim
        self.agg_mode = agg_mode
        self.add_self_loop = add_self_loop
        self.use_dynamic_graph = use_dynamic_graph

        coords = coords.float()
        static_adj = build_static_adj(coords, sigma=sigma, add_self_loop=add_self_loop, top_k=top_k)
        self.register_buffer("coords", coords)
        self.register_buffer("static_adj", static_adj)
        self.gcn1 = GraphConv(num_features, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.proj = nn.Linear(hidden_dim, pred_length * out_dim)

    def forward(self, x: torch.Tensor, wind_speed_seq: torch.Tensor, wind_dir_seq: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        batch_size, seq_len, num_nodes, _ = x.shape
        if self.use_dynamic_graph:
            adj = build_dynamic_adj(
                static_adj=self.static_adj,
                coords=self.coords,
                wind_speed_seq=wind_speed_seq,
                wind_dir_seq=wind_dir_seq,
                add_self_loop=self.add_self_loop,
            )
        else:
            adj = self.static_adj.view(1, 1, num_nodes, num_nodes).expand(batch_size, seq_len, num_nodes, num_nodes)
        adj = normalize_adjacency(adj)

        spatial_states = []
        for t in range(seq_len):
            h_t = x[:, t, :, :]
            a_t = adj[:, t, :, :]
            h_t = self.gcn1(h_t, a_t)
            h_t = self.gcn2(h_t, a_t)
            spatial_states.append(h_t)
        h = torch.stack(spatial_states, dim=1)
        h = h.mean(dim=1) if self.agg_mode == "mean" else h[:, -1, :, :]
        out = self.proj(h)
        out = out.view(batch_size, num_nodes, self.pred_length, self.out_dim)
        return out.permute(0, 2, 1, 3).contiguous()
