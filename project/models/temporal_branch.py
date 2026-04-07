"""Simplified Crossformer temporal branch."""

from __future__ import annotations

import math

import torch
from torch import nn

from .crossformer_blocks import PatchEmbedding, TwoStageAttentionBlock


class TemporalBranch(nn.Module):
    """Crossformer-like temporal branch.

    Input:
        x: [B, T, N, F]
    Output:
        y_time: [B, K, N, C]
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        input_length: int,
        pred_length: int,
        target_feature_indices: list[int],
        patch_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.pred_length = pred_length
        self.out_dim = out_dim
        self.target_feature_indices = target_feature_indices
        self.patch_embed = PatchEmbedding(patch_len=patch_len, d_model=d_model)
        self.blocks = nn.ModuleList(
            [
                TwoStageAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        num_patches = math.ceil(input_length / patch_len)
        self.head = nn.Linear(num_patches * d_model, self.pred_length)

    def _target_var_indices(self, device: torch.device) -> torch.Tensor:
        indices = []
        for node_idx in range(self.num_nodes):
            base = node_idx * self.num_features
            for feat_idx in self.target_feature_indices:
                indices.append(base + feat_idx)
        return torch.tensor(indices, device=device, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        batch_size, seq_len, num_nodes, num_features = x.shape
        assert num_nodes == self.num_nodes
        assert num_features == self.num_features

        x = x.view(batch_size, seq_len, num_nodes * num_features)
        h = self.patch_embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(h)

        num_patches = h.size(1)
        d_model = h.size(-1)
        h = h.permute(0, 2, 1, 3).contiguous()
        h = h.view(batch_size, num_nodes * num_features, num_patches * d_model)

        target_indices = self._target_var_indices(h.device)
        h_target = h.index_select(dim=1, index=target_indices)
        out = self.head(h_target)
        out = out.view(batch_size, num_nodes, self.out_dim, self.pred_length)
        return out.permute(0, 3, 1, 2).contiguous()
