"""Crossformer baseline wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.models.temporal_branch import TemporalBranch  # noqa: E402


class CrossformerBaseline(nn.Module):
    """Temporal-only Crossformer baseline.

    Input: x [B, T, N, F]
    Output: y_pred [B, K, N, C]
    """

    def __init__(self, num_nodes: int, num_features: int, input_length: int, pred_length: int, out_dim: int, target_feature_indices: list[int], patch_len: int, d_model: int, num_heads: int, num_layers: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.model = TemporalBranch(
            num_nodes=num_nodes,
            num_features=num_features,
            input_length=input_length,
            pred_length=pred_length,
            target_feature_indices=target_feature_indices,
            patch_len=patch_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            out_dim=out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
