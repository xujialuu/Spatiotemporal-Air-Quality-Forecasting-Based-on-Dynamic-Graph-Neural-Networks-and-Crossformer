"""Crossformer-style patch embedding and two-stage attention blocks."""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify [B, T, V] into [B, P, V, D]."""

    def __init__(self, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: [B, T, V]
        Returns:
        patches: [B, P, V, D]
        """

        assert x.ndim == 3
        batch_size, seq_len, num_vars = x.shape
        pad_len = (self.patch_len - seq_len % self.patch_len) % self.patch_len
        if pad_len > 0:
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x = torch.cat([x, pad], dim=1)
        num_patches = x.size(1) // self.patch_len
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_vars, num_patches, self.patch_len)
        x = self.proj(x)
        return x.permute(0, 2, 1, 3).contiguous()


class FeedForward(nn.Module):
    """Transformer FFN."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStageAttentionBlock(nn.Module):
    """Time attention followed by variable attention.

    Input/Output:
        x: [B, P, V, D]
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(d_model)
        self.time_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.var_norm = nn.LayerNorm(d_model)
        self.var_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, num_vars, d_model = x.shape

        time_in = self.time_norm(x).permute(0, 2, 1, 3).reshape(batch_size * num_vars, num_patches, d_model)
        time_out, _ = self.time_attn(time_in, time_in, time_in, need_weights=False)
        time_out = time_out.view(batch_size, num_vars, num_patches, d_model).permute(0, 2, 1, 3)
        x = x + time_out

        var_in = self.var_norm(x).reshape(batch_size * num_patches, num_vars, d_model)
        var_out, _ = self.var_attn(var_in, var_in, var_in, need_weights=False)
        var_out = var_out.view(batch_size, num_patches, num_vars, d_model)
        x = x + var_out

        x = x + self.ffn(self.ffn_norm(x))
        return x
