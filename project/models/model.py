"""Top-level Dynamic-GNN + Crossformer model."""

from __future__ import annotations

import torch
from torch import nn

from .fusion import FusionHead
from .spatial_branch import SpatialBranch
from .temporal_branch import TemporalBranch


class DGNNCrossformer(nn.Module):
    """Parallel Dynamic-GCN and Crossformer forecasting model with ablation switches."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        input_length: int,
        out_dim: int,
        pred_length: int,
        coords: torch.Tensor,
        target_feature_indices: list[int],
        model_cfg: dict,
    ) -> None:
        super().__init__()
        self.use_spatial = model_cfg.get('use_spatial', True)
        self.use_temporal = model_cfg.get('use_temporal', True)
        self.use_dynamic_graph = model_cfg.get('use_dynamic_graph', True)
        if not self.use_spatial and not self.use_temporal:
            raise ValueError('At least one branch must be enabled.')

        self.spatial_branch = None
        self.temporal_branch = None
        if self.use_spatial:
            self.spatial_branch = SpatialBranch(
                num_features=num_features,
                hidden_dim=model_cfg['spatial_hidden_dim'],
                pred_length=pred_length,
                out_dim=out_dim,
                coords=coords,
                sigma=model_cfg['sigma'],
                top_k=model_cfg['top_k'],
                add_self_loop=model_cfg['add_self_loop'],
                dropout=model_cfg['spatial_dropout'],
                agg_mode=model_cfg['spatial_agg'],
                use_dynamic_graph=self.use_dynamic_graph,
            )
        if self.use_temporal:
            self.temporal_branch = TemporalBranch(
                num_nodes=num_nodes,
                num_features=num_features,
                input_length=input_length,
                pred_length=pred_length,
                target_feature_indices=target_feature_indices,
                patch_len=model_cfg['patch_len'],
                d_model=model_cfg['d_model'],
                num_heads=model_cfg['num_heads'],
                num_layers=model_cfg['num_layers'],
                ffn_dim=model_cfg['temporal_ffn_dim'],
                dropout=model_cfg['temporal_dropout'],
                out_dim=out_dim,
            )
        self.fusion_head = FusionHead(out_dim=out_dim, hidden_dim=model_cfg['fusion_hidden_dim']) if (self.use_spatial and self.use_temporal) else None

    def forward(self, x: torch.Tensor, wind_speed_seq: torch.Tensor, wind_dir_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
        x: [B, T, N, F]
        wind_speed_seq: [B, T, N]
        wind_dir_seq: [B, T, N]
        """

        y_space = self.spatial_branch(x, wind_speed_seq, wind_dir_seq) if self.use_spatial else None
        y_time = self.temporal_branch(x) if self.use_temporal else None

        if self.use_spatial and self.use_temporal:
            y_fusion = self.fusion_head(y_space, y_time)
        elif self.use_spatial:
            y_fusion = y_space
        else:
            y_fusion = y_time

        if y_space is None:
            y_space = torch.zeros_like(y_fusion)
        if y_time is None:
            y_time = torch.zeros_like(y_fusion)

        return {
            'y_space': y_space,
            'y_time': y_time,
            'y_fusion': y_fusion,
            'branch_mask': {
                'use_spatial': self.use_spatial,
                'use_temporal': self.use_temporal,
            },
        }
