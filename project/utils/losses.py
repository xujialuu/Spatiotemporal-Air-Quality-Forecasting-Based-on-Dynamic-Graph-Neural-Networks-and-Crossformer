"""Loss functions for multi-branch forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.2,
    gamma: float = 0.6,
) -> dict[str, torch.Tensor]:
    """Compute weighted branch and fusion MSE losses with branch masking."""

    mask = outputs.get('branch_mask', {'use_spatial': True, 'use_temporal': True})
    loss_space = F.mse_loss(outputs['y_space'], target) if mask['use_spatial'] else torch.zeros((), device=target.device)
    loss_time = F.mse_loss(outputs['y_time'], target) if mask['use_temporal'] else torch.zeros((), device=target.device)
    loss_fusion = F.mse_loss(outputs['y_fusion'], target)

    alpha_eff = alpha if mask['use_spatial'] else 0.0
    beta_eff = beta if mask['use_temporal'] else 0.0
    gamma_eff = gamma
    weight_sum = alpha_eff + beta_eff + gamma_eff
    total_loss = (alpha_eff * loss_space + beta_eff * loss_time + gamma_eff * loss_fusion) / weight_sum
    return {
        'loss': total_loss,
        'loss_space': loss_space,
        'loss_time': loss_time,
        'loss_fusion': loss_fusion,
    }
