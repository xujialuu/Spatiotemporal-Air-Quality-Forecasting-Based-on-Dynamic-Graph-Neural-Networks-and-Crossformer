"""Evaluation metrics for regression forecasting."""

from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_mean = torch.mean(target)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mae": float(mae(pred, target).item()),
        "rmse": float(rmse(pred, target).item()),
        "r2": float(r2(pred, target).item()),
    }
