"""Utilities for baseline comparison experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_mean = torch.mean(target)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def save_history(history: List[Dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    csv_path = output_dir / 'loss_history.csv'
    json_path = output_dir / 'loss_history.json'
    fieldnames = list(history[0].keys())
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)


def save_loss_plot(history: List[Dict[str, float]], output_dir: Path, title: str) -> None:
    if not history:
        return
    epochs = [row['epoch'] for row in history]
    train_loss = [row['train_loss'] for row in history]
    val_loss = [row['val_loss'] for row in history]
    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', bbox_inches='tight')
    plt.close()
