"""Plot prediction vs ground truth curves for a trained run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_prediction_plot(pred: torch.Tensor, true: torch.Tensor, station_label: str, output_path: Path, max_points: int = 240) -> None:
    """Save a prediction-vs-truth figure.

    pred/true: [B, K, N, C] in original scale.
    """

    station_idx = 0
    pred_station = pred[:, :, station_idx, 0].reshape(-1).numpy()
    true_station = true[:, :, station_idx, 0].reshape(-1).numpy()
    global_pred = pred[:, :, :, 0].mean(dim=2).reshape(-1).numpy()
    global_true = true[:, :, :, 0].mean(dim=2).reshape(-1).numpy()
    max_points = min(max_points, len(pred_station))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160)
    axes[0].plot(true_station[:max_points], label='Ground Truth', linewidth=2)
    axes[0].plot(pred_station[:max_points], label='Prediction', linewidth=2)
    axes[0].set_title(f'PM2.5 Prediction vs Ground Truth - {station_label}')
    axes[0].set_xlabel('Forecast Points')
    axes[0].set_ylabel('PM2.5')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(global_true[:max_points], label='Ground Truth Mean', linewidth=2)
    axes[1].plot(global_pred[:max_points], label='Prediction Mean', linewidth=2)
    axes[1].set_title('PM2.5 Prediction vs Ground Truth - Mean Across Stations')
    axes[1].set_xlabel('Forecast Points')
    axes[1].set_ylabel('PM2.5')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
