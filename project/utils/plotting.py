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
    max_points = min(max_points, len(pred_station))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5), dpi=160)
    ax.plot(true_station[:max_points], label='Ground Truth', linewidth=2)
    ax.plot(pred_station[:max_points], label='Prediction', linewidth=2)
    ax.set_title(f'PM2.5 Prediction vs Ground Truth - {station_label}')
    ax.set_xlabel('Forecast Points')
    ax.set_ylabel('PM2.5')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
