"""Minimal trainer with early stopping, checkpointing, and loss history export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .losses import compute_loss
from .metrics import compute_metrics

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _iter_loader(loader: Iterable, desc: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False)


class Trainer:
    """Simple training loop wrapper with persisted convergence traces."""

    def __init__(self, model, optimizer, device, train_cfg: Dict, output_dir: Path) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_cfg = train_cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / train_cfg["checkpoint_name"]
        self.history_csv_path = self.output_dir / "loss_history.csv"
        self.history_json_path = self.output_dir / "loss_history.json"
        self.history_plot_path = self.output_dir / "loss_curves.png"
        self.history: List[Dict[str, float]] = []

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved = {}
        for key, value in batch.items():
            if key == "coords":
                moved[key] = value
            else:
                moved[key] = value.to(self.device)
        return moved

    def run_epoch(self, loader, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        total = {"loss": 0.0, "loss_space": 0.0, "loss_time": 0.0, "loss_fusion": 0.0}
        preds = []
        targets = []
        for batch in _iter_loader(loader, desc="train" if train else "eval"):
            batch = self._move_batch(batch)
            outputs = self.model(
                x=batch["x"],
                wind_speed_seq=batch["wind_speed_seq"],
                wind_dir_seq=batch["wind_dir_seq"],
            )
            losses = compute_loss(
                outputs,
                batch["y"],
                alpha=self.train_cfg["alpha"],
                beta=self.train_cfg["beta"],
                gamma=self.train_cfg["gamma"],
            )
            if train:
                self.optimizer.zero_grad()
                losses["loss"].backward()
                self.optimizer.step()
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            for key in total:
                total[key] += float(losses[key].detach().item())
            preds.append(outputs["y_fusion"].detach().cpu())
            targets.append(batch["y"].detach().cpu())
        num_batches = max(len(loader), 1)
        metrics = compute_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
        metrics.update({key: value / num_batches for key, value in total.items()})
        return metrics

    def _record_history(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_gnn_loss": train_metrics["loss_space"],
            "train_crossformer_loss": train_metrics["loss_time"],
            "train_fusion_loss": train_metrics["loss_fusion"],
            "val_loss": val_metrics["loss"],
            "val_gnn_loss": val_metrics["loss_space"],
            "val_crossformer_loss": val_metrics["loss_time"],
            "val_fusion_loss": val_metrics["loss_fusion"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        self.history.append(record)

    def _save_history(self) -> None:
        if not self.history:
            return
        fieldnames = list(self.history[0].keys())
        with self.history_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        with self.history_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)
        self._save_plot()

    def _save_plot(self) -> None:
        if plt is None or not self.history:
            return
        epochs = [row["epoch"] for row in self.history]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        axes[0].plot(epochs, [row["train_gnn_loss"] for row in self.history], label="Train GNN", linewidth=2)
        axes[0].plot(epochs, [row["val_gnn_loss"] for row in self.history], label="Val GNN", linewidth=2)
        axes[0].plot(epochs, [row["train_crossformer_loss"] for row in self.history], label="Train Crossformer", linewidth=2)
        axes[0].plot(epochs, [row["val_crossformer_loss"] for row in self.history], label="Val Crossformer", linewidth=2)
        axes[0].set_title("Branch Loss Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, [row["train_loss"] for row in self.history], label="Train Total", linewidth=2)
        axes[1].plot(epochs, [row["val_loss"] for row in self.history], label="Val Total", linewidth=2)
        axes[1].plot(epochs, [row["train_fusion_loss"] for row in self.history], label="Train Fusion", linewidth=2)
        axes[1].plot(epochs, [row["val_fusion_loss"] for row in self.history], label="Val Fusion", linewidth=2)
        axes[1].set_title("Total/Fusion Loss Convergence")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(self.history_plot_path, bbox_inches="tight")
        plt.close(fig)

    def fit(self, train_loader, val_loader) -> None:
        best_val = float("inf")
        bad_epochs = 0
        self.history = []
        for epoch in range(1, self.train_cfg["max_epochs"] + 1):
            train_metrics = self.run_epoch(train_loader, train=True)
            val_metrics = self.run_epoch(val_loader, train=False)
            self._record_history(epoch, train_metrics, val_metrics)
            self._save_history()
            print(
                f"Epoch {epoch:03d} | "
                f"train_total={train_metrics['loss']:.4f} val_total={val_metrics['loss']:.4f} | "
                f"train_gnn={train_metrics['loss_space']:.4f} val_gnn={val_metrics['loss_space']:.4f} | "
                f"train_cross={train_metrics['loss_time']:.4f} val_cross={val_metrics['loss_time']:.4f} | "
                f"train_fusion={train_metrics['loss_fusion']:.4f} val_fusion={val_metrics['loss_fusion']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f} val_r2={val_metrics['r2']:.4f}"
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                bad_epochs = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                bad_epochs += 1
            if bad_epochs >= self.train_cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break
        self._save_history()

    def evaluate(self, loader) -> Dict[str, float]:
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        metrics = self.run_epoch(loader, train=False)
        print(
            f"Test | loss={metrics['loss']:.4f} mae={metrics['mae']:.4f} "
            f"rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}"
        )
        return metrics
