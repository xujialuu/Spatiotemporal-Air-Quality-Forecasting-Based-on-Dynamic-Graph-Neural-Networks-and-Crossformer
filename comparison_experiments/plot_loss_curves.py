from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
MODEL_LABELS = {
    "full_model": "full_model (ours)",
    "gcn": "GCN",
    "gru": "GRU",
    "lstm": "LSTM",
}
MODEL_ORDER = ["full_model", "gcn", "gru", "lstm"]


def load_histories() -> dict[str, dict[str, list[float]]]:
    histories: dict[str, dict[str, list[float]]] = {}
    for model_name in MODEL_ORDER:
        csv_path = OUTPUTS_DIR / model_name / "loss_history.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"epoch", "train_loss", "val_loss"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                continue
            data = {"epoch": [], "train_loss": [], "val_loss": []}
            for row in reader:
                data["epoch"].append(float(row["epoch"]))
                data["train_loss"].append(float(row["train_loss"]))
                data["val_loss"].append(float(row["val_loss"]))
            histories[model_name] = data
    return histories


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_per_model(histories: dict[str, dict[str, list[float]]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    colors = {"train": "#1f77b4", "val": "#d62728"}

    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        df = histories[model_name]
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color=colors["train"], linewidth=2)
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color=colors["val"], linewidth=2)
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("Loss Curves for Each Model", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUTS_DIR / "loss_curves_by_model.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(histories: dict[str, dict[str, list[float]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=False, sharey=False)
    palette = {
        "full_model": "#e41a1c",
        "gcn": "#377eb8",
        "gru": "#4daf4a",
        "lstm": "#984ea3",
    }

    for model_name in MODEL_ORDER:
        df = histories[model_name]
        label = MODEL_LABELS[model_name]
        color = palette[model_name]
        axes[0].plot(df["epoch"], df["train_loss"], label=label, color=color, linewidth=2)
        axes[1].plot(df["epoch"], df["val_loss"], label=label, color=color, linewidth=2)

    axes[0].set_title("Training Loss Comparison")
    axes[1].set_title("Validation Loss Comparison")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("Model Loss Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUTS_DIR / "loss_curves_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    configure_style()
    histories = load_histories()
    missing = [model for model in MODEL_ORDER if model not in histories]
    if missing:
        raise FileNotFoundError(f"Missing valid loss history for: {', '.join(missing)}")

    per_model_path = plot_per_model(histories)
    comparison_path = plot_model_comparison(histories)
    print(f"Saved: {per_model_path}")
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
