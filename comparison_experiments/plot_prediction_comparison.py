from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
MODEL_ORDER = ["full_model", "gcn", "gru", "lstm"]
MODEL_LABELS = {
    "full_model": "Full Model (Ours)",
    "gcn": "GCN",
    "gru": "GRU",
    "lstm": "LSTM",
}


def load_metrics(model_name: str) -> str:
    metrics_path = OUTPUTS_DIR / model_name / "test_metrics.json"
    if not metrics_path.exists():
        return MODEL_LABELS[model_name]
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return (
        f"{MODEL_LABELS[model_name]}\n"
        f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R虏={metrics['r2']:.3f}"
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor("white")

    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        image_path = OUTPUTS_DIR / model_name / "pred_vs_true.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(load_metrics(model_name), fontsize=11, pad=10)
        ax.axis("off")

    fig.suptitle("Prediction vs Ground Truth Comparison", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    png_path = OUTPUTS_DIR / "pred_vs_true_comparison.png"
    pdf_path = OUTPUTS_DIR / "pred_vs_true_comparison.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
