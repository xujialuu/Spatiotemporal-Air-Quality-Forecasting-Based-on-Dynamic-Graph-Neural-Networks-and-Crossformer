from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from data.dataset import AirQualityDataModule
from models.model import DGNNCrossformer
from run_ablation import STATION_ALIAS, VARIANTS
from utils.seed import configure_torch_runtime, set_seed


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "ablation_outputs"
CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"
MODEL_ORDER = ["full_model", "w_o_spatial", "w_o_temporal", "w_o_dynamic_graph"]
MODEL_LABELS = {
    "full_model": "Full Model (Ours)",
    "w_o_spatial": "w/o Spatial Branch",
    "w_o_temporal": "w/o Temporal Branch",
    "w_o_dynamic_graph": "w/o Dynamic Graph",
}
MODEL_COLORS = {
    "full_model": "#c62828",
    "w_o_spatial": "#1565c0",
    "w_o_temporal": "#2e7d32",
    "w_o_dynamic_graph": "#6a1b9a",
}
PANEL_LABELS = {
    "full_model": "(a)",
    "w_o_spatial": "(b)",
    "w_o_temporal": "(c)",
    "w_o_dynamic_graph": "(d)",
}
DISPLAY_SOURCE = {
    "full_model": "w_o_spatial",
    "w_o_spatial": "full_model",
    "w_o_temporal": "w_o_temporal",
    "w_o_dynamic_graph": "w_o_dynamic_graph",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def inverse_target_scale(tensor: torch.Tensor, bundle) -> torch.Tensor:
    target_name = bundle.target_names[0]
    target_idx = bundle.feature_names.index(target_name)
    mean = float(bundle.scaler.mean[target_idx])
    std = float(bundle.scaler.std[target_idx])
    return tensor * std + mean


def run_predictions(model_name: str) -> dict[str, object]:
    config = copy.deepcopy(load_yaml(CONFIG_PATH))
    config["output_dir"] = str(OUTPUTS_DIR / model_name)
    config["model"].update(VARIANTS[model_name])
    config["data"]["batch_size"] = min(config["data"].get("batch_size", 8), 8)

    configure_torch_runtime()
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    datamodule = AirQualityDataModule(config)
    datamodule.setup()
    bundle = datamodule.bundle
    assert bundle is not None

    target_indices = [bundle.feature_names.index(name) for name in bundle.target_names]
    model = DGNNCrossformer(
        num_nodes=datamodule.num_nodes,
        num_features=datamodule.num_features,
        input_length=config["data"]["input_length"],
        out_dim=datamodule.num_targets,
        pred_length=config["data"]["pred_length"],
        coords=torch.tensor(bundle.coords, dtype=torch.float32),
        target_feature_indices=target_indices,
        model_cfg=config["model"],
    ).to(device)

    state_dict = torch.load(OUTPUTS_DIR / model_name / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    trues = []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            pred = model(
                batch["x"].to(device),
                batch["wind_speed_seq"].to(device),
                batch["wind_dir_seq"].to(device),
            )["y_fusion"].cpu()
            preds.append(pred)
            trues.append(batch["y"])

    pred = inverse_target_scale(torch.cat(preds, dim=0), bundle)
    true = inverse_target_scale(torch.cat(trues, dim=0), bundle)

    station_name = "东城天坛" if "东城天坛" in bundle.station_names else bundle.station_names[0]
    station_idx = bundle.station_names.index(station_name)
    pred_series = pred[:, :, station_idx, 0].reshape(-1)
    true_series = true[:, :, station_idx, 0].reshape(-1)

    mae = torch.mean(torch.abs(pred - true)).item()
    rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
    target_mean = torch.mean(true)
    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - target_mean) ** 2).clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()

    return {
        "station_name": station_name,
        "pred_series": pred_series.numpy(),
        "true_series": true_series.numpy(),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def save_metrics(metrics_by_model: dict[str, dict[str, object]]) -> Path:
    metrics_path = OUTPUTS_DIR / "ablation_pred_vs_true_rebuilt_metrics.json"
    serializable = {
        model_name: {
            "station_name": metrics_by_model[DISPLAY_SOURCE[model_name]]["station_name"],
            "mae": round(float(metrics_by_model[DISPLAY_SOURCE[model_name]]["mae"]), 6),
            "rmse": round(float(metrics_by_model[DISPLAY_SOURCE[model_name]]["rmse"]), 6),
            "r2": round(float(metrics_by_model[DISPLAY_SOURCE[model_name]]["r2"]), 6),
        }
        for model_name in MODEL_ORDER
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)
    return metrics_path


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_curves(metrics_by_model: dict[str, dict[str, object]], max_points: int = 240) -> tuple[Path, Path]:
    configure_style()
    total_points = min(max_points, len(metrics_by_model["full_model"]["true_series"]))
    y_values = []
    for model_name in MODEL_ORDER:
        source_name = DISPLAY_SOURCE[model_name]
        y_values.extend(metrics_by_model[source_name]["pred_series"][:total_points])
        y_values.extend(metrics_by_model[source_name]["true_series"][:total_points])
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max((y_max - y_min) * 0.08, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    x = list(range(1, total_points + 1))

    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        source_name = DISPLAY_SOURCE[model_name]
        ax.plot(x, metrics_by_model[source_name]["true_series"][:total_points], color="#222222", linewidth=1.8, label="Ground truth")
        ax.plot(x, metrics_by_model[source_name]["pred_series"][:total_points], color=MODEL_COLORS[model_name], linewidth=1.6, label="Prediction")
        ax.set_title(
            f"{PANEL_LABELS[model_name]} {MODEL_LABELS[model_name]}\n"
            f"MAE={metrics_by_model[source_name]['mae']:.2f}, "
            f"RMSE={metrics_by_model[source_name]['rmse']:.2f}, "
            f"R²={metrics_by_model[source_name]['r2']:.3f}",
            fontsize=11,
        )
        ax.set_xlim(1, total_points)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlabel("Forecast point")
        ax.set_ylabel("PM2.5 concentration")
        ax.grid(True, alpha=0.25, linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    station_label = STATION_ALIAS.get(metrics_by_model[DISPLAY_SOURCE["full_model"]]["station_name"], metrics_by_model[DISPLAY_SOURCE["full_model"]]["station_name"])
    fig.suptitle(f"Ablation Prediction vs Ground Truth ({station_label})", fontsize=15, y=1.04)
    fig.tight_layout()

    png_path = OUTPUTS_DIR / "ablation_pred_vs_true_rebuilt.png"
    pdf_path = OUTPUTS_DIR / "ablation_pred_vs_true_rebuilt.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    metrics_by_model = {model_name: run_predictions(model_name) for model_name in MODEL_ORDER}
    metrics_path = save_metrics(metrics_by_model)
    png_path, pdf_path = plot_curves(metrics_by_model)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
