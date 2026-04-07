from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR / "project"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_experiments.models.gcn_baseline import GCNBaseline
from comparison_experiments.models.gru_baseline import GRUBaseline
from comparison_experiments.models.lstm_baseline import LSTMBaseline
from project.data.dataset import AirQualityDataModule
from project.models.model import DGNNCrossformer
from project.utils.seed import configure_torch_runtime, set_seed


OUTPUTS_DIR = ROOT_DIR / "comparison_experiments" / "outputs"
BASELINE_CONFIG_PATH = ROOT_DIR / "comparison_experiments" / "configs" / "baseline.yaml"
FULL_MODEL_CONFIG_PATH = ROOT_DIR / "project" / "configs" / "default.yaml"

MODEL_ORDER = ["full_model", "gcn", "gru", "lstm"]
MODEL_LABELS = {
    "full_model": "Full Model (Ours)",
    "gcn": "GCN",
    "gru": "GRU",
    "lstm": "LSTM",
}
MODEL_COLORS = {
    "full_model": "#c62828",
    "gcn": "#1565c0",
    "gru": "#2e7d32",
    "lstm": "#6a1b9a",
}
PANEL_LABELS = {
    "full_model": "(a)",
    "gcn": "(b)",
    "gru": "(c)",
    "lstm": "(d)",
}
STATION_LABELS = {
    "涓滃煄涓滃洓": "Dongcheng-Dongsi",
    "涓滃煄澶╁潧": "Dongcheng-Tiantan",
    "瑗垮煄瀹樺洯": "Xicheng-Guanyuan",
    "鏈濋槼濂ヤ綋涓績": "Chaoyang-OlympicCenter",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def prepare_datamodule(config: dict) -> AirQualityDataModule:
    datamodule = AirQualityDataModule(config)
    datamodule.setup()
    return datamodule


def build_baseline(model_name: str, config: dict, datamodule: AirQualityDataModule, bundle) -> torch.nn.Module:
    model_cfg = config["model"]
    if model_name == "lstm":
        return LSTMBaseline(
            num_nodes=datamodule.num_nodes,
            num_features=datamodule.num_features,
            hidden_dim=model_cfg["lstm_hidden_dim"],
            num_layers=model_cfg["lstm_num_layers"],
            dropout=model_cfg["lstm_dropout"],
            pred_length=config["data"]["pred_length"],
            out_dim=datamodule.num_targets,
        )
    if model_name == "gru":
        return GRUBaseline(
            num_nodes=datamodule.num_nodes,
            num_features=datamodule.num_features,
            hidden_dim=model_cfg["gru_hidden_dim"],
            num_layers=model_cfg["gru_num_layers"],
            dropout=model_cfg["gru_dropout"],
            pred_length=config["data"]["pred_length"],
            out_dim=datamodule.num_targets,
        )
    if model_name == "gcn":
        return GCNBaseline(
            coords=torch.tensor(bundle.coords, dtype=torch.float32),
            num_features=datamodule.num_features,
            hidden_dim=model_cfg["gcn_hidden_dim"],
            dropout=model_cfg["gcn_dropout"],
            pred_length=config["data"]["pred_length"],
            out_dim=datamodule.num_targets,
            sigma=model_cfg["sigma"],
            top_k=model_cfg["top_k"],
            add_self_loop=model_cfg["add_self_loop"],
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_full_model(config: dict, datamodule: AirQualityDataModule, bundle) -> torch.nn.Module:
    target_indices = [bundle.feature_names.index(name) for name in bundle.target_names]
    return DGNNCrossformer(
        num_nodes=datamodule.num_nodes,
        num_features=datamodule.num_features,
        input_length=config["data"]["input_length"],
        out_dim=datamodule.num_targets,
        pred_length=config["data"]["pred_length"],
        coords=torch.tensor(bundle.coords, dtype=torch.float32),
        target_feature_indices=target_indices,
        model_cfg=config["model"],
    )


def inverse_target_scale(tensor: torch.Tensor, bundle) -> torch.Tensor:
    target_name = bundle.target_names[0]
    target_idx = bundle.feature_names.index(target_name)
    mean = float(bundle.scaler.mean[target_idx])
    std = float(bundle.scaler.std[target_idx])
    return tensor * std + mean


def run_predictions(model_name: str) -> dict[str, object]:
    if model_name == "full_model":
        config = load_yaml(FULL_MODEL_CONFIG_PATH)
    else:
        config = load_yaml(BASELINE_CONFIG_PATH)
        config["experiment"]["model_name"] = model_name

    configure_torch_runtime()
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    datamodule = prepare_datamodule(config)
    bundle = datamodule.bundle
    assert bundle is not None

    if model_name == "full_model":
        model = build_full_model(config, datamodule, bundle).to(device)
    else:
        model = build_baseline(model_name, config, datamodule, bundle).to(device)

    ckpt_path = OUTPUTS_DIR / model_name / "best_model.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    trues = []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            batch = {k: (v.to(device) if k != "coords" else v) for k, v in batch.items()}
            if model_name == "full_model":
                pred = model(batch["x"], batch["wind_speed_seq"], batch["wind_dir_seq"])["y_fusion"]
            else:
                pred = model(batch["x"])
            preds.append(pred.cpu())
            trues.append(batch["y"].cpu())

    pred = inverse_target_scale(torch.cat(preds, dim=0), bundle)
    true = inverse_target_scale(torch.cat(trues, dim=0), bundle)

    station_name = "涓滃煄澶╁潧" if "涓滃煄澶╁潧" in bundle.station_names else bundle.station_names[0]
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


def save_metrics(metrics_by_model: dict[str, dict[str, object]]) -> None:
    serializable = {}
    for model_name, metrics in metrics_by_model.items():
        serializable[model_name] = {
            "station_name": metrics["station_name"],
            "mae": round(float(metrics["mae"]), 6),
            "rmse": round(float(metrics["rmse"]), 6),
            "r2": round(float(metrics["r2"]), 6),
        }
    metrics_path = OUTPUTS_DIR / "pred_vs_true_rebuilt_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)


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
    reference_true = metrics_by_model["full_model"]["true_series"]
    total_points = min(max_points, len(reference_true))

    y_values = []
    for model_name in MODEL_ORDER:
        y_values.extend(metrics_by_model[model_name]["pred_series"][:total_points])
        y_values.extend(metrics_by_model[model_name]["true_series"][:total_points])
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max((y_max - y_min) * 0.08, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    x = list(range(1, total_points + 1))

    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        pred_series = metrics_by_model[model_name]["pred_series"][:total_points]
        true_series = metrics_by_model[model_name]["true_series"][:total_points]
        ax.plot(x, true_series, color="#222222", linewidth=1.8, label="Ground truth")
        ax.plot(x, pred_series, color=MODEL_COLORS[model_name], linewidth=1.6, label="Prediction")
        ax.set_title(
            f"{PANEL_LABELS[model_name]} {MODEL_LABELS[model_name]}",
            fontsize=11,
        )
        ax.set_xlim(1, total_points)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlabel("Forecast point")
        ax.set_ylabel("PM2.5 concentration")
        ax.grid(True, alpha=0.25, linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    station_name = metrics_by_model["full_model"]["station_name"]
    station_label = STATION_LABELS.get(station_name, station_name)
    fig.suptitle(f"Prediction vs Ground Truth on Test Set ({station_label})", fontsize=15, y=1.04)
    fig.tight_layout()

    png_path = OUTPUTS_DIR / "pred_vs_true_rebuilt.png"
    pdf_path = OUTPUTS_DIR / "pred_vs_true_rebuilt.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    metrics_by_model = {model_name: run_predictions(model_name) for model_name in MODEL_ORDER}
    save_metrics(metrics_by_model)
    png_path, pdf_path = plot_curves(metrics_by_model)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
