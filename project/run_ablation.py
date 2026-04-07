"""Run one ablation experiment or collect all results with full_model reused from outputs/."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import yaml

from data.dataset import AirQualityDataModule
from models.model import DGNNCrossformer
from utils.plotting import save_prediction_plot
from utils.seed import set_seed, configure_torch_runtime
from utils.trainer import Trainer


VARIANTS = {
    'full_model': {},
    'w_o_spatial': {'use_spatial': False, 'use_temporal': True, 'use_dynamic_graph': True},
    'w_o_temporal': {'use_spatial': True, 'use_temporal': False, 'use_dynamic_graph': True},
    'w_o_dynamic_graph': {'use_spatial': True, 'use_temporal': True, 'use_dynamic_graph': False},
}

STATION_ALIAS = {
    '东城东四': 'Dongcheng-Dongsi',
    '东城天坛': 'Dongcheng-Tiantan',
    '西城官园': 'Xicheng-Guanyuan',
    '朝阳奥体中心': 'Chaoyang-OlympicCenter',
}

ABLATION_BATCH_SIZE = 8
FULL_MODEL_SOURCE_DIR = Path('outputs')


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_cfg)


def copy_full_model_outputs(root_output_dir: Path) -> dict[str, float]:
    src = FULL_MODEL_SOURCE_DIR.resolve()
    dst = (root_output_dir / 'full_model').resolve()
    dst.mkdir(parents=True, exist_ok=True)
    required = ['best_model.pt', 'loss_history.csv', 'loss_history.json', 'loss_curves.png']
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise FileNotFoundError(f'Missing full_model artifacts in outputs/: {missing}')
    for name in required:
        shutil.copy2(src / name, dst / name)
    metrics_path = src / 'test_metrics.json'
    if metrics_path.exists():
        shutil.copy2(metrics_path, dst / metrics_path.name)
        with metrics_path.open('r', encoding='utf-8') as handle:
            metrics = json.load(handle)
    else:
        metrics = {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')}
    pred_path = src / 'pred_vs_true.png'
    if pred_path.exists():
        shutil.copy2(pred_path, dst / pred_path.name)
    branch_pred_path = src / 'pred_vs_true_station_en.png'
    if branch_pred_path.exists():
        shutil.copy2(branch_pred_path, dst / branch_pred_path.name)
    metrics['variant'] = 'full_model'
    return metrics


def evaluate_and_plot(model, datamodule, bundle, device: torch.device, output_dir: Path) -> dict[str, float]:
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            out = model(batch['x'].to(device), batch['wind_speed_seq'].to(device), batch['wind_dir_seq'].to(device))['y_fusion'].cpu()
            preds.append(out)
            trues.append(batch['y'])
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    target_name = bundle.target_names[0]
    target_idx = bundle.feature_names.index(target_name)
    mean = float(bundle.scaler.mean[target_idx])
    std = float(bundle.scaler.std[target_idx])
    pred_real = pred * std + mean
    true_real = true * std + mean

    station_name = '东城天坛' if '东城天坛' in bundle.station_names else bundle.station_names[0]
    station_label = STATION_ALIAS.get(station_name, station_name)
    save_prediction_plot(pred_real, true_real, station_label, output_dir / 'pred_vs_true.png')

    mae = torch.mean(torch.abs(pred_real - true_real)).item()
    rmse = torch.sqrt(torch.mean((pred_real - true_real) ** 2)).item()
    target_mean = torch.mean(true_real)
    ss_res = torch.sum((true_real - pred_real) ** 2)
    ss_tot = torch.sum((true_real - target_mean) ** 2).clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
    with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


def run_variant(base_config: dict, variant_name: str, overrides: dict, root_output_dir: Path) -> dict[str, float]:
    config = copy.deepcopy(base_config)
    config['output_dir'] = str(root_output_dir / variant_name)
    config['model'].update(overrides)
    config['data']['batch_size'] = min(config['data'].get('batch_size', ABLATION_BATCH_SIZE), ABLATION_BATCH_SIZE)
    configure_torch_runtime()
    set_seed(config['seed'])
    device = resolve_device(config['device'])

    datamodule = AirQualityDataModule(config)
    datamodule.setup()
    bundle = datamodule.bundle
    assert bundle is not None
    target_indices = [bundle.feature_names.index(name) for name in bundle.target_names]

    model = DGNNCrossformer(
        num_nodes=datamodule.num_nodes,
        num_features=datamodule.num_features,
        input_length=config['data']['input_length'],
        out_dim=datamodule.num_targets,
        pred_length=config['data']['pred_length'],
        coords=torch.tensor(bundle.coords, dtype=torch.float32),
        target_feature_indices=target_indices,
        model_cfg=config['model'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    output_dir = Path(config['output_dir']).resolve()
    trainer = Trainer(model=model, optimizer=optimizer, device=device, train_cfg=config['train'], output_dir=output_dir)
    trainer.fit(datamodule.train_dataloader(), datamodule.val_dataloader())
    metrics = evaluate_and_plot(model, datamodule, bundle, device, output_dir)
    metrics['variant'] = variant_name

    del trainer, optimizer, model, datamodule, bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_summary(root_output_dir: Path, summary: list[dict[str, float]]) -> Path:
    summary_path = root_output_dir / 'ablation_summary.csv'
    with summary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['variant', 'mae', 'rmse', 'r2'])
        writer.writeheader()
        writer.writerows(summary)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--variant', type=str, default='all', choices=['all', 'full_model', 'w_o_spatial', 'w_o_temporal', 'w_o_dynamic_graph'])
    args = parser.parse_args()

    config = load_config(Path(args.config))
    root_output_dir = Path('ablation_outputs').resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    if args.variant in ('all', 'full_model'):
        print('===== Reusing full_model from outputs/ =====')
        metrics = copy_full_model_outputs(root_output_dir)
        summary.append(metrics)
        print(f"full_model: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")

    variants_to_run = []
    if args.variant == 'all':
        variants_to_run = ['w_o_spatial', 'w_o_temporal', 'w_o_dynamic_graph']
    elif args.variant != 'full_model':
        variants_to_run = [args.variant]

    for variant_name in variants_to_run:
        overrides = VARIANTS[variant_name]
        batch_size = min(config['data'].get('batch_size', ABLATION_BATCH_SIZE), ABLATION_BATCH_SIZE)
        print(f'===== Running {variant_name} (batch_size={batch_size}) =====')
        metrics = run_variant(config, variant_name, overrides, root_output_dir)
        summary.append(metrics)
        print(f"{variant_name}: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")

    summary_path = write_summary(root_output_dir, summary)
    print(summary_path)


if __name__ == '__main__':
    main()
