"""Train one baseline model for comparison experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR / 'project'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.data.dataset import AirQualityDataModule  # noqa: E402
from project.utils.plotting import save_prediction_plot  # noqa: E402
from project.utils.seed import configure_torch_runtime, set_seed  # noqa: E402

from comparison_experiments.baseline_utils import mae, mse_loss, r2, rmse, save_history, save_loss_plot
from comparison_experiments.models.gcn_baseline import GCNBaseline
from comparison_experiments.models.gru_baseline import GRUBaseline
from comparison_experiments.models.lstm_baseline import LSTMBaseline


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_cfg)


def apply_runtime_overrides(config: dict, model_name: str) -> dict:
    runtime = config.get('runtime', {})
    overrides = runtime.get('batch_size_overrides', {})
    batch_size = overrides.get(model_name)
    if batch_size is not None:
        config['data']['batch_size'] = int(batch_size)
    return config


def build_model(model_name: str, config: dict, datamodule: AirQualityDataModule, bundle) -> torch.nn.Module:
    model_cfg = config['model']
    if model_name == 'lstm':
        return LSTMBaseline(
            num_nodes=datamodule.num_nodes,
            num_features=datamodule.num_features,
            hidden_dim=model_cfg['lstm_hidden_dim'],
            num_layers=model_cfg['lstm_num_layers'],
            dropout=model_cfg['lstm_dropout'],
            pred_length=config['data']['pred_length'],
            out_dim=datamodule.num_targets,
        )
    if model_name == 'gru':
        return GRUBaseline(
            num_nodes=datamodule.num_nodes,
            num_features=datamodule.num_features,
            hidden_dim=model_cfg['gru_hidden_dim'],
            num_layers=model_cfg['gru_num_layers'],
            dropout=model_cfg['gru_dropout'],
            pred_length=config['data']['pred_length'],
            out_dim=datamodule.num_targets,
        )
    if model_name == 'gcn':
        return GCNBaseline(
            coords=torch.tensor(bundle.coords, dtype=torch.float32),
            num_features=datamodule.num_features,
            hidden_dim=model_cfg['gcn_hidden_dim'],
            dropout=model_cfg['gcn_dropout'],
            pred_length=config['data']['pred_length'],
            out_dim=datamodule.num_targets,
            sigma=model_cfg['sigma'],
            top_k=model_cfg['top_k'],
            add_self_loop=model_cfg['add_self_loop'],
        )
    raise ValueError(f'Unsupported model: {model_name}')


def forward_model(model_name: str, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if model_name in {'lstm', 'gru', 'gcn'}:
        return model(batch['x'])
    raise ValueError(model_name)


def run_epoch(model_name: str, model: torch.nn.Module, loader, optimizer, device: torch.device, train: bool) -> dict[str, float]:
    model.train(train)
    losses = []
    preds = []
    targets = []
    for batch in loader:
        batch = {k: (v.to(device) if k != 'coords' else v) for k, v in batch.items()}
        pred = forward_model(model_name, model, batch)
        loss = mse_loss(pred, batch['y'])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize(device)
        losses.append(float(loss.detach().cpu().item()))
        preds.append(pred.detach().cpu())
        targets.append(batch['y'].detach().cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    return {
        'loss': sum(losses) / max(len(losses), 1),
        'mae': float(mae(pred_all, target_all).item()),
        'rmse': float(rmse(pred_all, target_all).item()),
        'r2': float(r2(pred_all, target_all).item()),
    }


def evaluate_and_plot(model_name: str, model: torch.nn.Module, loader, bundle, device: torch.device, output_dir: Path) -> dict[str, float]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if k != 'coords' else v) for k, v in batch.items()}
            pred = forward_model(model_name, model, batch).cpu()
            preds.append(pred)
            trues.append(batch['y'].cpu())
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    target_name = bundle.target_names[0]
    target_idx = bundle.feature_names.index(target_name)
    mean = float(bundle.scaler.mean[target_idx])
    std = float(bundle.scaler.std[target_idx])
    pred_real = pred * std + mean
    true_real = true * std + mean
    save_prediction_plot(pred_real, true_real, 'Dongcheng-Tiantan', output_dir / 'pred_vs_true.png')
    metrics = {
        'mae': float(mae(pred_real, true_real).item()),
        'rmse': float(rmse(pred_real, true_real).item()),
        'r2': float(r2(pred_real, true_real).item()),
    }
    with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='comparison_experiments/configs/baseline.yaml')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru', 'gcn'], default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.model is not None:
        config['experiment']['model_name'] = args.model
    model_name = config['experiment']['model_name']
    config = apply_runtime_overrides(config, model_name)

    configure_torch_runtime()
    set_seed(config['seed'])
    device = resolve_device(config['device'])

    datamodule = AirQualityDataModule(config)
    datamodule.setup()
    bundle = datamodule.bundle
    assert bundle is not None

    model = build_model(model_name, config, datamodule, bundle).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

    output_dir = ROOT_DIR / 'comparison_experiments' / 'outputs' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float('inf')
    bad_epochs = 0
    ckpt_path = output_dir / config['train']['checkpoint_name']

    print(f"Running {model_name} with batch_size={config['data']['batch_size']} on {device}")
    for epoch in range(1, config['train']['max_epochs'] + 1):
        train_metrics = run_epoch(model_name, model, datamodule.train_dataloader(), optimizer, device, train=True)
        val_metrics = run_epoch(model_name, model, datamodule.val_dataloader(), optimizer, device, train=False)
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
        })
        save_history(history, output_dir)
        save_loss_plot(history, output_dir, title=f'{model_name.upper()} Loss Curves')
        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f} val_r2={val_metrics['r2']:.4f}"
        )
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
        if bad_epochs >= config['train']['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    metrics = evaluate_and_plot(model_name, model, datamodule.test_dataloader(), bundle, device, output_dir)
    print(f"Test | mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}")


if __name__ == '__main__':
    main()
