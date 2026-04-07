"""Training entrypoint for Dynamic-GNN + Crossformer air quality forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.dataset import AirQualityDataModule
from models.model import DGNNCrossformer
from utils.seed import set_seed, configure_torch_runtime
from utils.trainer import Trainer


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_cfg=config["train"],
        output_dir=Path(config["output_dir"]).resolve(),
    )

    trainer.fit(datamodule.train_dataloader(), datamodule.val_dataloader())
    trainer.evaluate(datamodule.test_dataloader())


if __name__ == "__main__":
    main()
