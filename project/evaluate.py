"""Evaluation entrypoint for a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.dataset import AirQualityDataModule
from models.model import DGNNCrossformer
from utils.seed import set_seed
from utils.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    trainer = Trainer(model, optimizer, device, config["train"], Path(config["output_dir"]).resolve())
    trainer.evaluate(datamodule.test_dataloader())


if __name__ == "__main__":
    main()
