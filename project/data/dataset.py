"""Dataset and datamodule for multi-station air quality forecasting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess import StandardScaler, fill_missing, load_station_coords, split_time_indices


class AirQualityWindowDataset(Dataset):
    """Sliding-window dataset returning x, y, wind sequences, and coords."""

    def __init__(
        self,
        data: np.ndarray,
        target: np.ndarray,
        coords: np.ndarray,
        input_length: int,
        pred_length: int,
        wind_speed_idx: int,
        wind_dir_idx: int,
    ) -> None:
        self.data = data.astype(np.float32)
        self.target = target.astype(np.float32)
        self.coords = coords.astype(np.float32)
        self.input_length = input_length
        self.pred_length = pred_length
        self.wind_speed_idx = wind_speed_idx
        self.wind_dir_idx = wind_dir_idx
        total_steps = data.shape[0]
        self.indices = list(range(0, total_steps - input_length - pred_length + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = self.indices[index]
        mid = start + self.input_length
        end = mid + self.pred_length
        x = self.data[start:mid]
        y = self.target[mid:end]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "wind_speed_seq": torch.from_numpy(x[:, :, self.wind_speed_idx]),
            "wind_dir_seq": torch.from_numpy(x[:, :, self.wind_dir_idx]),
            "coords": torch.from_numpy(self.coords),
        }


@dataclass
class DatasetBundle:
    train_dataset: AirQualityWindowDataset
    val_dataset: AirQualityWindowDataset
    test_dataset: AirQualityWindowDataset
    feature_names: List[str]
    target_names: List[str]
    station_names: List[str]
    coords: np.ndarray
    scaler: StandardScaler


class AirQualityDataModule:
    """Prepare datasets and dataloaders from station-wise csv files."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.bundle: DatasetBundle | None = None

    def _read_station_csv(self, path: Path, feature_columns: List[str]) -> tuple[list[str], np.ndarray]:
        timestamps = []
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_cols = [col for col in feature_columns if col not in reader.fieldnames]
            if missing_cols:
                raise KeyError(f"{path.name} missing columns: {missing_cols}")
            for row in reader:
                timestamps.append(f"{row['date']} {row['time']}")
                values = []
                for col in feature_columns:
                    raw = str(row.get(col, "")).strip()
                    values.append(float(raw) if raw != "" else np.nan)
                rows.append(values)
        return timestamps, np.asarray(rows, dtype=np.float32)

    def setup(self) -> None:
        data_cfg = self.config["data"]
        project_root = Path(__file__).resolve().parents[1]
        data_dir = (project_root / data_cfg["root_dir"]).resolve()
        coords_file = (project_root / data_cfg["coords_file"]).resolve()

        feature_names = list(data_cfg["feature_columns"].keys())
        feature_columns = list(data_cfg["feature_columns"].values())
        target_names = list(data_cfg["target_features"])
        target_indices = [feature_names.index(name) for name in target_names]
        wind_speed_idx = feature_names.index("wind_speed")
        wind_dir_idx = feature_names.index("wind_direction")

        coord_map = load_station_coords(coords_file)
        station_paths = [path for path in sorted(data_dir.glob("*.csv")) if path.stem in coord_map]
        if not station_paths:
            raise FileNotFoundError(f"No station csv files found in {data_dir}")

        timestamps_ref = None
        station_names = []
        station_arrays = []
        for path in station_paths:
            timestamps, features = self._read_station_csv(path, feature_columns)
            if timestamps_ref is None:
                timestamps_ref = timestamps
            elif timestamps != timestamps_ref:
                raise ValueError(f"Timestamp mismatch detected in {path.name}")
            station_names.append(path.stem)
            station_arrays.append(features)

        data = np.stack(station_arrays, axis=1)
        data = fill_missing(data, strategy=data_cfg["missing_strategy"])
        coords = np.asarray([coord_map[name] for name in station_names], dtype=np.float32)

        slices = split_time_indices(
            total_steps=data.shape[0],
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            test_ratio=data_cfg["test_ratio"],
        )
        scaler = StandardScaler.fit(data[slices["train"]].reshape(-1, data.shape[-1]))
        scaled = scaler.transform(data)
        target = scaled[..., target_indices]

        common_kwargs = {
            "coords": coords,
            "input_length": data_cfg["input_length"],
            "pred_length": data_cfg["pred_length"],
            "wind_speed_idx": wind_speed_idx,
            "wind_dir_idx": wind_dir_idx,
        }
        self.bundle = DatasetBundle(
            train_dataset=AirQualityWindowDataset(scaled[slices["train"]], target[slices["train"]], **common_kwargs),
            val_dataset=AirQualityWindowDataset(scaled[slices["val"]], target[slices["val"]], **common_kwargs),
            test_dataset=AirQualityWindowDataset(scaled[slices["test"]], target[slices["test"]], **common_kwargs),
            feature_names=feature_names,
            target_names=target_names,
            station_names=station_names,
            coords=coords,
            scaler=scaler,
        )

    @property
    def num_nodes(self) -> int:
        assert self.bundle is not None
        return len(self.bundle.station_names)

    @property
    def num_features(self) -> int:
        assert self.bundle is not None
        return len(self.bundle.feature_names)

    @property
    def num_targets(self) -> int:
        assert self.bundle is not None
        return len(self.bundle.target_names)

    def get_dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        assert self.bundle is not None
        dataset = getattr(self.bundle, f"{split}_dataset")
        return DataLoader(
            dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["data"]["num_workers"],
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", shuffle=False)
