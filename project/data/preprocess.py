"""Data preprocessing helpers for air quality spatiotemporal forecasting."""

from __future__ import annotations

import math
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class StandardScaler:
    """Z-score scaler fitted on training data only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray) -> "StandardScaler":
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


def load_station_coords(xlsx_path: Path) -> Dict[str, Tuple[float, float]]:
    """Parse station coordinates from xlsx without openpyxl/pandas."""

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                texts = [node.text or "" for node in si.findall(".//a:t", ns)]
                shared_strings.append("".join(texts))
        sheet_name = next(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))
        sheet_root = ET.fromstring(archive.read(sheet_name))

    rows: List[List[str]] = []
    for row in sheet_root.findall(".//a:row", ns):
        values = []
        for cell in row.findall("a:c", ns):
            cell_type = cell.get("t")
            value_node = cell.find("a:v", ns)
            value = value_node.text if value_node is not None else ""
            if cell_type == "s" and value:
                value = shared_strings[int(value)]
            values.append(value)
        rows.append(values)

    coords: Dict[str, Tuple[float, float]] = {}
    for row in rows[2:]:
        if len(row) < 3:
            continue
        station = str(row[0]).strip()
        try:
            lon = float(row[1])
            lat = float(row[2])
        except (TypeError, ValueError):
            continue
        coords[station] = (lon, lat)
    return coords


def _forward_fill_1d(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    valid_idx = np.where(~np.isnan(out))[0]
    if valid_idx.size == 0:
        return np.zeros_like(out)
    first = valid_idx[0]
    out[:first] = out[first]
    for i in range(first + 1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def fill_missing(data: np.ndarray, strategy: str = "interpolate_ffill") -> np.ndarray:
    """Fill missing values for [T, N, F] arrays."""

    filled = data.astype(np.float32, copy=True)
    total_steps, num_nodes, num_features = filled.shape
    for n in range(num_nodes):
        for f in range(num_features):
            series = filled[:, n, f]
            if not np.isnan(series).any():
                continue
            valid_mask = ~np.isnan(series)
            if not valid_mask.any():
                filled[:, n, f] = 0.0
                continue
            if strategy == "ffill":
                filled[:, n, f] = _forward_fill_1d(series)
                continue
            x_idx = np.arange(total_steps)
            interp = np.interp(x_idx, x_idx[valid_mask], series[valid_mask])
            interp = _forward_fill_1d(interp)
            filled[:, n, f] = interp
    return filled


def split_time_indices(total_steps: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, slice]:
    """Return sequential time slices for train/val/test."""

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")
    train_end = int(total_steps * train_ratio)
    val_end = train_end + int(total_steps * val_ratio)
    val_end = min(val_end, total_steps)
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, total_steps),
    }
