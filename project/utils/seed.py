"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_torch_runtime() -> None:
    """Use conservative CUDA kernels for better stability on Windows GPUs."""

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    configure_torch_runtime()
