"""
model.py — Model loading for the OCT Analysis Pipeline.

load_model(path, device) loads the hardened DenseNet-121 state dict.
No side effects on import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


def load_model(
    path: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build a DenseNet-121 with a 4-class classifier head and load the
    project's hardened state dict.

    Parameters
    ----------
    path   : Path to the .pth file.  Defaults to config.MODEL_PATH.
    device : Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    model  : nn.Module in eval() mode on the requested device.

    Raises
    ------
    FileNotFoundError  – if the checkpoint file does not exist.
    RuntimeError       – if state_dict keys do not match (with diff shown).
    """
    from src.config import MODEL_PATH, CLASS_NAMES  # local import → no circular dep

    if path is None:
        path = MODEL_PATH
    path = Path(path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Build the skeleton ----------------------------------------
    model = tv_models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features          # 1024 for densenet121
    model.classifier = nn.Linear(num_ftrs, len(CLASS_NAMES))

    # ---- 2. Load checkpoint -------------------------------------------
    if not path.is_file():
        raise FileNotFoundError(
            f"[model.py] Checkpoint not found at: {path}\n"
            f"Make sure 'densenet_pathwayA_HARDENED.pth' is in the project root."
        )

    print(f"[model.py] Loading checkpoint from: {path}  →  device={device}")
    state_dict = torch.load(path, map_location=device)

    # Friendly mismatch reporting
    missing, unexpected = [], []
    try:
        result = model.load_state_dict(state_dict, strict=True)
        missing    = result.missing_keys
        unexpected = result.unexpected_keys
    except RuntimeError as exc:
        raise RuntimeError(
            f"[model.py] State-dict mismatch while loading {path}.\n"
            f"Original error: {exc}"
        ) from exc

    if missing or unexpected:
        import warnings
        warnings.warn(
            f"[model.py] State-dict loaded with non-fatal mismatches.\n"
            f"  Missing keys   : {missing}\n"
            f"  Unexpected keys: {unexpected}"
        )

    model.to(device)
    model.eval()
    print(f"[model.py] Model ready on {device}.")
    return model
