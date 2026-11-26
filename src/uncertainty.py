"""
uncertainty.py — Monte Carlo Dropout uncertainty estimation.

IMPORTANT NOTE ON METHODOLOGY
------------------------------
The hardened DenseNet-121 checkpoint was trained WITHOUT dropout in the
classifier head (it is a plain nn.Linear).  The dense block layers inside
DenseNet do contain BatchNorm layers but no explicit Dropout.

To get *any* stochastic signal we inject dropout via a forward pre-hook on
model.classifier.  This means:
  * BatchNorm layers stay in eval() mode (running stats used — correct).
  * The injected dropout rate (config.MC_P) was NOT used during training.
  * The resulting uncertainty scores are APPROXIMATIONS and the thresholds
    in config.py are UNCALIBRATED PLACEHOLDERS.

Do NOT use these scores for clinical decision-making.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CLASS_NAMES,
    MC_P,
    MC_T,
    UNCERTAINTY_LOW_MAX,
    UNCERTAINTY_MODERATE_MAX,
)


# ---------------------------------------------------------------------------
# Hook-based dropout injection
# ---------------------------------------------------------------------------

def _make_dropout_hook(p: float):
    """Return a pre-hook that applies dropout to the classifier's input."""
    def _hook(module, args):
        x = args[0]
        x = F.dropout(x, p=p, training=True)   # training=True forces active dropout
        return (x,)
    return _hook


@contextmanager
def _mc_dropout_context(model: nn.Module, p: float):
    """
    Context manager that:
      1. Keeps the model in eval() (BatchNorm uses running stats).
      2. Attaches a pre-hook to model.classifier that applies F.dropout.
      3. Removes the hook on exit.
    """
    hook = model.classifier.register_forward_pre_hook(_make_dropout_hook(p))
    try:
        yield
    finally:
        hook.remove()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    T: int = MC_T,
    p: float = MC_P,
) -> dict:
    """
    Run T stochastic forward passes and return uncertainty metrics.

    Parameters
    ----------
    model        : nn.Module, already in eval() mode.
    input_tensor : (1, 3, H, W) tensor on the correct device.
    T            : Number of MC passes.
    p            : Dropout probability injected at the classifier.

    Returns
    -------
    dict with keys:
        mean_probs      : (n_classes,) numpy array — mean softmax probabilities
        std_pred        : float — std-dev of the predicted class probability
        entropy         : float — predictive entropy (nats)
        uncertainty_level : "LOW" | "MODERATE" | "HIGH"
        mc_passes       : int — T used
    """
    device = input_tensor.device
    all_probs: list[torch.Tensor] = []

    with _mc_dropout_context(model, p):
        with torch.no_grad():
            for _ in range(T):
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)   # (1, n_classes)
                all_probs.append(probs.squeeze(0))     # (n_classes,)

    stacked = torch.stack(all_probs, dim=0)            # (T, n_classes)

    mean_probs = stacked.mean(dim=0).cpu().numpy()     # (n_classes,)
    pred_class = int(mean_probs.argmax())
    std_pred   = float(stacked[:, pred_class].std().item())

    # Predictive entropy: H = -sum(p * log(p))  over classes
    mean_t = stacked.mean(dim=0).clamp(min=1e-9)
    entropy = float(-torch.sum(mean_t * torch.log(mean_t)).item())

    # Map std to level
    if std_pred < UNCERTAINTY_LOW_MAX:
        level = "LOW"
    elif std_pred < UNCERTAINTY_MODERATE_MAX:
        level = "MODERATE"
    else:
        level = "HIGH"

    return {
        "mean_probs":        mean_probs,
        "std_pred":          std_pred,
        "entropy":           entropy,
        "uncertainty_level": level,
        "mc_passes":         T,
    }
