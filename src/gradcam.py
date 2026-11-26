"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).

Hooks are registered on construction and REMOVED after generate() to avoid
memory leaks when the object is reused across many images.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for any PyTorch model given an explicit target layer.

    Usage
    -----
        gradcam = GradCAM(model, target_layer)
        cam_array, pred_class = gradcam.generate(input_tensor)
        # cam_array: float32 numpy array in [0, 1], shape (224, 224)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._target_layer = target_layer
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management — clean install and removal
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        self._hooks.append(
            self._target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self._target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------
    # CAM generation
    # ------------------------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Run one deterministic forward-backward pass and return the CAM.

        Parameters
        ----------
        input_tensor : (1, 3, 224, 224) tensor, requires_grad will be set.
        target_class : If None, uses argmax of the model output.

        Returns
        -------
        cam          : float32 numpy array in [0, 1], shape IMAGE_SIZE.
        pred_class   : int index of the class used for backprop.
        """
        from src.config import IMAGE_SIZE

        self.model.eval()

        # Enable gradient tracking for backprop
        inp = input_tensor.detach().clone().requires_grad_(True)

        # Forward pass (hooks capture activations)
        output = self.model(inp)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Backward pass (hooks capture gradients)
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM formula
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # global avg pool
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Min-max normalise
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Bilinear upsample to input resolution
        cam = F.interpolate(
            cam, size=IMAGE_SIZE, mode="bilinear", align_corners=False
        )
        cam_np: np.ndarray = cam.squeeze().cpu().numpy().astype(np.float32)

        # Clear stored tensors to free VRAM
        self._gradients = None
        self._activations = None

        return cam_np, target_class

    def __del__(self) -> None:
        """Best-effort hook cleanup on garbage collection."""
        try:
            self._remove_hooks()
        except Exception:
            pass
