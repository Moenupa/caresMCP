"""Model-independent Grad-CAM computation and image rendering."""

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


def compute_gradcam(
    model: "Module",
    tensor: "Tensor",
    target_layer: "Module",
    class_index: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute a positive Grad-CAM map scaled to [0, 1] on the input canvas."""
    import torch

    activations: list[Tensor] = []

    def capture_activation(
        _module: Any, _inputs: tuple[Any, ...], output: "Tensor"
    ) -> None:
        activations.append(output)

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.enable_grad():
            scores = model(tensor)
            feature_map = activations[0]
            gradients = torch.autograd.grad(
                scores[0, class_index], feature_map, retain_graph=False
            )[0]
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((channel_weights * feature_map).sum(dim=1))[0]
    finally:
        hook.remove()
    heatmap = torch.nn.functional.interpolate(
        heatmap[None, None], size=output_size, mode="bilinear", align_corners=False
    )[0, 0]
    maximum = heatmap.max()
    if maximum > 0:
        heatmap = heatmap / maximum
    return heatmap.detach().cpu().numpy().clip(0, 1)


def blend_saliency(
    image: Image.Image, heatmap: np.ndarray, opacity: float = 0.45
) -> Image.Image:
    """Blend a [0, 1] heatmap with an already aligned RGB image."""
    if heatmap.shape != (image.height, image.width):
        raise ValueError("Saliency and image dimensions must match")
    red = np.clip(1.5 * heatmap, 0, 1)
    green = np.clip(1.5 - np.abs(2 * heatmap - 1.0) * 1.5, 0, 1)
    blue = np.clip(1.0 - 1.5 * heatmap, 0, 1)
    color_map = np.stack((red, green, blue), axis=-1)
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha = opacity * heatmap[..., None]
    result = np.uint8(np.clip((1 - alpha) * base + alpha * color_map, 0, 1) * 255)
    return Image.fromarray(result)
