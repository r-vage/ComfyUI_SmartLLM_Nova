# Centralized image helper functions for SmartLLM nodes.
# Provides tensor/PIL conversion, mask conversion, color parsing, and image batches.
#
# USAGE:
#   from ..core.image_helpers import tensor2pil, pil2tensor, image2mask, hex_to_rgb

import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore

from typing import Any

# ---------------------------------------------------------------------------
# Tensor ↔ PIL conversion
# ---------------------------------------------------------------------------


def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    # Convert a ComfyUI image tensor [1,H,W,C] or [H,W,C] to PIL Image.
    return Image.fromarray(
        np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    # Convert a PIL Image to a ComfyUI image tensor [1,H,W,C].
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image2mask(image: Image.Image) -> torch.Tensor:
    # Convert a PIL Image to a mask tensor [1,H,W] (grayscale).
    return torch.from_numpy(
        np.array(image.convert("L")).astype(np.float32) / 255.0
    ).unsqueeze(0)


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    # Parse hex color string (#RGB or #RRGGBB) → (R, G, B) as ints 0-255.
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    if len(hex_str) != 6:
        return (0, 0, 0)
    try:
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    except ValueError:
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Centralized List and Batch Utilities
# ---------------------------------------------------------------------------


def unwrap_value(val: Any, default: Any = None) -> Any:
    # Recursively unwraps a value if it is wrapped inside a list.
    if isinstance(val, list):
        if len(val) > 0:
            return unwrap_value(val[0], default)
        return default
    return val if val is not None else default


def flatten_images(images: Any) -> list[torch.Tensor]:
    # Recursively flattens nested lists of tensors or 4D/3D tensors into a list of 4D [1, H, W, C] tensors.
    flat_images = []
    def _process(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                _process(sub)
        elif isinstance(item, torch.Tensor):
            if item.dim() == 4:
                for i in range(item.shape[0]):
                    flat_images.append(item[i : i + 1])
            elif item.dim() == 3:
                flat_images.append(item.unsqueeze(0))
            else:
                flat_images.append(item)
    if images is not None:
        _process(images)
    return flat_images
