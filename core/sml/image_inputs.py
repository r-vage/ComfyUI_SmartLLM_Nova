"""Per-request image grouping and bounded disk-frame input protocol.

SMARTLLM_FRAME_SOURCE v1 supplies ``iter_frames()`` yielding RGB uint8 arrays
and ``len(source)``. It owns its disk files; consumers never retain its pixels.
No companion pack is imported here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from math import sqrt

import numpy as np
import torch
from PIL import Image

from .vlm_detection import get_max_pixels_for_model_type

IMAGE_HANDLING = ("individual_images", "frame_sequence", "legacy")
sequence_request = ContextVar("smartllm_sequence_request", default=False)


@contextmanager
def sequence_context(enabled):
    token = sequence_request.set(enabled)
    try:
        yield
    finally:
        sequence_request.reset(token)


def sampled_positions(count, every_nth=1):
    if every_nth < 1:
        raise ValueError("Every Nth frame must be positive")
    return tuple(sorted({*range(0, count, every_nth), count - 1})) if count else ()


def group_images(images, handling, task):
    """Keep explicit choices independent of task names; preserve old batches."""
    if handling not in IMAGE_HANDLING:
        raise ValueError("Unknown image handling mode")
    batches = []

    def visit(item):
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif item is not None:
            if not isinstance(item, torch.Tensor) or item.ndim not in (3, 4):
                raise ValueError("Images must be RGB tensors")
            batch = item.unsqueeze(0) if item.ndim == 3 else item
            if batch.shape[-1] != 3 or not len(batch):
                raise ValueError("Images must contain at least one RGB frame")
            batches.append(batch)

    visit(images)
    if handling == "frame_sequence" and batches:
        if len({tuple(batch.shape[1:]) for batch in batches}) != 1:
            raise ValueError("Together requires matching frame dimensions; resize the images first")
        return [torch.cat(batches)] if len(batches) > 1 else batches
    if handling == "legacy" and any(word in task.lower() for word in ("video", "timeline")):
        return batches
    return [batch[index:index + 1] for batch in batches for index in range(len(batch))]


def image_limits(instance):
    """Model-specific ceilings; processors still own their supported layout."""
    architecture = getattr(instance, "architecture", "").upper()
    model_type = getattr(instance, "model_type", None)
    maximum = get_max_pixels_for_model_type(model_type)
    factor, minimum, fixed_side = 1, 1, 0
    if "QWEN" in architecture and ("VL" in architecture or "35" in architecture):
        maximum, factor, minimum = 12845056, 32 if "QWEN3" in architecture else 28, 3136
    processor = getattr(getattr(instance, "processor", None), "image_processor", None)
    if processor is not None:
        maximum = min(maximum, int(getattr(processor, "max_pixels", maximum)))
        minimum = int(getattr(processor, "min_pixels", 1))
        patch = getattr(processor, "patch_size", 1)
        merge = getattr(processor, "merge_size", 1)
        if isinstance(patch, int) and isinstance(merge, int):
            factor = patch * merge
        size = getattr(processor, "size", {})
        if isinstance(size, dict):
            fixed_side = max(size.get("height", 0), size.get("width", 0))
            maximum = min(maximum, size.get("longest_edge", maximum) ** 2)
    return maximum, factor, minimum, fixed_side


def resize_frame(frame, instance, max_side):
    """Resize one frame before batching. Zero retains automatic model sizing."""
    if type(max_side) is not int or max_side < 0:
        raise ValueError("Longest side must be a nonnegative integer")
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Frame source must yield HWC RGB arrays")
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
    height, width = frame.shape[:2]
    if min(width, height) < 1:
        raise ValueError("Frames must have nonempty dimensions")
    maximum, factor, minimum, fixed_side = image_limits(instance)
    if max_side and fixed_side > max_side:
        raise ValueError(f"This processor requires {fixed_side}px images; Longest side {max_side} is too small")
    scale = min(1.0, sqrt(maximum / (width * height)), max_side / max(width, height) if max_side else 1.0)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    if max_side and factor > 1:
        new_w, new_h = (new_w // factor) * factor, (new_h // factor) * factor
        if min(new_w, new_h) < factor or new_w * new_h < minimum:
            raise ValueError("Longest side is too small for this model's minimum image/patch size; increase it or use 0")
    if (new_w, new_h) != (width, height):
        with Image.fromarray(frame) as image:
            frame = np.asarray(image.resize((new_w, new_h), Image.Resampling.LANCZOS)).copy()
    return torch.from_numpy(np.array(frame, copy=True)).float().div_(255)


def prepare_images(image, instance, max_side):
    if image is None:
        return None
    # Preserve exact legacy automatic processing and ordinary output passthrough.
    if not max_side:
        return image
    batch = image.unsqueeze(0) if image.ndim == 3 else image
    return torch.stack([resize_frame(frame, instance, max_side) for frame in batch])


def materialize_source(source, instance, max_side, every_nth=1):
    if getattr(source, "version", None) != 1 or not callable(getattr(source, "iter_frames", None)):
        raise ValueError("Expected SMARTLLM_FRAME_SOURCE version 1 with iter_frames()")
    if not 1 <= len(source) <= 256:
        raise ValueError("Frame source must contain 1–256 frames; split longer sequences into windows")
    iterator = iter(source.iter_frames())
    frames = []
    selected = set(sampled_positions(len(source), every_nth))
    count = 0
    try:
        for index, frame in enumerate(iterator):
            if index >= len(source):
                raise ValueError("Frame source yielded more frames than declared")
            count += 1
            if index in selected:
                frames.append(resize_frame(frame, instance, max_side))
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
    if count != len(source):
        raise ValueError("Frame source yielded fewer frames than declared")
    if len({tuple(frame.shape) for frame in frames}) != 1:
        raise ValueError("Frame source images must have matching dimensions")
    return torch.stack(frames)
