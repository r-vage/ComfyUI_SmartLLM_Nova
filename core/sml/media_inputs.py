"""Bounded visual sequences, shared by IMAGE, VIDEO and lazy disk inputs.

VIDEO decoding uses ComfyUI's trim/component interface in small chunks. This
keeps native crop/rotation/concatenation semantics without retaining a full film
or importing a companion node pack. Video times are frame-rate estimates, as
the component API does not expose presentation timestamps.
"""
from contextlib import closing
from dataclasses import dataclass
from math import ceil, sqrt

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .image_inputs import image_limits, resize_frame, sampled_positions, validate_source


@dataclass
class VisualFrame:
    pixels: torch.Tensor
    position: int
    seconds: float | None = None
    shot: int = 0


def video_inputs(value):
    """Distinguish native VIDEO objects before the legacy tensor grouping path."""
    from comfy_api.input import VideoInput

    items = []

    def visit(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif item is not None:
            items.append(item)

    visit(value)
    if not any(isinstance(item, VideoInput) for item in items):
        return None
    if not all(isinstance(item, VideoInput) for item in items):
        raise ValueError("Connect an IMAGE batch or VIDEO inputs, not a mixed list")
    return items


def check_interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()


def image_frames(value, instance, max_side, stride=1, *, disk=False):
    """Resize one source frame at a time; preserve original positional evidence."""
    count = validate_source(value) if disk else len(value)
    selected = set(sampled_positions(count, stride))
    iterator = iter(value.iter_frames()) if disk else iter(value)
    actual = 0
    try:
        for index, frame in enumerate(iterator):
            check_interrupt()
            if index >= count:
                raise ValueError("Frame source yielded more frames than declared")
            actual += 1
            if index in selected:
                yield VisualFrame(resize_frame(frame, instance, max_side), index)
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
    if actual != count:
        raise ValueError("Frame source yielded fewer frames than declared")


def _video_components(video):
    """Decode at most 32 MiB of RGB float frames per native trim (one minimum).

    Tensor videos already own their storage; views need no re-encoding. Native
    concatenations are traversed by part so joins are reliable shot boundaries.
    """
    from comfy_api.input_impl.video_types import VideoFromComponents, VideoFromList

    if isinstance(video, VideoFromList):
        elapsed, position = 0.0, 0
        for part_index, part in enumerate(video.videos):
            count = 0
            with closing(_video_components(part)) as frames:
                for frame, index, seconds, _ in frames:
                    count += 1
                    yield frame, position + index, elapsed + seconds, part_index
            position += count
            elapsed += float(part.get_duration())
        return
    fps = float(video.get_frame_rate())
    duration = float(video.get_duration())
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("VIDEO must have a positive frame rate and duration")
    if isinstance(video, VideoFromComponents):
        components = video.get_components()
        for index, frame in enumerate(components.images):
            check_interrupt()
            yield frame, index, index / fps, 0
        return
    width, height = video.get_dimensions()
    chunk_frames = max(1, min(32, (32 * 1024**2) // (max(1, width * height) * 12)))
    chunk_seconds = chunk_frames / fps
    position = 0
    for chunk in range(ceil(duration / chunk_seconds)):
        check_interrupt()
        start = chunk * chunk_seconds
        # Tiny positive tolerance keeps float conversion from rounding a boundary
        # down by one presentation tick in the native trim implementation.
        clip = video.as_trimmed(start_time=start + 1e-8, duration=min(chunk_seconds, duration - start),
                                strict_duration=False)
        if clip is None:
            continue
        components = clip.get_components()
        for frame in components.images:
            yield frame, position, position / fps, 0
            position += 1
        del components, clip
    if not position:
        raise ValueError("VIDEO contains no decodable frames")


def video_frames(video, instance, max_side, sample_fps=2.0, cut_threshold=27.0):
    """Sample chronologically, retaining both endpoints of each detected shot.

    Adjacent-frame HSV change is a cut heuristic, not a claim of perfect scene
    detection. Zero disables it for continuous camera movement. Native video
    concatenation boundaries remain explicit even with detection disabled.
    """
    if sample_fps <= 0 or cut_threshold < 0:
        raise ValueError("Video sample rate must be positive and cut threshold nonnegative")
    previous = None
    previous_hsv = None
    last_selected = -1
    next_time = 0.0
    shot, last_part = 0, None
    with closing(_video_components(video)) as decoded:
        for pixels, index, seconds, part in decoded:
            check_interrupt()
            thumb = cv2.resize(pixels.detach().cpu().numpy(), (64, 64))
            hsv = cv2.cvtColor((np.clip(thumb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            score = float(np.abs(hsv.astype(np.int16) - previous_hsv).mean()) if previous_hsv is not None else 0
            cut = previous is not None and (part != last_part or cut_threshold > 0 and score >= cut_threshold)
            if cut:
                if previous.position != last_selected:
                    yield previous
                shot += 1
                next_time = seconds
            current = VisualFrame(resize_frame(pixels, instance, max_side), index, seconds, shot)
            if previous is None or cut or seconds + 1e-7 >= next_time:
                yield current
                last_selected = index
                next_time = seconds + 1 / sample_fps
            previous, previous_hsv, last_part = current, hsv.astype(np.int16), part
        if previous is not None and previous.position != last_selected:
            yield previous


def grid_capacity(instance, tile_side, grid_side, maximum):
    """Keep a 256px tile long side where the source has that much detail."""
    pixels, _, _, fixed = image_limits(instance)
    side = min(grid_side, int(sqrt(pixels)), fixed or grid_side)
    readable = min(tile_side, 256)
    return max(1, min(maximum, (side // (readable + 24)) ** 2))


def visual_windows(frames, maximum=8, overlap=2):
    """Bound memory and never cross a supplied/detected shot boundary."""
    if maximum < 1 or overlap < 0 or overlap >= maximum:
        raise ValueError("Window overlap must be smaller than its frame capacity")
    pending = []
    last_sent = -1
    try:
        for frame in frames:
            if pending and (frame.shot != pending[-1].shot or frame.pixels.shape != pending[-1].pixels.shape):
                if pending[-1].position != last_sent:
                    yield pending
                pending = []
            if len(pending) == maximum:
                yield pending
                last_sent = pending[-1].position
                pending = pending[-overlap:] if overlap else []
            pending.append(frame)
        if pending and pending[-1].position != last_sent:
            yield pending
    finally:
        close = getattr(frames, "close", None)
        if close:
            close()


def pack_grid(frames, instance, max_side=2048):
    """Letterbox tiles in row-major order, with labels outside source pixels."""
    if not frames:
        raise ValueError("Cannot build an empty image grid")
    height, width = frames[0].pixels.shape[:2]
    pixels, _, _, fixed = image_limits(instance)
    label_height, gap = 24, 4
    # Prefer a roughly square contact sheet, preserving each tile's aspect ratio.
    layouts = []
    for columns in range(1, len(frames) + 1):
        rows = ceil(len(frames) / columns)
        w, h = columns * (width + gap) + gap, rows * (height + label_height + gap) + gap
        scale = min(1.0, max_side / max(w, h), sqrt(pixels / (w * h)),
                    fixed / max(w, h) if fixed else 1.0)
        layouts.append((scale, -abs(w - h), columns, rows))
    scale, _, columns, rows = max(layouts)
    tile_w, tile_h = max(1, int(width * scale)), max(1, int(height * scale))
    canvas = Image.new("RGB", (columns * (tile_w + gap) + gap,
                               rows * (tile_h + label_height + gap) + gap), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # Pillow before the scalable default font API.
        font = ImageFont.load_default()
    tiles = []
    for index, frame in enumerate(frames):
        x = gap + index % columns * (tile_w + gap)
        y = gap + index // columns * (tile_h + label_height + gap)
        label = f"{index + 1} | f{frame.position}"
        if frame.seconds is not None:
            label += f" | {frame.seconds:.2f}s"
        draw.text((x + 2, y + 2), label, font=font, fill=(255, 255, 255))
        with Image.fromarray(frame.pixels.mul(255).round().byte().numpy()) as tile:
            tile.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x + (tile_w - tile.width) // 2, y + label_height + (tile_h - tile.height) // 2))
        tiles.append({"tile": index + 1, "position": frame.position, "seconds": frame.seconds,
                      "row": index // columns, "column": index % columns})
    # Account for fixed-size gutters after scaling. Limits are enforced again
    # here, before any backend-specific image encoder sees the contact sheet.
    with canvas:
        tensor = resize_frame(np.asarray(canvas), instance, max_side).unsqueeze(0)
    return tensor, {"columns": columns, "rows": rows, "width": tensor.shape[2],
                    "height": tensor.shape[1], "tiles": tiles}
