"""Generation workspace reserved before ComfyUI places Qwen3.5 GGUF weights."""

import math
from contextlib import contextmanager

import torch


def _image_patches(image):
    # Match installed Qwen3.5 process_qwen2vl_images without preprocessing pixels.
    height, width = image.shape[-3:-1]
    factor = 32  # 16-pixel patches, merged 2x2
    resized_h, resized_w = round(height / factor) * factor, round(width / factor) * factor
    if resized_h * resized_w > 12845056:
        scale = math.sqrt(height * width / 12845056)
        resized_h = max(factor, math.floor(height / scale / factor) * factor)
        resized_w = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_h * resized_w < 3136:
        scale = math.sqrt(3136 / (height * width))
        resized_h = math.ceil(height * scale / factor) * factor
        resized_w = math.ceil(width * scale / factor) * factor
    return (resized_h // 16) * (resized_w // 16)


def qwen35_generation_memory(transformer, tokens, max_tokens, *, execution_bytes):
    """Conservative peak estimate, excluding resident weights and Comfy's reserve.

    Native generation allocates its cache for prompt + max_tokens up front.
    DeltaNet prefill keeps several fp32 Q/K/V/chunk workspaces alive, and GGUF
    casts/dequantization need temporary weights even when storage is quantized.
    Image placeholders expand into merged visual tokens before language prefill.
    """
    cfg = transformer.model.config
    batches = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
    lengths, largest_image = [], 0
    for batch in batches:
        length = 0
        for token, *_ in batch:
            if isinstance(token, dict) and token.get("type") == "image":
                patches = _image_patches(token["data"])
                largest_image = max(largest_image, patches)
                length += patches // 4
            elif isinstance(token, dict) and token.get("type") == "embedding":
                length += math.prod(token["data"].shape[:-1])
            elif torch.is_tensor(token):
                length += math.prod(token.shape[:-1])
            else:
                length += 1
        lengths.append(length)
    batch_size, positions = max(1, len(lengths)), max(lengths, default=1)
    capacity = positions + max_tokens + 8  # native MTP draft window
    linear_layers = cfg.layer_types.count("linear_attention")
    full_layers = cfg.num_hidden_layers - linear_layers + int(transformer.mtp is not None)
    kv = 2 * batch_size * full_layers * cfg.num_key_value_heads * cfg.head_dim * capacity * execution_bytes
    value_heads = cfg.linear_num_value_heads
    key_dim, value_dim = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    conv_dim = 2 * cfg.linear_num_key_heads * key_dim + value_heads * value_dim
    recurrent = batch_size * linear_layers * (
        value_heads * key_dim * value_dim * 4
        + conv_dim * (cfg.conv_kernel_size + 32) * execution_bytes
    )
    padded = math.ceil(positions / 64) * 64
    delta = batch_size * padded * value_heads * (12 * key_dim + 8 * value_dim + 4 * 64) * 4
    feed_forward = batch_size * positions * cfg.intermediate_size * execution_bytes * 4
    residual = batch_size * positions * cfg.hidden_size * 4 * 6
    # Source/cast buffers can coexist. Count logical shapes, not packed GGUF bytes.
    language_cast, head_cast, vision_cast = 0, 0, 0
    for name, parameter in transformer.named_parameters():
        if name.startswith("visual."):
            vision_cast = max(vision_cast, math.prod(parameter.shape) * 4 * 2)
        elif name in ("model.lm_head.weight", "model.embed_tokens.weight"):
            head_cast = max(head_cast, math.prod(parameter.shape) * execution_bytes * 2)
        else:
            language_cast = max(language_cast, math.prod(parameter.shape) * execution_bytes * 2)
    # The vision tower executes in fp32. Bound simultaneous attention/MLP and
    # image preprocessing buffers; each image is processed sequentially.
    visual = transformer.visual
    width = visual.pos_embed.weight.shape[-1]
    vision_ffn = visual.blocks[0].mlp.linear_fc1.weight.shape[0]
    vision = largest_image * (10 * width + 3 * vision_ffn + 16 * 16 * 3 * 4) * 4
    decoder = max(delta, feed_forward) + residual
    # Prefill and the vocabulary-head projection run at separate times; the
    # cache survives both. A 512-MiB margin covers allocator/kernel workspaces.
    return int(kv + recurrent + max(decoder + language_cast, head_cast, vision + vision_cast) + 512 * 1024**2)


@contextmanager
def gguf_generation_reserve(clip, tokens, max_tokens, architecture):
    reserve = getattr(clip.patcher, "reserve_generation_memory", None)
    if reserve is None or not architecture.startswith("QWEN35_"):
        yield
        return
    import comfy.model_management as mm

    device = clip.patcher.load_device
    if mm.is_device_cpu(device) or device == clip.patcher.offload_device:
        yield
        return
    transformer = getattr(clip.cond_stage_model, clip.cond_stage_model.clip).transformer
    memory = qwen35_generation_memory(
        transformer, tokens, max_tokens,
        execution_bytes=2 if mm.should_use_bf16(device) else 4,
    )
    with reserve(memory):
        yield
