"""Reverse llama.cpp's Qwen3.5 export transforms for native ComfyUI generation."""

import torch

from .dequant import dequantize_tensor, is_quantized
from .mappings import qwen35_map
from .ops import GGMLTensor


def _dense(tensor, dtype=torch.float32):
    return dequantize_tensor(tensor, dtype=dtype).as_subclass(torch.Tensor)


def _rows(tensor, indices):
    """Move whole encoded rows without interpreting quantization blocks as heads."""
    if isinstance(tensor, GGMLTensor) and is_quantized(tensor):
        raw = tensor.as_subclass(torch.Tensor).reshape(tensor.shape[0], -1)
        result = GGMLTensor(raw.index_select(0, indices.to(raw.device)),
                            tensor_type=tensor.tensor_type, tensor_shape=tensor.shape)
        result.is_largest_weight = getattr(tensor, "is_largest_weight", False)
        return result
    return tensor.as_subclass(torch.Tensor).index_select(0, indices.to(tensor.device))


def qwen35_convert(state, *, layers, key_heads):
    """Keep large projections quantized; expand parameters read by core ops directly.

    llama.cpp conversion/qwen.py groups V heads in tiled order, adds one to
    non-DeltaNet norms, exports -exp(A_log), and squeezes the depthwise conv.
    ComfyUI expects the original grouped order and original normalization values.
    """
    state = qwen35_map(state, layers)
    for index in range(layers):
        prefix = f"model.language_model.layers.{index}.linear_attn."
        if prefix + "A_log" not in state:
            continue
        value_heads = state[prefix + "A_log"].shape[0]
        head_dim = state[prefix + "norm.weight"].shape[0]
        if not key_heads or value_heads % key_heads:
            raise ValueError("Invalid Qwen3.5 GGUF linear attention head counts.")
        if value_heads != key_heads:
            # Invert grouped -> tiled: [repeat, key, channel] -> [key, repeat, channel].
            heads = torch.arange(value_heads).reshape(-1, key_heads).T.reshape(-1)
            values = (heads[:, None] * head_dim + torch.arange(head_dim)).reshape(-1)
            qk_rows = key_heads * head_dim * 2
            qkv = torch.cat((torch.arange(qk_rows), qk_rows + values))
            for name, order in (("in_proj_qkv.weight", qkv), ("in_proj_z.weight", values),
                                ("conv1d.weight", qkv), ("in_proj_a.weight", heads),
                                ("in_proj_b.weight", heads), ("A_log", heads), ("dt_bias", heads)):
                state[prefix + name] = _rows(state[prefix + name], order)
            output = state[prefix + "out_proj.weight"]
            if is_quantized(output):
                # A 128-channel head may split a 256-value K-quant block. Apply
                # the column permutation after dequantization, before LoRA patches.
                output.column_permutation = values
            else:
                state[prefix + "out_proj.weight"] = _dense(output).index_select(1, values)
        decay = _dense(state[prefix + "A_log"])
        if not torch.isfinite(decay).all() or not (decay < 0).all():
            raise ValueError("Invalid Qwen3.5 GGUF decay weights; expected finite negative values.")
        state[prefix + "A_log"] = (-decay).log()
        state[prefix + "conv1d.weight"] = _dense(state[prefix + "conv1d.weight"]).unsqueeze(1)
        # Fused decode uses comfy.ops.CastBiasWeightContext, bypassing GGMLOps.
        for name in ("in_proj_a.weight", "in_proj_b.weight", "dt_bias", "norm.weight"):
            state[prefix + name] = _dense(state[prefix + name])

    for key in list(state):
        if (key.endswith(("norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"))
                and not key.endswith("linear_attn.norm.weight")):
            state[key] = _dense(state[key]) - 1.0
    if "lm_head.weight" in state:
        # BaseGenerate.logits also bypasses custom Linear.forward.
        state["lm_head.weight"] = _dense(state["lm_head.weight"], torch.float16)
    return state
