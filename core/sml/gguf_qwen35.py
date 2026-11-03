"""Header-only qualification of dense Qwen3.5 layouts supported by ComfyUI."""

import math

# hidden -> (layers, FFN, attention heads, KV heads, DeltaNet V heads,
#            vision width, vision layers, vision FFN)
_CONFIGS = {
    1024: (24, 3584, 8, 2, 16, 768, 12, 3072),
    2048: (24, 6144, 8, 2, 16, 1024, 24, 4096),
    2560: (32, 9216, 16, 4, 32, 1024, 24, 4096),
    4096: (32, 12288, 16, 4, 32, 1152, 27, 4304),
    5120: (64, 17408, 24, 4, 48, 1152, 27, 4304),
}


def compatible_language(header, hidden):
    meta, shapes = header.metadata, header.shapes
    layers, ffn, heads, kv_heads, value_heads, *_ = _CONFIGS[hidden]
    mtp = meta.get("qwen35.nextn_predict_layers", 0)
    if type(mtp) is not int or mtp not in (0, 1):
        return False
    required_meta = {
        "block_count": layers + mtp, "embedding_length": hidden,
        "feed_forward_length": ffn, "attention.head_count": heads,
        "attention.head_count_kv": kv_heads, "attention.key_length": 256,
        "attention.value_length": 256, "ssm.conv_kernel": 4,
        "ssm.state_size": 128, "ssm.group_count": 16,
        "ssm.time_step_rank": value_heads, "ssm.inner_size": value_heads * 128,
        "full_attention_interval": 4, "rope.dimension_count": 64,
        "rope.freq_base": 10000000.0, "attention.layer_norm_rms_epsilon": 1e-6,
    }
    for key, value in required_meta.items():
        actual = meta.get("qwen35." + key)
        if not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=1e-6):
            return False
    if meta.get("qwen35.rope.dimension_sections", (11, 11, 10, 0)) != (11, 11, 10, 0):
        return False
    recurrent = tuple(i < layers and (i + 1) % 4 != 0 for i in range(layers + mtp))
    if meta.get("qwen35.attention.recurrent_layers", recurrent) != recurrent:
        return False
    # These transformed variants need their own runtime; do not guess from width.
    if any("hadamard" in key for key in meta) or meta.get("general.type", "model") != "model":
        return False
    required = {"token_embd.weight": (248320, hidden), "output_norm.weight": (hidden,)}
    if hidden >= 4096 or "output.weight" in shapes:
        required["output.weight"] = (248320, hidden)
    for index in range(layers + mtp):
        block = {
            "attn_norm.weight": (hidden,), "post_attention_norm.weight": (hidden,),
            "ffn_up.weight": (ffn, hidden), "ffn_gate.weight": (ffn, hidden),
            "ffn_down.weight": (hidden, ffn),
        }
        if index < layers and (index + 1) % 4:
            value_dim, conv_dim = value_heads * 128, (32 + value_heads) * 128
            block.update({
                "attn_qkv.weight": (conv_dim, hidden), "attn_gate.weight": (value_dim, hidden),
                "ssm_alpha.weight": (value_heads, hidden), "ssm_beta.weight": (value_heads, hidden),
                "ssm_a": (value_heads,), "ssm_dt.bias": (value_heads,),
                "ssm_conv1d.weight": (conv_dim, 4), "ssm_norm.weight": (128,),
                "ssm_out.weight": (hidden, value_dim),
            })
        else:
            block.update({
                "attn_q.weight": (heads * 512, hidden), "attn_k.weight": (kv_heads * 256, hidden),
                "attn_v.weight": (kv_heads * 256, hidden), "attn_output.weight": (hidden, heads * 256),
                "attn_q_norm.weight": (256,), "attn_k_norm.weight": (256,),
            })
        required.update({f"blk.{index}." + key: value for key, value in block.items()})
    if mtp:
        required.update({f"blk.{layers}.nextn." + key: value for key, value in {
            "eh_proj.weight": (hidden, hidden * 2), "enorm.weight": (hidden,),
            "hnorm.weight": (hidden,), "shared_head_norm.weight": (hidden,),
        }.items()})
    return shapes == required


def compatible_projector(header, hidden):
    meta, shapes = header.metadata, header.shapes
    *_, width, layers, ffn = _CONFIGS[hidden]
    required_meta = {
        "clip.projector_type": "qwen3vl_merger", "clip.vision.projection_dim": hidden,
        "clip.vision.embedding_length": width, "clip.vision.block_count": layers,
        "clip.vision.feed_forward_length": ffn, "clip.vision.patch_size": 16,
        "clip.vision.attention.head_count": 12 if hidden == 1024 else 16,
        "clip.vision.spatial_merge_size": 2,
    }
    if any(meta.get(key) != value for key, value in required_meta.items()):
        return False
    required = {
        "v.position_embd.weight": (2304, width), "v.patch_embd.bias": (width,),
        "v.post_ln.weight": (width,), "v.post_ln.bias": (width,),
        "mm.0.weight": (width * 4, width * 4), "mm.0.bias": (width * 4,),
        "mm.2.weight": (hidden, width * 4), "mm.2.bias": (hidden,),
    }
    if "v.patch_embd.weight.1" in shapes:
        required.update({"v.patch_embd.weight": (width, 3, 16, 16),
                         "v.patch_embd.weight.1": (width, 3, 16, 16)})
    else:
        required["v.patch_embd.weight"] = (width, 3, 2, 16, 16)
    for index in range(layers):
        block = {
            "attn_qkv.weight": (width * 3, width), "attn_qkv.bias": (width * 3,),
            "attn_out.weight": (width, width), "attn_out.bias": (width,),
            "ln1.weight": (width,), "ln1.bias": (width,),
            "ln2.weight": (width,), "ln2.bias": (width,),
            "ffn_up.weight": (ffn, width), "ffn_up.bias": (ffn,),
            "ffn_down.weight": (width, ffn), "ffn_down.bias": (width,),
        }
        required.update({f"v.blk.{index}." + key: value for key, value in block.items()})
    # A Qwen3-VL tower with deepstack weights is not a Qwen3.5 companion.
    return shapes == required
