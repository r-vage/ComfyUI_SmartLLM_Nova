# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Adapted for SmartLLM Nova native text generation.

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}


GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update(
    {
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm": "post_feedforward_layernorm",
        "post_attention_norm": "post_attention_layernorm",
    },
)

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

CLIP_VISION_QWEN3_MAP = {
    "v.blk": "model.visual.blocks",
    ".fc": ".linear_fc",
    "ck.8.": "st.0.",
    "ck.16.": "st.1.",
    "ck.24.": "st.2.",
    "ck.5.": "st.0.",
    "ck.11.": "st.1.",
    "ck.17.": "st.2.",
    "attn_out": "attn.proj",
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "mm.0": "model.visual.merger.linear_fc1",
    "mm.2": "model.visual.merger.linear_fc2",
    "v.post_ln": "model.visual.merger.norm",
    "v.patch_embd": "model.visual.patch_embed.proj",
    "v.position_embd.weight": "visual.pos_embed.weight",
    "v.deepstack.": "model.visual.deepstack_merger_list.",
    "v.deepstast.": "model.visual.deepstack_merger_list.",
}


def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd

