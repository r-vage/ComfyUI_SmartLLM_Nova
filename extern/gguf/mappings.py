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

# Qwen3.5 has both gated full attention and DeltaNet blocks. Match complete
# components: the generic attn_q replacement would corrupt attn_qkv.
QWEN35_SD_MAP = {
    "blk.": "model.language_model.layers.",
    "attn_norm.": "input_layernorm.",
    "post_attention_norm.": "post_attention_layernorm.",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_qkv.": "linear_attn.in_proj_qkv.",
    "attn_gate.": "linear_attn.in_proj_z.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "ssm_alpha.": "linear_attn.in_proj_a.",
    "ssm_beta.": "linear_attn.in_proj_b.",
    "ssm_out.": "linear_attn.out_proj.",
    "ssm_norm.": "linear_attn.norm.",
    "ssm_conv1d.": "linear_attn.conv1d.",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
    "ffn_gate.": "mlp.gate_proj.",
    "token_embd.": "model.language_model.embed_tokens.",
    "output_norm.": "model.language_model.norm.",
    "output.weight": "lm_head.weight",
}


def qwen35_map(raw_sd, layers):
    """Map the backbone and optional single MTP head, also usable on shapes."""
    mapped = sd_map_replace(raw_sd, QWEN35_SD_MAP)
    prefix = f"model.language_model.layers.{layers}."
    mtp = {"nextn.eh_proj.": "fc.", "nextn.enorm.": "pre_fc_norm_embedding.",
           "nextn.hnorm.": "pre_fc_norm_hidden.", "nextn.shared_head_norm.": "norm."}
    result = {}
    for key, value in mapped.items():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            key = "mtp.layers.0." + suffix
            for source, target in mtp.items():
                if suffix.startswith(source):
                    key = "mtp." + target + suffix[len(source):]
                    break
        result[key] = value
    return result

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

