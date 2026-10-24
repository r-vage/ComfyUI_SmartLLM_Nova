# Native text encoder GGUF adapter

Vendored from City96 ComfyUI-GGUF via Smart Model Loader Nova; see NOTICE and
LICENSE for attribution and the copied source revision. It registers no nodes
or model folders and does not import another custom-node pack.

SmartLLM owns model discovery, supported generation layouts, companion matching,
path checks and cache identity in `core/sml/gguf_text_encoders.py`. The adapter
loads only the exact encoder and optional projector paths selected there.
`ops.py` and `dequant.py` retain the upstream quantized operations. `patcher.py`
contains only the upstream patcher, with isolated clone state. `loader.py` keeps
text mappings/tokenizer reconstruction and adds explicit Qwen/Gemma projectors;
it excludes diffusion, T5 and Llama node loading.

Qwen3-VL detector markers without a projector are not vision weights. Capability
checks inspect the original GGUF headers before those markers are injected.
