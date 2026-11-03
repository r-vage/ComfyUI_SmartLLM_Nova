# Native text encoder GGUF adapter

Vendored from City96 ComfyUI-GGUF via Smart Model Loader Nova; see NOTICE and
LICENSE for attribution and the copied source revision. It registers no nodes
or model folders and does not import another custom-node pack.

SmartLLM owns model discovery, supported generation layouts, companion matching,
path checks and cache identity in `core/sml/gguf_text_encoders.py`. The adapter
loads only the exact encoder and optional projector paths selected there.
`ops.py` and `dequant.py` retain the upstream quantized operations. `patcher.py`
adapts the upstream patcher with isolated clone state and CPU-only mmap
detachment. Qwen3.5 generation reserves prompt/image, KV-cache and cast workspace
using native partial loading and unloading, including when a cached model needs
more weights offloaded for a larger request. `loader.py` keeps
text mappings/tokenizer reconstruction and adds explicit Qwen/Gemma projectors;
it excludes diffusion, T5 and Llama node loading.

Qwen3-VL detector markers without a projector are not vision weights. Capability
checks inspect the original GGUF headers before those markers are injected.

Qwen3.5 dense GGUFs use their own hybrid-attention and optional MTP mappings.
`qwen35.py` reverses llama.cpp's norm, decay, convolution and V-head transforms.
Whole encoded rows are reordered without changing quantization; output-projection
columns are reordered after dequantization because a K-quant block can span two
heads. The permutation survives device moves and runs before weight patches.
Native fused decoding and logits require dense gate/convolution parameters and
output heads. Matched Qwen3.5 projectors use the Qwen3-VL-style tower without
deepstack, with a dense position embedding and quantization-aware Conv3d.
