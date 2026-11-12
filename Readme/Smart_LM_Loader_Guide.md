# Smart LM Loader Guide

A comprehensive guide to the **Smart LM Loader** node for ComfyUI SmartLLM Nova. See the [visual tour](Visual_Tour.md) and [Nova panels](Nova_Panels.md) for the layout and connected overrides.

---

## Table of Contents

- [Smart LM Loader Guide](#smart-lm-loader-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [What Can This Node Do?](#what-can-this-node-do)
  - [Key Features](#key-features)
  - [Node Overview](#node-overview)
    - [Inputs](#inputs)
      - [Panel settings](#panel-settings)
      - [Mode bar options](#mode-bar-options)
      - [Image handling section](#image-handling-section)
      - [Sampling and hardware sections](#sampling-and-hardware-sections)
      - [WD14 tagging section](#wd14-tagging-section)
      - [Connection Slots](#connection-slots)
    - [Outputs](#outputs)
  - [Model Families](#model-families)
    - [Qwen](#qwen)
    - [Mistral](#mistral)
    - [Florence](#florence)
    - [LLaVA](#llava)
    - [VLM (Generic)](#vlm-generic)
    - [LLM (Text-Only)](#llm-text-only)
    - [WD14 Tagger](#wd14-tagger)
  - [Backends](#backends)
    - [ComfyUI text encoders](#comfyui-text-encoders)
    - [Transformers](#transformers)
    - [GGUF (llama-cpp-python)](#gguf-llama-cpp-python)
    - [vLLM (Docker)](#vllm-docker)
    - [SGLang (Docker)](#sglang-docker)
    - [Ollama (Docker)](#ollama-docker)
    - [llama.cpp (Docker)](#llamacpp-docker)
    - [WD14 (ONNX)](#wd14-onnx)
  - [Compatibility Matrix](#compatibility-matrix)
  - [Tasks Reference](#tasks-reference)
    - [Custom Tasks](#custom-tasks)
    - [Vision Tasks (all VLM families)](#vision-tasks-all-vlm-families)
    - [Vision Tasks (Florence-only)](#vision-tasks-florence-only)
    - [Text Tasks (all families)](#text-tasks-all-families)
  - [Multi-Task Mode](#multi-task-mode)
    - [How It Works](#how-it-works)
    - [Example: Image → Tags → Natural Language → Expanded Prompt](#example-image--tags--natural-language--expanded-prompt)
    - [Notes](#notes)
  - [Quantization](#quantization)
    - [Transformers](#transformers-1)
    - [GGUF](#gguf)
    - [Docker (vLLM / SGLang / Ollama / llama.cpp)](#docker-vllm--sglang--ollama--llamacpp)
  - [Docker Configuration](#docker-configuration)
  - [Quick Start Examples](#quick-start-examples)
    - [Image Description (Registry Model)](#image-description-registry-model)
    - [WD14 Tagging](#wd14-tagging)
    - [Text Expansion (No Image)](#text-expansion-no-image)
    - [Audio-Video Prompt (Wan, LTX, or MiniMax H3)](#audio-video-prompt-wan-ltx-or-minimax-h3)
    - [Song Lyrics](#song-lyrics)
    - [Direct Chat with a Prompt Pipe](#direct-chat-with-a-prompt-pipe)
    - [Docker Backend (Ollama)](#docker-backend-ollama)
  - [Troubleshooting](#troubleshooting)
    - [Model Not in Dropdown](#model-not-in-dropdown)
    - [Florence-2 Not Loading](#florence-2-not-loading)
    - [Mistral3 Not Loading](#mistral3-not-loading)
    - [Docker Container Won't Start](#docker-container-wont-start)
    - [Out of Memory (OOM)](#out-of-memory-oom)
    - [GGUF Vision Not Working](#gguf-vision-not-working)
    - [Debug Logging](#debug-logging)
  - [Configuration Files](#configuration-files)
  - [Image sequences and scene motion](#image-sequences-and-scene-motion)
    - [Scene poses and ongoing motion](#scene-poses-and-ongoing-motion)
    - [Concise scene observations and black-frame filtering](#concise-scene-observations-and-black-frame-filtering)

---

## Overview

The **Smart LM Loader** is a unified node for loading and running vision-language models, text-only LLMs, and WD14 taggers in ComfyUI. Pick a registered model or a compatible local ComfyUI text encoder from the unified dropdown, choose a task, and generate.

### What Can This Node Do?

- **Image Analysis** — describe, analyze, and extract information from images
- **Text Generation** — chat, expand prompts, translate, summarize
- **Tag Generation** — WD14 tagger with booru-style output
- **Video Analysis** — summarize frame sequences or extract scene motion/full descriptions with compatible vision models
- **Prompt Pipelines** — chain up to 4 tasks sequentially via multi-task mode
- **OCR** — extract text from images

> **Note:** For object detection with bounding boxes, masks, and SEGS output, use the [Smart Detection](Smart_Detection_Guide.md) node instead.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Registry-Based Model Selection** | Unified dropdown with 50+ models grouped by backend — no templates, no manual paths |
| **Auto-Download** | Full repositories download from Hugging Face; targeted GGUF/mmproj files can also use ModelScope, with immutable revision and integrity verification |
| **Backends** | ComfyUI text encoders, Transformers, GGUF, vLLM, SGLang, Ollama, llama.cpp and WD14; YOLO uses Smart Detection |
| **Image Handling** | Frames/Grid mode, Together, bounded native VIDEO sampling, sizing and frame stride; text-only requests need no images |
| **Scene Tasks** | Motion for a separate image reference or full T2V descriptions, with generation prose and observations in one response |
| **Multi-Task Chaining** | Select compatible tasks in order, with output→input flow |
| **Task Instructions** | One editable instruction set per task, with no example conversations or profile switches |
| **Editable System Prompts** | Preview instructions in Prompt; edit the task file or Direct Chat’s local system text |
| **Mode Bar** | Cleanup, Keep loaded, Sampling, Thinking (supported backends), Local seed |
| **Persist-on-Execute** | Advanced parameters (temperature, top_p, etc.) saved to defaults on each run |
| **Docker Lifecycle** | Auto-start/stop containers, stale image detection |
| **Image Passthrough** | Input images flow through to the output for downstream nodes |
| **WD14 Tagger** | ONNX-based image tagging with configurable thresholds |

Hugging Face repositories download one file at a time. The console identifies
the active file and reports real byte, percentage, and elapsed progress. If
Xet produces no visible update for 30 seconds, SmartLLM prints a truthful
elapsed-time heartbeat. It reports that it is waiting when no bytes are known,
or the latest real byte count after a sparse early callback; it never estimates
progress.

---

## Node Overview

### Inputs

#### Panel settings

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **model** | Dropdown | — | Model from registry. Suffix indicates backend (no suffix=Transformers, -GGUF, -vLLM, -SGLang, -Ollama, -llama.cpp) |
| **quantization** | Dropdown | Q4_K_M | Native GGUF and llama.cpp Docker — quantization variant |
| **Tasks** | Separate resizable panel | Detailed Description | Select one task or an ordered compatible chain on the left; edit reusable instructions and templates on the right |
| **user_prompt** | String | — | Optional additional information, question, or constraints. The selected task already loads its system prompt, so this is normally empty for standard vision tasks |
| **context_size** | Integer | 8192 | Model context window (512–131072). Persisted on execute |
| **max_tokens** | Integer | 4096 | Maximum tokens to generate (1–32768) |
| **attention_mode** | Dropdown | auto | Transformers only — auto, flash_attention_2, sdpa, eager |
| **seed** | 64-bit integer | Random | Seed section with fixed/random/increment/decrement policies and −/+ controls |

#### Mode bar options

| Option | Default | Description |
|------|---------|-------------|
| **Cleanup** | ON | Pre-load VRAM cleanup — free memory before loading |
| **Keep Loaded** | OFF | Cache model in VRAM between runs |
| **Sampling** | ON | Enable custom sampling values |
| **Thinking** | OFF | Requests reasoning through ComfyUI, Ollama, Transformers, vLLM, SGLang or llama.cpp server controls. The model/template must support switching. Saved per node and applied through task chains; older workflows default to off. |
| **Local seed** | ON | Use the Seed section; disable to use a connected seed_input |

Each task has one instruction set. Training and NSFW switches are removed; their
values in older workflows are ignored. The initial library uses the current bundled
tasks. Old prompt, training and profile files are ignored; they do not create drafts
or supply examples. Edit the individual tasks to customize their instructions.
See [Task library](Task_Library.md) for editing and upgrade details.

The **Sampling** mode button enables custom sampling values. Model contains
Registry, Delete and Refresh actions; full verification is available in Registry Manager.

#### Model lifetime and explicit unloading

The loader handles prompt/image lists and chained tasks within one execution.
It loads once and reuses that model for the whole call. With **Keep loaded off**,
cleanup runs after the entire call, including its task chain, or after an error.

An external loop invokes the loader again for each iteration. **Keep loaded on**
allows those calls to reuse the model; the loader does not know which iteration
is last. Add **Smart LM Unload [SmartLLM Nova]** after the completed loop:

`Loop End → Smart LM Unload → downstream consumer`

The unload node has one `value` input and one matching output. It preserves the
data type, list shape and value, with no extra settings. Active, it clears
SmartLLM's retained native model caches and stops its managed inference
containers, including Ollama. Bypassed, it only passes data through. It runs
again on each queue even when the upstream result is cached. Place it after all
tasks that need the retained models; it refuses cleanup while another SmartLLM
execution is using them and reports a failed container stop rather than passing
the result onward. Unrelated ComfyUI models and containers are unaffected.

Keep the output connected to the downstream work that should wait for cleanup.
For scene analysis, connect **Loop End → Smart LM Unload → Save Scene Analysis**;
putting it directly after the loader inside the loop would unload every iteration.

#### Image handling section

Open **Image handling** below Model. It has its own collapsed summary and pin
state, so its controls remain accessible with Model collapsed or Direct Chat
selected.

| Control | Default | Description |
| --- | --- | --- |
| **Together** (`image_handling`) | Off | On analyzes the ordered selection in one request; Off processes one image per request. Older saved configurations retain Legacy until changed. |
| **Longest side** (`max_image_side`) | 0 | Zero uses automatic model limits; a positive value reduces images within those limits without enlarging the source. |
| **Every Nth frame** (`image_stride`) | 1 | Together only; samples the supplied selection and always retains its first and last image. |
| **Max frames** (`frame_count`) | 8 | Applicable Legacy configurations only. Explicit Together uses the supplied selection without this last-N cap. |

The mode bar's **Frames / Grid** button defaults to **Frames**, preserving saved
workflows. **Grid** builds numbered contact sheets inside the loader and implies
Together without overwriting its saved value. Switching back restores that choice.
In Grid mode, Image handling shows:

| Control | Default | Description |
| --- | --- | --- |
| **Frames per grid** | 8 | At most 16; the loader reduces capacity when necessary to keep source tiles readable. |
| **Grid longest side** | 2048 | Contact-sheet ceiling, separate from individual-frame sizing and subject to model limits. |
| **Tile longest side** | 512 | Source tile ceiling; aspect ratio is preserved and source images are never enlarged. |
| **Every Nth frame** | 1 | Samples IMAGE/disk frames, retaining both endpoints. Native VIDEO uses Sample FPS instead. |

Tiles are read left to right, top to bottom. Labels sit outside the source image.
The selected task instructions still apply. Scene JSON records the
original frame positions plus the actual grid layout; H3 validates the number of
source frames, not the number of contact sheets. Florence and WD14 require
individual images and cannot use Grid or native VIDEO analysis.

Supply one scene as an IMAGE sequence or disk source. Long selections use bounded
windows with up to two overlapping frames. Each window is analyzed independently;
a final text-only synthesis combines the observations into one result, removing
overlap without treating each window as a new opening pose. Large observation
sets use staged synthesis within the configured context allowance. Max tokens
applies to each generation, including synthesis. Single-window requests need no
extra call. The loader reuses the loaded model throughout.

Grid can only see the frames supplied in the current call. For a complete scene
from Eclipse, sample across that scene and set its upstream window capacity high
enough to deliver the selection together. Existing external windows remain
separate loader calls; Grid does not read other jobs or change detected cuts.

The **image / video** socket also accepts native ComfyUI **VIDEO**, without a
conversion node. Its **Video input** section provides **Sample FPS** (default 2)
and **Cut threshold** (default 27). Video is decoded in small chunks, honoring
native trims/crops, and retains both shot endpoints. Frames mode uses up to 16
images per window; with Longest side 0, native VIDEO frames are bounded to 1024px
before model processing. Grid uses its configured tile size and capacity. Native concatenation
boundaries are preserved. Cut detection is a heuristic: lower thresholds detect
more changes and may split camera movement. Set 0 for a known continuous shot.
Video timestamps are frame-rate estimates because the native component API does
not provide presentation timestamps. Use Eclipse's measured timing for exact
cut planning. Audio is not analyzed. VIDEO and disk inputs return no IMAGE
passthrough, avoiding full-video retention in graph caches.

Bound large sequences upstream with sampling or overlapping windows. For 16
images together, supply up to 16 images and set Together On with Every Nth = 1;
the model's own image/context limits still apply. See [image sequences and scene
motion](#image-sequences-and-scene-motion) for the full behavior.

#### Sampling and hardware sections

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **device** | Dropdown | default | Default follows ComfyUI's device policy; CPU forces CPU processing. |
| **temperature** | Float | 0.7 | Sampling temperature (0.1–2.0) |
| **top_p** | Float | 0.9 | Nucleus sampling (0.1–1.0) |
| **top_k** | Integer | 50 | Top-k sampling, 0=disabled |
| **num_beams** | Integer | 1 | Beam search count. 1=greedy |
| **do_sample** | Boolean | True | Enable sampling vs greedy decoding |
| **repetition_penalty** | Float | 1.0 | Repeat penalty (1.0–2.0) |
| **use_torch_compile** | Boolean | False | torch.compile for faster inference |

Panel values are saved with the workflow. Shared generation defaults are updated on execution; opening a panel does not save or acquire a model.

#### WD14 tagging section

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **threshold** | Float | 0.35 | General tag confidence threshold |
| **char_threshold** | Float | 0.85 | Character tag confidence threshold |
| **replace_underscore** | Boolean | True | Replace underscores with spaces |

#### Connection Slots

| Input | Type | Description |
|-------|------|-------------|
| **image / video** (`images`) | IMAGE, VIDEO | IMAGE tensors or native ComfyUI VIDEO; video requires a supported VLM. Existing IMAGE links and slot indices are preserved. |
| **frame_source** | SMARTLLM_FRAME_SOURCE | Optional lazy frame input for bounded analysis loops; use instead of images, never both |
| **seed_input** | INT | Optional unsigned 64-bit seed, used when Local seed is disabled |
| **pipe_opt** | Settings | Optional Smart LM Settings or Smart LM Prompt Pipe output; carries settings and user/system prompts. See [subgraph settings](Nova_Panels.md#native-settings-and-subgraphs). |

Connect external `user_prompt` and `system_prompt` strings to **Smart LM Prompt
Pipe**, then connect its `settings` output to `pipe_opt`. It works without native
settings widgets, or passes through **Smart LM Settings** while overriding its
prompts. User prompt lists are supported with one shared system instruction.
Unconnected prompt inputs preserve incoming values; empty text is an explicit override.

A nonblank `system_prompt` in the pipe replaces
that task's built-in instructions, so the provided text must be self-contained:
include the model's role, the complete objective, any rules or constraints, and
the required output format. In this mode, `user_prompt` is the corresponding user
message or source material. Leave `system_prompt` empty or absent when you want to
use a predefined task such as Detailed Description, Wan/LTX/MiniMax H3 prompting,
or Song Lyrics.

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| **image** | IMAGE | Ordinary input image passthrough, or a 64×64 placeholder without images; native VIDEO and lazy frame_source return an empty list |
| **text** | STRING | Generated text or tags |

---

## Model Families

The model family is **auto-detected from the registry entry**. Each model in the registry has a `family` field that determines which generation path is used.

### Qwen

**Vision-language models from Alibaba** — Qwen2.5-VL and Qwen3-VL series.

| Property | Value |
|----------|-------|
| **Vision** | ✅ Yes |
| **Video** | ✅ Yes (multi-frame) |
| **Backends** | Transformers, GGUF, vLLM, SGLang, Ollama |
| **Best For** | Versatile — descriptions, analysis, detection, video |
| **VRAM (3B)** | ~7 GB (FP16), ~4 GB (4-bit) |

**Registry Models:** Qwen2.5-VL-3B/7B, Qwen3-VL-8B

### Mistral

**Vision-language models from Mistral AI** — Ministral-3 (3B, 8B) and Mistral Small 3.

| Property | Value |
|----------|-------|
| **Vision** | ✅ Yes |
| **Video** | ❌ No |
| **Backends** | Transformers, vLLM, SGLang, Ollama |
| **Best For** | High-quality image descriptions |
| **Requires** | transformers v5.0+ |

**Registry Models:** Ministral-3-3B/8B-Instruct

### Florence

**Microsoft Florence-2** specialized vision models with 15+ task types.

| Property | Value |
|----------|-------|
| **Vision** | ✅ Yes |
| **Video** | ❌ No |
| **Backends** | Transformers only |
| **Best For** | OCR, prompt generation (PromptGen models) |
| **VRAM** | ~4 GB |

**Registry Models:** Florence-2-base/large, base-ft/large-ft, PromptGen v1/v2/v1.5/v2.0

### LLaVA

**Large Language and Vision Assistant** — LLaVA 1.5/1.6 and Llama 3.2 Vision.

| Property | Value |
|----------|-------|
| **Vision** | ✅ Yes |
| **Video** | ❌ No |
| **Backends** | Transformers, GGUF, Ollama |
| **Best For** | General vision tasks |

### VLM (Generic)

**Catch-all** for vision-language models not in a specific family (Gemma 3, Phi-3/4 Vision, MiniCPM-V, Moondream, etc.). Auto-detected from model config — uses the LLaVA execution path for Docker backends.

### LLM (Text-Only)

**Text-only language models** without vision.

| Property | Value |
|----------|-------|
| **Vision** | ❌ No |
| **Backends** | GGUF, Ollama |
| **Best For** | Text expansion, translation, tag conversion |

### WD14 Tagger

**ONNX-based image tagger** using SmilingWolf's WaifuDiffusion models. Outputs booru-style comma-separated tags — completely separate from the VLM/LLM pipeline.

| Property | Value |
|----------|-------|
| **Runtime** | ONNX (CUDA or CPU) |
| **Best For** | Fast tag generation for Stable Diffusion prompts |
| **Speed** | ~1–2s per image |

**Registry Models:** wd-eva02-large-tagger-v3, wd-vit-large-tagger-v3, wd-swinv2-tagger-v3, wd-convnext-tagger-v3, wd-vit-tagger-v3

**Output format:** `1girl, solo, long hair, looking at viewer, smile, blue eyes, ...`

---

## Backends

### ComfyUI text encoders

This additional backend reuses local encoder files through the native loading and `tokenize → generate → decode` APIs used by ComfyUI's **Load CLIP** and **Generate Text** nodes. Existing Transformers, GGUF, Docker and tagging backends remain available.

1. Place a compatible `.safetensors`, `.sft` or `.gguf` file in ComfyUI's `models/text_encoders` folder. Configured extra text encoder folders and subfolders are included too.
2. Open the loader's **Model** selector and choose a `ComfyUI: filename` entry under **ComfyUI vision encoders** or **ComfyUI text-only encoders**. Opening the selector refreshes the available files.
3. Choose a task, enter a prompt and run the node. Connect images when the selected file includes supported vision weights.

Discovery reads checkpoint headers, not model weights or filename guesses. ComfyUI identifies the architecture from tensor names and shapes; Nova checks for supported vision-weight prefixes (`vision_model.`, `visual.`, `model.visual.`). Vision entries can use image inputs as well as text. Text-only entries expose text tasks. A `VL` filename is not sufficient: a text-only conversion of a vision model belongs in the text-only group when it is a supported native generator. Header inspection is cached until the file changes; it does not load the model onto the GPU.

Supported native generation families include Gemma 2/3/4, Qwen3, Qwen3.5, Qwen3-VL and Qwen2.5-VL 7B where supported by the installed ComfyUI. Ordinary CLIP/T5 encoders, truncated diffusion encoders, music vocabularies, missing Gemma tokenizers and Qwen checkpoints missing a required output head are excluded. A text encoder used for diffusion conditioning is not necessarily a complete language generator. Files are reused in place; no model download, conversion or registry registration is needed.

**GGUF files in `text_encoders`.** SmartLLM bundles its own ComfyUI-GGUF adapter; Smart Model Loader and a separate GGUF node pack are not required. Supported layouts are Qwen3 0.6B/4B/8B, Qwen3.5 dense 0.8B/2B/4B/9B/27B, Qwen2.5-VL 7B, Qwen3-VL 4B/8B and Gemma3 4B/12B, subject to the installed ComfyUI generation API. Qwen3.5 supports its hybrid attention and optional single MTP prediction layer. Compatible `qwen35` checkpoints such as Qwen3.8-27B are identified by their metadata and tensors. T5/UMT5, projector-only files, MoE or other unsupported layouts and incomplete models stay out of the list.

For GGUF vision, copy **that model's own matching mmproj** into the **same subfolder** as its encoder. The base filenames must match after removing the quantization suffix and `mmproj` marker. For example:

```text
models/text_encoders/
  Qwen2.5-VL-7B-Abliterated-Caption-it.Q4_K_S.gguf
  Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-Q8_0.gguf
```

Nova also checks the projector architecture, dimensions and tensor inventory. A projector is not interchangeable merely because it has the same size or contains `VL` in its name. Generic names such as `mmproj-model-f16.gguf` need an unambiguous matching base filename; only rename a projector after verifying it belongs to that model. If multiple projectors match, keep one matching companion beside the encoder. Missing, mismatched or ambiguous projectors leave the encoder in the text-only group. The Model section shows the selected projector or explains why vision is unavailable. The `LLM` folder is not searched for companions.

Opening the Model selector refreshes discovery. Copying, replacing or deleting a companion also invalidates the native model and workflow cache on the next run. Large transformer projections remain quantized; token embeddings, Qwen2.5/Qwen3.5 output heads and small Qwen3.5 decode parameters are expanded for native generation, so runtime memory can exceed the GGUF file size. Shared encoder/projector files remain read-only to SmartLLM.

**Device** offers **Default** and **CPU**. Default uses ComfyUI's native text-encoder placement and offloading policy; CPU forces CPU processing. Older Auto/CUDA/MPS selections load as Default. Checkpoint precision, attention and memory offloading follow ComfyUI. **Keep loaded** reuses the encoder while the file and resolved loading settings match. Turning it off releases Nova's encoder after execution, including errors. ComfyUI can still offload cached weights to manage memory.

Native Qwen3.5 GGUF generation reserves working memory for expanded image tokens,
prompt processing, the output-length KV cache and temporary weight conversions.
ComfyUI keeps the remaining weights in system RAM and transfers them as needed.
A longer request can offload more of a cached model. This makes **Default** useful
for checkpoints larger than available VRAM, provided system RAM is sufficient;
CPU transfers can make generation slower. Smaller input images and lower
**max_tokens** reduce the working-memory requirement.

Local Transformers, YOLO and WD14 loading also use ComfyUI's selected device for memory reservations. Unsupported WD14/Florence accelerator paths use CPU. Managed GPU servers and native llama.cpp vision projectors cannot honor CPU selection and report that restriction; local Transformers and ComfyUI encoders support CPU vision.

Prompts, task chaining, image batches, connected instructions and exact seeds retain Nova's existing workflow. The native tokenizer receives the task's system instructions and user guidance without example conversations.

The **Thinking** button in the Mode bar defaults to **off**. Native ComfyUI
requests the tokenizer's non-thinking template. Turn it on to allow reasoning
where the installed ComfyUI tokenizer supports it. Each node saves its choice in
the workflow and applies it to every chained task; it does not change global
defaults. Older workflows without the setting use off. The caption output
excludes reasoning; the raw output retains it. **Max tokens** covers both
reasoning and the final answer. Nova reports a missing final answer instead of
returning reasoning as a caption.

The same button also controls Ollama (`think`), Transformers (`enable_thinking`
in the chat template), native vLLM (`LLM.chat` template arguments), and vLLM,
SGLang and llama.cpp servers (`chat_template_kwargs.enable_thinking`). Text and
vision requests use the same choice. These are controls for compatible templates,
including switchable Qwen models; they cannot disable a model that always thinks
or add reasoning to a model that lacks it. Templates using a different control,
such as DeepSeek's `thinking`, are not covered by the `enable_thinking` switch.
Server answer fields are kept separate from their reasoning fields. Transformers
also removes reasoning when its opening tag was already supplied by the template.
Ollama and the other non-native backends share **Max tokens** between reasoning
and the answer without the native Qwen caption budget described below. Increase
the limit when reasoning leaves too little room for complete JSON.

The installed local **GGUF / llama-cpp-python** generation API has no supported
per-request thinking parameter. Its button remains hidden; the model's own
template determines behavior. Florence and detector/tagger models also have no
Thinking button. Switching backends preserves the node's saved choice.

With **Thinking on**, native Qwen3/Qwen3.5 **Simple**, **Detailed**, and
**Ultra Detailed Description** use at most 128 thinking tokens or one quarter of **Max tokens**, whichever is
smaller. At that limit, generation closes the thinking block and continues with
the caption in the same pass. At 512 total tokens, up to 128 are available for
thinking and approximately 380 remain for the caption after the closing markers.
Natural earlier completion is preserved. Both phases count towards the existing
total; no second image-processing pass or automatic retry is added. These requests
use ordinary decoding if the checkpoint has speculative MTP weights so the
thinking limit is enforced. Other tasks and families use their supported native
templates without this budget; custom raw chat templates retain their own policy.
With **Thinking off**, the full output allowance is available for the answer.
This bounds reasoning length, not seconds per token;
CPU offloading can still make a large checkpoint slow.

Sampling supports temperature, top-k, top-p, min-p, repetition penalty and greedy generation. Native generation determines its own context allocation, so context-size, beam-search and Transformers-specific controls are hidden. Custom stop strings trim the decoded response; they do not stop native token generation early. Native EOS tokens still stop generation normally.

These shared files are outside Registry Manager's editing and deletion actions. Remove or replace them using your normal ComfyUI model-folder workflow. A missing saved selection remains visible as unavailable instead of silently choosing another model.

### Transformers

HuggingFace Transformers — direct Python loading.

| Property | Value |
|----------|-------|
| **Docker** | ❌ No |
| **Quantization** | BitsAndBytes (4-bit, 8-bit) |
| **Families** | All (Qwen, Mistral, Florence, LLaVA, VLM, LLM) |
| **Best For** | Simplest setup |

Repository-supplied Python model or processor code is unsupported. SmartLLM
explicitly disables remote code at Transformers Auto-class boundaries. Florence
uses SmartLLM's local vendored implementation; other models must use an
architecture supported by the installed Transformers package.

### GGUF (llama-cpp-python)

llama-cpp-python for GGUF format models.

| Property | Value |
|----------|-------|
| **Docker** | ❌ No |
| **Quantization** | Built into GGUF file (Q3–Q8, IQ3–IQ4) |
| **Families** | Qwen, LLaVA, LLM |
| **Best For** | Pre-quantized models, lower VRAM |

> GGUF vision requires an mmproj file (auto-downloaded from registry). Mistral3 architecture is not supported.

### vLLM (Docker)

High-performance inference server via Docker.

| Property | Value |
|----------|-------|
| **Docker** | ✅ Yes |
| **Quantization** | FP8, AWQ, GPTQ (auto-detected) |
| **Families** | Qwen, Mistral, LLM |
| **Best For** | Pre-quantized FP8 models, continuous batching |

SmartLLM never adds `--trust-remote-code` to vLLM launches. Models that require
repository-supplied Python code are unsupported.

### SGLang (Docker)

Alternative to vLLM with RadixAttention for KV cache reuse.

| Property | Value |
|----------|-------|
| **Docker** | ✅ Yes |
| **Quantization** | FP8, AWQ, GPTQ (pre-quantized only) |
| **Families** | Qwen, Mistral, LLM |
| **Best For** | Better throughput for repeated requests |

### Ollama (Docker)

Easy model management with auto-pull from Ollama registry.

| Property | Value |
|----------|-------|
| **Docker** | ✅ Yes |
| **Quantization** | Pre-quantized from Ollama registry |
| **Families** | Qwen, Mistral, LLaVA, LLM |
| **Best For** | Easiest setup, fast batch processing |

### llama.cpp (Docker)

Reference GGUF engine via Docker with vision support.

Models with the `-llama.cpp` suffix select this backend explicitly and reuse
the same verified GGUF files used by native `-GGUF` entries.

| Property | Value |
|----------|-------|
| **Docker** | ✅ Yes |
| **Quantization** | Built into GGUF file |
| **Families** | Qwen, Mistral, LLaVA, LLM |
| **Best For** | GGUF models with vision, GPU layer offloading |

### WD14 (ONNX)

ONNX Runtime for SmilingWolf WD14 tagger models.

| Property | Value |
|----------|-------|
| **Docker** | ❌ No |
| **Best For** | Fast booru-style tag generation |

---

## Compatibility Matrix

| Backend | Qwen | Mistral | Florence | LLaVA | LLM | WD14 |
|---------|:----:|:-------:|:--------:|:-----:|:---:|:----:|
| **Transformers** | ✅ | ✅¹ | ✅ | ✅ | ✅ | — |
| **GGUF** | ✅ | ❌² | ❌ | ✅³ | ✅ | — |
| **vLLM (Docker)** | ✅ | ✅ | ❌ | ❌ | ✅ | — |
| **SGLang (Docker)** | ✅ | ✅ | ❌ | ❌ | ✅ | — |
| **Ollama (Docker)** | ✅ | ✅ | ❌ | ✅ | ✅ | — |
| **llama.cpp (Docker)** | ✅ | ✅ | ❌ | ✅ | ✅ | — |
| **WD14 (ONNX)** | — | — | — | — | — | ✅ |

1. ¹ Mistral requires transformers v5.0+
2. ² Mistral3 architecture not supported by llama-cpp-python
3. ³ LLaVA GGUF requires mmproj file

---

## Tasks Reference

### Custom Tasks

| Task | Image | Description |
|------|:-----:|-------------|
| **Direct Chat** | Optional | Interactive conversation |
| **Question Answering** | Optional | Answer questions about image or text |
| **Custom Instruction** | Optional | Use your own prompt in user_prompt |
| **Wan 2.2** | Optional | One cinematic motion paragraph from text, an image or an ordered scene sequence; no timestamps or fixed duration |
| **Wan 2.2 CN Atomic** | Optional | Chinese Wan prompt using explicit initial state, ordered physical actions, and final state |
| **LTX 2.3 I2V** | Optional | One `Style: ...` paragraph combining text, a reference image or a scene sequence with requested motion and audio |
| **MiniMax H3 Scene** | 0–2 | One continuous duration-neutral scene; zero/one/two images select T2VA/I2VA/FL2VA, and an empty prompt enables automatic storytelling |
| **MiniMax H3 T2VA Timeline** | None | Text-only action and requested shot progression, without numeric timing |
| **MiniMax H3 I2VA Timeline** | First | Timeline beginning exactly on one first-frame image |
| **MiniMax H3 FL2VA Timeline** | First; optional last | Timeline starting from the first reference and landing on the encoder-supplied last frame; an optional second reference lets the VLM inspect both endpoints |
| **MiniMax H3 L2VA Timeline** | Last | Timeline converging on one last-frame image |

These reference counts describe Legacy H3 inputs. With **Together** or **Grid**,
the four image-capable H3 tasks also accept an ordered scene sequence from the
image / video socket or `frame_source`. Every selected sample contributes to one
description in the same three-field format. A grid's tiles represent moments in
the scene; one contact sheet does not select single-reference mode. The first and
last samples establish the observed boundaries, and intermediate samples remain
motion evidence. **T2VA Timeline** remains text-only.

Enter additional guidance in **Prompt** with either input mode. Scene requests
retain that guidance through window analysis and final synthesis. They describe
the observed scene unless you explicitly request a change, and do not force the
reference modes' shot counts or endpoint interpolation.
With **Together off**, each supplied image gets its own reference-mode request.

Wan and LTX also combine user guidance with a single image or an ordered scene
sequence. They retain observed poses, opening framing, camera movement and cuts;
stationary moments do not require invented motion. Video duration comes from the
downstream generation settings. The built-in Wan, LTX and H3 prompts describe
action order without adding seconds, timestamps, cut times or total duration.
The former English Wan duration/beat variants resolve to **Wan 2.2**. Untouched
old variants are archived on startup; customized variants retain their own files
and selections. Chinese **Wan 2.2 CN Atomic** remains a separate output format.

### Vision Tasks (all VLM families)

| Task | Description |
|------|-------------|
| **Simple Description** | Concise description of the visible scene |
| **Detailed Description** | Natural, detailed description of action, people, appearance, pose, camera angle, background, and lighting |
| **Ultra Detailed Description** | Fuller connected paragraphs developing action, people, pose, camera angle, setting, textures, colours, and lighting |
| **Cinematic Description** | Film/cinematography style description |
| **Image Analysis** | Technical analysis |
| **Detailed Analysis** | Analytical breakdown |
| **Tags** | Generate comma-separated tags |
| **Video Summary** | Summarize video frames (Qwen only) |
| **Scene Motion Analysis** | Generation paragraph and observations for transferable camera, pose, movement and expression; separate reference supplies appearance |
| **Scene Description (T2V)** | Generation paragraph and observations including visible appearance, setting, lighting and motion |
| **OCR** | Extract text from image |

General vision tasks use short instructions without example conversations. Simple Description requests a compact overview, Detailed Description a developed
paragraph, and Ultra Detailed Description several connected paragraphs with smaller
visible details. They cover subjects, appearance, action, overall pose, camera angle,
setting and lighting. Cinematic Description emphasizes framing and visual style;
analysis tasks examine composition and spatial relationships; Tags returns only
comma-separated tags; Video Summary follows the supplied frames chronologically.

Visual instructions include direct anatomical terms for supported content and keep the
task's requested format and detail. Uncertain or unshown details are omitted silently,
and nudity is not inferred beneath clothing. Edit the task to change its focus or wording.
Updates replace recognized defaults while preserving customized instructions.

### Vision Tasks (Florence-only)

| Task | Florence Token | Description |
|------|---------------|-------------|
| **PromptGen Analyse** | `<ANALYZE>` | Analytical description |
| **PromptGen Mixed Caption** | `<MIXED_CAPTION>` | Mixed-style caption |
| **PromptGen Mixed Caption Plus** | `<MIXED_CAPTION_PLUS>` | Enhanced mixed caption |

### Text Tasks (all families)

| Task | Description |
|------|-------------|
| **Expand Text** | Expand input text into detailed form |
| **Refine & Expand Prompt** | Improve and expand prompts |
| **Rewrite Style** | Rewrite in different style |
| **Tags to Natural Language** | Convert tags to sentences |
| **Natural Language to Tags** | Convert sentences to tags |
| **Translate to English** | Translate to English |
| **Short Story** | Generate a short story |
| **Song Lyrics** | Turn a short story, theme, or song concept into structured, paste-ready lyrics |
| **Summarize** | Summarize text |
| **Prompt Variations** | Generate 5 variations of the same action with different manner / speed / emotion, separated by `---` |

> **Note:** Text tasks work on VLM or LLM models. Without images they run in text-only mode. A vision-capable model can use supplied images alongside your text as context while retaining the selected task's instructions.

---

## Task selection and chaining

Click **Tasks** below **Model** to open the task panel. The left side keeps selected
tasks at the top in execution order, then groups available tasks under Vision and
Text. Search filters available tasks while selected tasks remain easy to deselect.
**Edit** opens a task's fields on the right without changing the selection.
Check compatible tasks to form a chain (up to 32 stages), then use the
numbered selection and arrows to review its order. The same model serves every stage.
Unavailable choices explain their input or model requirements. Missing/deleted saved
tasks remain visible; select a replacement or restore them explicitly.

The first stage receives the user's guidance and supplied images. Each following
stage receives the prior text, its own system instructions and template. Structured scene/H3/music outputs must finish the chain. Lyric sheets can
feed Music 3 or YuE2 conversion. Florence executes one supported native operation.
Legacy Task 2–4 selections retain their saved order.

**Prompt** previews system instructions. **Direct Chat** also provides a local editable
system field. Connected system instructions still select Direct Chat and replace
local instructions. Native ComfyUI encoders receive a system/user pair;
older LLaVA/llama-cpp-python vision handlers combine instructions with the image's
user message. Chat-capable backends retain separate system and user roles.
[Task library](Task_Library.md) describes reusable edits and custom tasks.

### Example: Image → Tags → Natural Language → Expanded Prompt

| Step | Task | Input | Output |
|------|------|-------|--------|
| 1 | Tags | Image | `1girl, long hair, blue eyes, dress...` |
| 2 | Tags to Natural Language | Tags from step 1 | `A girl with long hair and blue eyes wearing a dress...` |
| 3 | Expand Text | Text from step 2 | Detailed expanded description |
| 4 | Refine & Expand Prompt | Text from step 3 | Final polished prompt |

### Notes

- Only Task 1 uses the image input — subsequent tasks are text-only
- Model is loaded once and reused for all tasks
- KV cache is cleared between tasks to prevent VRAM accumulation
- Each stage sends its task instructions and user text without example conversations
- Florence does not support multi-task chaining

---

## Quantization

Quantization is **auto-detected** — there is no manual quantization dropdown for Transformers or Docker backends.

### Transformers

BitsAndBytes precision is auto-selected based on available VRAM:
- **fp16** — full precision, used when enough VRAM is available
- **8-bit** — BitsAndBytes int8, ~50% VRAM reduction
- **4-bit** — BitsAndBytes NF4, ~75% VRAM reduction (fallback when VRAM is tight)

### GGUF

GGUF quantization is built into the file. Select the variant from the **quantization** dropdown:
- **Q4_K_M** — good balance of quality/size
- **Q5_K_M** — better quality, larger
- **Q6_K** — near-lossless
- **Q8_0** — highest quality GGUF

### Docker (vLLM / SGLang / Ollama / llama.cpp)

FP8, AWQ, and GPTQ models are auto-detected from registry metadata. No user configuration needed.

---

## Docker Configuration

Docker backends are configured in `docker_config.json`:

```json
{
  "allow_unpinned_docker_images": false,
  "vllm":     { "docker_image": "vllm/vllm-openai:v0.15.1@sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0", "port": 8000, "allow_mistral_weight_conversion": false },
  "sglang":   { "docker_image": "lmsysorg/sglang:v0.5.9@sha256:e216b7dc4ac1938b599b982233ccf7eb2b11dd1f07fc2e00a7b9841052c553be", "port": 30000 },
  "ollama":   { "docker_image": "ollama/ollama:0.33.1@sha256:075246f72d4109385b4a01c3ac8e9cbd26a0bcb21cd7aa30edbccd24e1b3180c", "port": 11434 },
  "llamacpp": { "docker_image": "ghcr.io/ggml-org/llama.cpp:server-cuda-b8067@sha256:e2c4612f86f6c24408f87f2743fe33063d343c7e9f523ce24a9a60ee401fde05", "port": 8080 }
}
```

- Containers auto-start when needed
- Stop behavior follows the **Keep loaded** Mode option; disable it to release the model after inference
- AMD/ROCm GPUs auto-detected — correct Docker images selected automatically
- Packaged images are pinned by release tag and immutable digest; legacy stock
  aliases resolve to those pins automatically
- Custom images must also use `tag@sha256:digest`. Set
  `allow_unpinned_docker_images` to `true` only for intentional development;
  mutable images disable reproducibility and produce a warning
- Changed image pins or local image identities force exact-container recreation
- Smart LM Manager's **Docker Images** tab can install and select the qualified
  Ollama 0.33.1 runtime or the pinned 0.20.2 legacy-compatibility runtime. This is
  a global runtime choice; it does not modify model registry entries or delete the
  persistent Ollama model store.
- Hugging Face Mistral3/Pixtral-to-native conversion is disabled by default.
  Set `vllm.allow_mistral_weight_conversion` to `true` only after reviewing
  the logged RAM/disk estimate. Conversion is sharded, transaction-marked,
  validated, and considered complete only after its final manifest commits.

---

## Quick Start Examples

### Image Description (Registry Model)

1. Add **Smart LM Loader** node
2. Select model: `Qwen2.5-VL-3B-Instruct` (Transformers)
3. Set task: **Detailed Description**
4. Connect an image to `images`
5. Leave `user_prompt` empty unless the task needs extra context
6. Queue Prompt

### WD14 Tagging

1. Select model: `wd-eva02-large-tagger-v3`
2. Connect an image
3. Adjust threshold (default 0.35)
4. Output: comma-separated booru tags

### Text Expansion (No Image)

1. Select an LLM model (e.g. GGUF text-only)
2. Set task: **Expand Text**
3. Type the source text in `user_prompt`
4. Queue Prompt

### Audio-Video Prompt (Wan, LTX, or MiniMax H3)

1. Select a text-capable model for text-to-video, or a vision-language model when images are required
2. Choose **Wan 2.2**, **LTX 2.3 I2V**, the compact **MiniMax H3 Scene**, or the explicit H3 T2VA/I2VA/FL2VA/L2VA task matching your inputs. Generation settings control duration; task prompts describe action and shot order without numeric timing
3. For H3 reference-based generation, connect no image for T2VA, one first-frame image for I2VA, one first-frame image or an ordered first/last pair for explicit FL2VA, or one last-frame image for L2VA. Legacy retains those count checks. To describe sampled frames, connect the image / video socket or `frame_source` and enable **Together** or **Grid**. Native VIDEO also uses scene mode. The four image-capable H3 tasks retain intermediate samples; long grid/video inputs use bounded windows and a final merge. Off processes images individually
4. For **MiniMax H3 Scene**, enter a short description of what should happen. With Together on and images supplied, an empty prompt describes the observed scene, without automatically inventing a story, cuts, duration, sound or music; explicit user instructions remain active. Without images, or with Together off/Legacy, an empty prompt retains automatic storytelling: zero images create an original story, one image grounds it in that frame, and a Legacy pair supplies its endpoints. That story mode adds scene-appropriate music and sound, with dialogue only when plausible. For a supplied prompt, describe the action, dialogue, sound, pacing, and any camera behavior that should occur. Natural language can request a shot count, such as “tell this in 3 shots.” Requested views are honored in order, so “side view, then POV, then overhead” maps those views to Shots 1–3. “Track Mara” or “track the blue cart” becomes a Tracking Shot naming that target
5. Queue Prompt
6. Send the generated text to the corresponding video workflow

For explicit FL2VA, Picture 1 is the required opening-frame prompt-writing
reference. Picture 2 is optional here because the downstream H3 encoder supplies
and enforces the actual final frame. When both are attached, their order is
contractual. Legacy configurations preserve the reference pair even when
`frame_count` is 1. Together/Grid instead interprets the supplied images as scene
samples, preserving the visible progression plus user guidance.
With both pictures the VLM can trace their observable
differences; with only Picture 1 it briefly describes the visible source state,
continues into the complete `user_prompt` action timeline, avoids inventing unseen
final-frame details, and still ends exactly on encoder-supplied Picture 2.
In reference-based generation without requested cuts, H3 tasks keep one continuous
shot. Requested cuts use ordered `[Shot N]` labels without timestamps. Scene input
preserves observed cuts and framing changes.

The images attached to Smart LM Loader are prompt-writing references: they let
the VLM inspect available keyframes while composing text. They are not keyframe
inputs to the downstream MiniMax H3 encoder. Connect the actual first/last images
separately to the H3 encoder's own keyframe inputs and follow that workflow's
ordering.

The selected task's system instructions request
only the three generator-ready H3 fields as plain text. The complete shot timeline,
soundscape, and music field are all mandatory; a response containing only image
analysis or sound is incomplete.
For nonempty prompts, an explicit request for no background music, no score, or
no soundtrack produces `non_diegetic_music: N/A`. This does not remove dialogue,
ambience, or action sounds from `overall_soundscape` unless silence is also
requested.
An empty **MiniMax H3 Scene** prompt without active scene input instead activates
story mode and creates an appropriate background score. Together/Grid with visual
input describes the observed scene when Prompt is empty. Supplying a nonblank custom `system_prompt` through the pipe
disables automatic story mode and uses the custom instructions. Workflows saved
with former H3 `5s` or `15s` labels are migrated automatically, although those
retired labels are no longer shown in the task dropdown.
The H3 tasks do not ask follow-up questions or add Markdown character, clothing,
camera, or scene sections. When an image-mode action is vague or missing, they
animate only subtle motion supported by visible image content instead of
inventing a new plot; the explicit empty Scene story mode is the exception.
Edit task instructions if a small-context model needs a shorter prompt; preserve the required output fields.

### Stories and songs with visual support

**Short Story**, **Song Lyrics**, **MiniMax Music 3** and **YuE2 Music** accept
text alone or optional images on a vision-language model. Enter the desired
storyline, language, tone or musical style in **Prompt**; images can supply the
setting, subjects, mood and progression. An empty Prompt can use the visuals as
inspiration. Text-only models ignore connected images, VIDEO and disk frames, so
the wiring can stay connected while switching models or tasks. IMAGE passthrough
is preserved, but unused media does not cause extra text requests or frame decoding.

A single image or ordinary Together request goes directly to the selected task.
Grid and VIDEO collect visual notes from each bounded window, combine those
observations, then write one complete story or song from the notes and original
user request. They do not merge independently written songs or stories. This adds
a final text composition call while reusing the loaded model. Together off writes
an independent result for each image.

Output contracts stay unchanged: Short Story returns plain narrative; Song Lyrics
returns a title, production line, lyric sections and Structure line; Music 3
returns `caption`/`lyrics` JSON; YuE2 returns `style`/`lyrics` JSON. During conversion
from existing lyrics, visual context can inform unspecified musical direction,
while the application preserves the retained lyrics exactly.

### Song Lyrics

1. Select a text-capable LLM, or a vision-language model when supplying images, and set task to **Song Lyrics**
2. Enter a short story, theme, mood, or song concept in `user_prompt`
3. Optionally include genre, language, tempo, point of view, or desired structure
4. Queue Prompt
5. Copy the returned title, production line, lyric sections, and structure into Suno, Mureka, or another music-generation tool

Here `user_prompt` supplies the concept and direction; optional images can provide
visual inspiration. The task's built-in system prompt turns that source into singable, consistently
formatted lyrics; supplying a custom `system_prompt` is unnecessary unless
you intentionally want to replace the entire task behavior.

### Direct Chat with a Prompt Pipe

1. Connect a STRING containing the complete custom instructions to **Smart LM Prompt Pipe → system_prompt**
2. Include every important role, goal, constraint, and output-format rule in that text
3. Connect the request, question, or source content to its `user_prompt` input, or keep using the loader's local Prompt field
4. Connect the pipe's `settings` output to the loader's `pipe_opt`, then queue the prompt

Supported models use **Direct Chat** while the pipe supplies nonblank system instructions. The selected
task's system prompt is not combined with the connected text, so omitted rules
cannot be inherited from that task.

### Docker Backend (Ollama)

1. Select model: `Qwen2.5-VL-7B -Ollama`
2. Docker daemon starts automatically if installed
3. Model auto-pulls on first use
4. Disable **Keep loaded** to free VRAM after

---

## Troubleshooting

### Model Not in Dropdown

Models are loaded from JSON registry files in `registry/`. User models can be added to `registry/user_models.json`.

### Florence-2 Not Loading

Check that the model path exists and files are complete. Florence-2 uses a custom loading path via `florence2_wrapper.py`.

### Mistral3 Not Loading

Mistral3 requires transformers v5.0+:
```bash
pip install transformers>=5.0.0
```

### Docker Container Won't Start

1. Check Docker is running: `docker info`
2. Check GPU access: `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`
3. Check `docker_config.json` settings
4. Check container logs: `docker logs <container_name>`

### Out of Memory (OOM)

1. Enable **Cleanup** button (mode bar)
2. Use GGUF quantization (Q4_K_M)
3. Disable **Keep Loaded** so Docker containers stop after the node finishes
4. Reduce `context_size`
5. Reduce `max_tokens` when the prompt plus requested completion approaches the context window
6. Use a smaller model
7. Reduce **Longest side** in Image handling and the number of frames per request.
   Together ignores the legacy Max frames cap; use upstream sampling/windows or
   Every Nth frame to bound the selection.

### GGUF Vision Not Working

Ensure mmproj file exists in the same folder as the GGUF model. Registry models auto-download mmproj files. For llama.cpp Docker, verify with: `curl http://localhost:8080/props` (should show `"vision": true`).

### Debug Logging

Enable in `config.json`:
```json
{ "log_level": "debug" }
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config.json` | Main config: LLM folder path, log level, HF token |
| `docker_config.json` | Docker backend settings: ports, timeouts, images |
| `registry/*.json` | Model registry files (8 backend files + defaults + user_models) |
| `tasks/*.json` | Per-task instructions, user templates and supported behavior |
| `tasks/.state.json`, `tasks/.trash/`, `tasks/.legacy/` | Default revisions, removed tasks and legacy migration backups |
| `.defaults/` | Git-tracked defaults (`.example` suffix); extracted to repo on first run |

---

## Image sequences and scene motion

**Scene JSON** in the Mode bar defaults off and is saved per node. Enable it when
using ordinary image tasks, including Detailed/Ultra Detailed Description or H3,
with Eclipse's **Scene Description Collector**. The task keeps its normal model
instructions; Python serializes the result, so quotes, line breaks and braces
cannot break the JSON. There is no extra inference call. Off preserves the normal
text output, including existing workflows without this field. Text-only requests
and WD14 tagging remain unchanged. Existing Scene Motion Analysis and Scene
Description (T2V) tasks keep their structured observations and validation.

Use **Together: On** for one description of each sampled scene window. Connect the
sampler's context through the Prompt Pipe's **user_prompt** to retain the measured
scene/window IDs. The portable `scene_caption` version-1 envelope contains
`scene_id`, `window_id`, `description_mode: "full"`, `source_task`,
`performance_prompt` and `source_response`. The complete response is kept for
review; the scene paragraph joins its whitespace. H3 uses its
`integrated_multimodal_description` without the opening `[Shot 1]` label, retaining
the full H3 response separately. It does not invent pose or action audit fields.
For task chains, the final task's response is packaged.

Select **include source appearance** in Scene Analysis to H3 for these captions:
the selected task's wording is retained, with no automatic appearance removal.
Ordinary task output can still contain mistaken or invented content; JSON packaging
does not improve visual understanding. Request evidence records the actual supplied
positions/count for Together and individual images. Legacy multi-image transports
can apply backend-specific caps, so captions in that compatibility mode omit
unverified input evidence. Use Together for auditable scene windows.

Text tasks such as **Refine & Expand Prompt**, **Tags to Natural Language** and **Natural Language to Tags** work with text alone. Without images, image grouping, sizing and frame stride do not process anything or require an image. With a vision-capable model and images, those same tasks receive both the images and your text, so the Prompt can give focus instructions or context. The selected task instructions remain active. Together sends the selected images in one request; Off sends one image per request with the corresponding text. Later multi-task stages still process the preceding text result. Models without vision weights cannot use image context, and model-specific limits still apply to supplied images.

The **Image handling** section below Model contains **Together**, **Longest side** and **Every Nth frame**. It opens and collapses independently of Model, shows the current image settings in its header, and can be pinned open while other sections are used. The legacy **Max frames** control also appears here when applicable. Values are saved per node and remain available for Direct Chat and custom system instructions.

- **Image handling** uses a **Together** toggle: **On** sends the ordered selection in one request; **Off** runs one request per supplied image, including H3 image tasks. Older configurations show **Legacy** until clicked, preserving their saved behavior; the first click enables Together. Together bypasses the legacy last-N frame cap; upstream sampling/windowing bounds the request. Unsupported multi-image processor paths fail explicitly. Existing configurations without the switch retain their historical behavior. H3 scene sequences and first/last pairs use Together.
- **Longest side = 0** keeps automatic model sizing. A positive value reduces input dimensions while respecting model limits and avoiding source upscaling. Dynamic patch alignment may round down. Known fixed/minimum processor sizes are checked; remote servers still own their final preprocessing. The ordinary IMAGE output retains the original input pixels.
- **Every Nth frame** applies to Together or Grid for IMAGE/disk input, defaults to 1, and always includes the first and last supplied image. It samples the supplied batch, which may already have been sampled upstream. Two at 30 source fps gives 15 samples/second only when the input actually contains every original frame. Use 1 to retain brief gestures/expressions or when sampling upstream.

**Scene Motion Analysis** is for image-to-video. It transfers camera geometry, posture, visible movement and expression to a separate character reference. The target reference supplies appearance and setting. The task explicitly instructs the model to keep source identity, gender, clothing, accessories, colors and environment out of every output field, including `performance_prompt`, and asks the model to check that scope before returning. The paragraph and supporting observations use the same inference per window. Use neutral subject/body terms and enough grounded detail for the scene, without a fixed sentence count or word limit. Physical contact remains an action without a description of the contacted item's appearance. The schema needs no `appearance` field. These instructions guide the model; review warnings remain advisory and cannot guarantee compliance.

**Scene Description (T2V)** includes relevant visible subjects, appearance, clothing, accessories, setting, lighting and motion directly in the generation paragraph. Its additional `appearance` field is optional supporting metadata. Omitting it needs no correction call; blank or structured values are retained with a review warning. Generation uses the unchanged `performance_prompt` paragraph. Both tasks use the same ordered images and one normal inference per window, with the existing bounded correction attempt for invalid output. Context may supply integer `scene_id` and `window_id` plus frame IDs/timestamps. A single image establishes a pose without invented movement. Neither task infers audio or unseen events.

The loader records the actual task policy as `description_mode: motion` or `full`. The optional [Eclipse Nova scene-analysis nodes](https://github.com/r-vage/ComfyUI_Eclipse_Nova/blob/main/Readme/Scene_Analysis.md) measure cuts, collect descriptions and export H3 part prompts. The collector accepts motion results without appearance and preserves new task paragraphs without appearance rewriting. Use the H3 adapter's **exclude source appearance** with Motion Analysis, or **include source appearance** with T2V. The selected task defines the intended policy; it cannot guarantee model compliance. Changing policy requires another analysis run because motion analysis does not request a complete appearance description. Custom system prompts remain user-owned and must follow the selected structure. Prompt compliance and observation accuracy still need review.

The paragraph and observations use the same inference call per window. Multi-window Grid or VIDEO requests add text synthesis; a single window does not. Usable text with recognized appearance, gendered wording or movement misplaced in boundary poses is preserved with `review_warnings`. Length alone is not a review warning. Content warnings neither retry generation nor stop the scene loop. Eclipse saves the text and reports the warnings so you can copy part prompts into the manual inputs and edit them. These checks recognize bounded patterns and cannot establish semantic correctness.

If `actions`, `gaze_expression` or `uncertainties` arrives as a text string, the loader accepts it as one list item with its wording intact. A blank string becomes an empty list. This normalization happens before validation, so it needs no correction inference. Commas are preserved rather than treated as action separators.

Only structural failures—invalid JSON, missing required fields, non-text list values or mismatched IDs—use the existing single correction attempt. A complete single JSON object is accepted inside an optional prose/code-fence wrapper without another call; partial JSON, arrays and multiple competing objects remain invalid. Persistent structural failures remain errors rather than silently losing scenes. New task paragraphs pass through without text cleanup; older unmarked saved analyses retain their existing cleanup. Measured cut times come from detection. A custom system prompt intentionally replaces task instructions without changing grouping or size. Recognized older defaults upgrade to unified task instructions; customizations remain user-owned and must satisfy the selected structure.

The structural correction retains the original images and provides the validation issue. It does not insert the rejected answer into the prompt. A usable description takes one inference, including when it needs manual content review.

Every loader task that accepts images can use **frame_source** instead. The source changes how pixels are loaded; the selected task keeps its instructions and output format. Together describes the selected scene in one request on a supported VLM; Off processes frames individually. Text tasks retain their text instructions and can use a supplied image as context on a vision model. Florence and WD14 accept disk frames individually; turn Together off for multiple frames with those models. Text-only models ignore connected visual sources; image-required tasks still require a vision model.

For file-backed loops, the optional **frame_source** socket accepts the public `SMARTLLM_FRAME_SOURCE` version-1 protocol: an object with `version == 1`, `len(source)` in 1–256 and `iter_frames()` yielding ordered HWC RGB uint8 arrays. The producer owns the files and keeps them alive. Smart LM validates the source before loading a model, loads/resizes only the active request, closes the iterator on success or failure and returns an **empty IMAGE list** for this input to avoid retaining pixels in graph caches. Do not connect both images and frame_source. Regular IMAGE passthrough is unchanged. Existing images/seed/pipe socket indices are preserved.

Successful scene-task JSON receives measured `input_evidence`: selected positions, count, dimensions, grouping, longest-side setting and stride. This metadata is supplied by the loader, not inferred by the model. Eclipse's scene collector maps it to source frame IDs and presentation timestamps.

Backend processors can differ in image limits and motion interpretation. Review the descriptions against their sampled frames when changing models, resolution or sampling density.

### Scene poses and ongoing motion

Both scene tasks instruct the model to begin `performance_prompt` with the observed shot size, camera angle, composition and visible starting pose, then describe the action and expression. Include head/torso facing direction and pose-defining hand placement when visible. A still pose still needs a description: "static" alone does not tell the video model which posture to hold. Keep these details in the generation paragraph itself; the separate `framing` and `start_pose` fields are supporting observations and are not added to H3 prompts. Existing saved paragraphs need another analysis run or an edit in Prompt Review to gain missing details.

Scene Motion Analysis treats each window as an excerpt that can begin mid-gesture. Its start/end pose fields describe positions already present in the boundary frames; actions describe observed changes between frames, including small finger/wrist movements while the arms stay raised. Partial or ambiguous arcs belong in uncertainty. Recognized movement verbs in pose fields produce a review warning while preserving the description. Existing saved descriptions cannot recover motion omitted by an earlier analysis, so rerun those windows for the updated task.

### Concise scene observations and black-frame filtering

Scene Motion Analysis summarizes distinct motion and expression changes rather than writing an event per sampled image. It contains no example gesture to copy. Routine speech-like lip motion is summarized once; visible distinctive expressions remain relevant. These instructions improve concision but do not make model observations factual by themselves.

The loader removes exactly black frames after grouping, stride and resizing, before model inference. Dark frames with any nonzero pixel remain. The request context includes only the frame IDs/timestamps actually sent, with their image count and first/last image ordinals. The model describes the first and last remaining visible samples, without instructions to interpret the omitted black images. Ordinary IMAGE passthrough is unchanged; lazy disk inputs retain no image passthrough.

`input_evidence.positions` records sent positions and `omitted_black_positions` records removed positions, both zero-based within the original supplied selection. If all selected images are black, the node skips inference and emits `not visible` audit strings, a `Black screen.` performance paragraph, empty action/expression lists, `image_count: 0` and `skipped_reason: all_black`. Model loading may still occur before frame preparation. Eclipse's collector preserves these entries and maps sent and omitted positions back to measured source frames/times. Scene boundaries and audio timing do not change.

Filtering applies to both scene tasks. Other tasks and custom-system-prompt Direct Chat retain their existing image behavior. Together/One by one remains an independent control; Every Nth applies only to Together. Each all-black selection skips generation, including chained tasks for that selection. Motion-only all-black results also omit appearance; T2V marks it not visible.

Filtering removes empty visual evidence, but the model can still misread partially visible hands or motion between samples. Review the resulting directions before generation. Recognized bundled task prompts upgrade automatically; customized prompts are preserved.
