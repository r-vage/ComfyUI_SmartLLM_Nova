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

---

## Overview

The **Smart LM Loader** is a unified node for loading and running vision-language models, text-only LLMs, and WD14 taggers in ComfyUI. It uses a **registry-based workflow** — pick a model from the unified dropdown, choose a task, and generate. No templates, no manual path resolution.

### What Can This Node Do?

- **Image Analysis** — describe, analyze, and extract information from images
- **Text Generation** — chat, expand prompts, translate, summarize
- **Tag Generation** — WD14 tagger with booru-style output
- **Video Analysis** — summarize video sequences (Qwen)
- **Prompt Pipelines** — chain up to 4 tasks sequentially via multi-task mode
- **OCR** — extract text from images

> **Note:** For object detection with bounding boxes, masks, and SEGS output, use the [Smart Detection](Smart_Detection_Guide.md) node instead.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Registry-Based Model Selection** | Unified dropdown with 50+ models grouped by backend — no templates, no manual paths |
| **Auto-Download** | Full repositories download from Hugging Face; targeted GGUF/mmproj files can also use ModelScope, with immutable revision and integrity verification |
| **8 Backends** | Transformers, GGUF, vLLM, SGLang, Ollama, llama.cpp, YOLO, WD14 |
| **Multi-Task Chaining** | Chain 2–4 sequential tasks with output→input flow |
| **Few-Shot Training** | Per-task example pairs (user-editable in `config/`), toggleable via **Training** button |
| **Editable System Prompts** | Customize per-task instructions in `config/system_prompts.json` |
| **Mode Bar** | Cleanup, Keep loaded, Multi-task, Training, Standard/NSFW, Sampling, Local seed |
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
| **task** | Dropdown | Detailed Description | Task to perform. Vision tasks require an image |
| **task_2** | Dropdown | None | Optional 2nd task (multi-task mode) |
| **task_3** | Dropdown | None | Optional 3rd task (multi-task mode) |
| **task_4** | Dropdown | None | Optional 4th task (multi-task mode) |
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
| **Multi-Task** | OFF | Enable sequential task chaining (shows task_2/3/4) |
| **Sampling** | ON | Enable custom sampling values |
| **Standard / NSFW** | Standard | Select this node's training examples. The button switches between Standard and NSFW; it is inactive while Training is off. Other nodes and root config are unchanged. |
| **Training** | ON | Include few-shot training examples in the prompt. Disable to reduce context size (saves ~1–3 KB per prompt; system prompts and task instructions still load) |
| **Local seed** | ON | Use the Seed section; disable to use a connected seed_input |

The **Sampling** mode button enables custom sampling values. Model contains
Registry, Delete and Refresh actions; full verification is available in Registry Manager.

#### Sampling and hardware sections

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| **device** | Dropdown | cuda | Compute device |
| **temperature** | Float | 0.7 | Sampling temperature (0.1–2.0) |
| **top_p** | Float | 0.9 | Nucleus sampling (0.1–1.0) |
| **top_k** | Integer | 50 | Top-k sampling, 0=disabled |
| **num_beams** | Integer | 1 | Beam search count. 1=greedy |
| **do_sample** | Boolean | True | Enable sampling vs greedy decoding |
| **repetition_penalty** | Float | 1.0 | Repeat penalty (1.0–2.0) |
| **frame_count** | Integer | 8 | Qwen VL — video frames to analyze |
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
| **images** | IMAGE | Optional. Image input for vision tasks and WD14 |
| **user_prompt** | STRING | Optional prompt override; supports prompt lists. Empty text is a valid override |
| **seed_input** | INT | Optional unsigned 64-bit seed, used when Local seed is disabled |
| **system_prompt** | STRING | Optional complete instruction override. Connecting text switches the node to Direct Chat and bypasses the selected task's default prompt template |

The connected `system_prompt` is not appended to the selected task. It replaces
that task's built-in instructions, so the provided text must be self-contained:
include the model's role, the complete objective, any rules or constraints, and
the required output format. In this mode, `user_prompt` is the corresponding user
message or source material. Leave `system_prompt` disconnected when you want to
use a predefined task such as Detailed Description, Wan/LTX/MiniMax H3 prompting,
or Song Lyrics.

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| **image** | IMAGE | Passthrough of input images (or 64×64 placeholder if no image) |
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
| **Wan 2.2 Scene 5s** | Optional | One cinematic paragraph for a 5-second Wan 2.2 video (no timeline markers) |
| **Wan 2.2 Timeline 5s** | Optional | One paragraph using `(At 0 seconds: ...) ... (At 5 seconds: ...)` markers (per-second beats) |
| **Wan 2.2 Timeline 5s 2s** | Optional | Slower 3-beat timeline `(At 0s)(At 2s)(At 4s)` — each beat unfolds before the next |
| **Wan 2.2 Timeline 5s 3s** | Optional | Slowest 2-beat timeline `(At 0s)(At 3s)` — long, unhurried action arcs |
| **Wan 2.2 Scene 10s** | Optional | Two continuous 5-second scene paragraphs with subject and setting continuity |
| **Wan 2.2 Timeline 10s** | Optional | Two 5-second timeline paragraphs with continuous action and per-second beats |
| **Wan 2.2 Scene 20s** | Optional | Four cinematic paragraphs (5s each) with maintained character / scene continuity |
| **Wan 2.2 Timeline 20s** | Optional | Four timeline paragraphs (5s each), each with per-second markers |
| **Wan 2.2 CN Atomic** | Optional | Chinese Wan prompt using explicit initial state, ordered physical actions, and final state |
| **LTX 2.3 I2V** | Recommended | One image-to-video paragraph combining the reference image with requested motion, sound, dialogue, and style |
| **MiniMax H3 Scene** | 0–2 | One continuous duration-neutral scene; zero/one/two images select T2VA/I2VA/FL2VA, and an empty prompt enables automatic storytelling |
| **MiniMax H3 T2VA Timeline** | None | Text-only shot timeline planned within H3's maximum 15-second range |
| **MiniMax H3 I2VA Timeline** | First | Timeline beginning exactly on one first-frame image |
| **MiniMax H3 FL2VA Timeline** | First; optional last | Timeline starting from the first reference and landing on the encoder-supplied last frame; an optional second reference lets the VLM inspect both endpoints |
| **MiniMax H3 L2VA Timeline** | Last | Timeline converging on one last-frame image |

### Vision Tasks (all VLM families)

| Task | Description |
|------|-------------|
| **Simple Description** | Brief one-sentence description |
| **Detailed Description** | Paragraph-length description |
| **Ultra Detailed Description** | Comprehensive description |
| **Cinematic Description** | Film/cinematography style description |
| **Image Analysis** | Technical analysis |
| **Detailed Analysis** | Analytical breakdown |
| **Tags** | Generate comma-separated tags |
| **Video Summary** | Summarize video frames (Qwen only) |
| **OCR** | Extract text from image |

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

> **Note:** Text tasks work on VLM or LLM models. When a VLM has no image connected, text tasks run in text-only mode automatically.

---

## Multi-Task Mode

Chain 2–4 sequential tasks where each task's output becomes the input for the next.

### How It Works

1. Enable the **Multi-task** button on the mode bar
2. Set **task_2** (and optionally task_3, task_4)
3. **Task 1** loads its task-specific system prompt and runs with the image plus any optional `user_prompt` context
4. **Task 2** receives the text output from Task 1
5. **Task 3/4** continue the chain
6. Final output is returned

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
- Few-shot training is applied per task (disable with the **Training** button to reduce context pressure on small-context models)
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
2. Choose the Wan/LTX task, the compact **MiniMax H3 Scene**, or the explicit H3 T2VA/I2VA/FL2VA/L2VA timeline task matching your inputs. H3 task labels are duration-neutral; timeline prompts use the model's 15-second maximum as their planning range unless you request a shorter duration
3. For H3, connect no image for T2VA, one first-frame image for I2VA, one first-frame image or an ordered first/last pair for explicit FL2VA, or one last-frame image for L2VA. The Scene task still infers T2VA/I2VA/FL2VA from zero/one/two images. Other count mismatches fail before generation
4. For **MiniMax H3 Scene**, either enter a short description of what should happen or leave `user_prompt` empty for an automatically generated story. With an empty prompt, zero images create an original story, one image grounds the story in that frame, and two images create a natural progression between the endpoints. SmartLLM adds scene-appropriate music and sound, with dialogue only when plausible. For a supplied prompt, describe the action, dialogue, sound, pacing, and any camera behavior that should occur. Natural language can set a feasible one-to-four shot count for timeline tasks, such as “tell this in 3 shots.” Requested views are honored in order, so “side view, then POV, then overhead” maps those views to Shots 1–3. “Track Mara” or “track the blue cart” becomes a Tracking Shot naming that target
5. Queue Prompt
6. Send the generated text to the corresponding video workflow

For explicit FL2VA, Picture 1 is the required opening-frame prompt-writing
reference. Picture 2 is optional here because the downstream H3 encoder supplies
and enforces the actual final frame. When both are attached, their order is
contractual and SmartLLM keeps the pair together as one VLM reference group even
when `frame_count` is 1. With both pictures the VLM can trace their observable
differences; with only Picture 1 it briefly describes the visible source state,
continues into the complete `user_prompt` action timeline, avoids inventing unseen
final-frame details, and still ends exactly on encoder-supplied Picture 2.
Without an explicit shot count, timeline tasks create two to four purposeful,
duration-safe shots within the maximum 15-second planning range and do not invent
extra camera angles.

The images attached to Smart LM Loader are prompt-writing references: they let
the VLM inspect available keyframes while composing text. They are not keyframe
inputs to the downstream MiniMax H3 encoder. Connect the actual first/last images
separately to the H3 encoder's own keyframe inputs and follow that workflow's
ordering.

The selected task's built-in system prompt and optional few-shot training produce
only the three generator-ready H3 fields as plain text. The complete shot timeline,
soundscape, and music field are all mandatory; a response containing only image
analysis or sound is incomplete.
For nonempty prompts, an explicit request for no background music, no score, or
no soundtrack produces `non_diegetic_music: N/A`. This does not remove dialogue,
ambience, or action sounds from `overall_soundscape` unless silence is also
requested.
An empty **MiniMax H3 Scene** prompt instead activates story mode and always creates
an appropriate background score. Connecting a nonblank custom `system_prompt`
disables automatic story mode and uses the custom instructions. Workflows saved
with former H3 `5s` or `15s` labels are migrated automatically, although those
retired labels are no longer shown in the task dropdown.
The H3 tasks do not ask follow-up questions or add Markdown character, clothing,
camera, or scene sections. When an image-mode action is vague or missing, they
animate only subtle motion supported by visible image content instead of
inventing a new plot; the explicit empty Scene story mode is the exception.
Disable the **Training** button when a small-context model needs a shorter prompt.

### Song Lyrics

1. Select a text-capable LLM and set task to **Song Lyrics**
2. Enter a short story, theme, mood, or song concept in `user_prompt`
3. Optionally include genre, language, tempo, point of view, or desired structure
4. Queue Prompt
5. Copy the returned title, production line, lyric sections, and structure into Suno, Mureka, or another music-generation tool

Here `user_prompt` is the creative source rather than an optional correction.
The task's built-in system prompt turns that source into singable, consistently
formatted lyrics; connecting a separate `system_prompt` is unnecessary unless
you intentionally want to replace the entire task behavior.

### Direct Chat with a Connected System Prompt

1. Connect a STRING containing the complete custom instructions to `system_prompt`
2. Include every important role, goal, constraint, and output-format rule in that text
3. Put the request, question, or source content in `user_prompt`
4. Queue Prompt

The node changes to **Direct Chat** while the connection is present. The selected
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
| `config/system_prompts.json` | Per-task system prompts (user-editable) |
| `config/llm_few_shot_training.json` | Standard few-shot training examples |
| `config/llm_few_shot_training_nsfw.json` | NSFW few-shot training examples |
| `.defaults/` | Git-tracked defaults (`.example` suffix); extracted to repo on first run |

---

*Guide for ComfyUI SmartLLM v3.3.4*
