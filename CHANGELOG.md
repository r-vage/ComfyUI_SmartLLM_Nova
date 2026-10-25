## 2026-10-25

### Version: 1.1.6

- **Fix**
  - Show missing-image warnings only for tasks that require an image, allowing text tasks and optional-image tasks such as Direct Chat to run on vision models without a misleading toast.
- **Changed files**
  - py/RvLoader_SmartLMLoader.py
  - pyproject.toml

## 2026-10-24

### Version: 1.1.5

- **Feat**
  - Load supported GGUF language encoders directly from ComfyUI text_encoders using a bundled adapter, with validated same-folder vision projectors, read-only model groups and projector-aware cache invalidation.
  - Separate native ComfyUI encoders into vision and text-only model groups using checkpoint contents rather than filenames, retaining sorted selections and capability-based tasks.
- **Fix**
  - Preserve Qwen2.5-VL generation output heads, stop tokens and image positions through the native backend; exclude incomplete 7B checkpoints without an output head.
- **Changed files**
  - Readme/Smart_LM_Loader_Guide.md
  - core/sml/backend_comfyui.py
  - core/sml/comfy_qwen25.py
  - core/sml/gguf_metadata.py
  - core/sml/gguf_text_encoders.py
  - core/sml/model_registry.py
  - core/sml/panel_configuration.py
  - core/sml/text_encoders.py
  - extern/gguf/
  - js/nova-panel-controls.js
  - js/nova-panel-form.js
  - pyproject.toml
  - requirements.txt

## 2026-10-23

### Version: 1.1.4

- **Fix**
  - Remove the unavailable Huihui Ministral 3B Instruct and 8B Reasoning Transformers entries from the bundled model registry after verifying their sources return 404.
- **Docs**
  - Remove unavailable Ministral repository links and copy/paste examples from the model references.
- **Changed files**
  - .defaults/registry/transformers_models.json.example
  - Readme/Model_Repos_Reference_CP.md
  - Readme/Model_Repos_Reference_Links.md
  - pyproject.toml

## 2026-10-23

### Version: 1.1.3

- **Fix**
  - Keep Florence-2 generation compatible with Transformers 4 and 5. Preserve checkpoint token, beam and repetition defaults, isolate per-request generation overrides and retain the vendored decoder's tuple-cache path.
  - Restore Florence-2 image resizing, rescaling and normalization with Transformers 5. Prevent rectangular-image detection/captioning failures while preserving detection coordinates in the original image.
- **Changed files**
  - core/sml/backend_transformers.py
  - core/sml/florence2_wrapper.py
  - extern/florence2/modeling_florence2.py
  - extern/florence2/processing_florence2.py
  - pyproject.toml

## 2026-10-22

### Version: 1.1.2

- **Fix**
  - Preserve manually chosen panel widths when expanding or collapsing mode bars and sections, while continuing to fit the controls vertically.
- **Changed files**
  - js/nova-configuration-panel.js
  - pyproject.toml

## 2026-10-21

### Version: 1.1.1

- **Fix**
  - Use 0.1 increments for general float controls, retaining finer precision where defaults or bounds require it. Preserve saved values and value ranges.
  - Keep WD14 tag thresholds at hundredth precision so the default 0.35 and 0.85 values remain valid in numeric controls.
- **Changed files**
  - py/RvLoader_SmartDetection.py
  - py/RvLoader_SmartLMLoader.py
  - pyproject.toml

## 2026-10-19

**Version: 1.1.0**

**Feat**
- Add Neutral Jade, Neutral Ocean, Neutral Amethyst and Neutral Amber, pairing Neutral node backgrounds and titles with each palette's colored controls. Eclipse's Theme groups option applies matching Neutral group colors.
- Add native ComfyUI text encoder generation alongside the existing Smart LM backends, with compatible local files grouped in the model selector.
- Reuse ComfyUI loading, tokenization, generation and memory management with Nova tasks, prompts, training examples, image inputs, sampling and Keep loaded behavior.
- Keep shared text encoder files outside registry editing and deletion; preserve saved selections and hide controls that do not apply to native generation.

**Fix**
- Resolve native CUDA selections to a concrete GPU index before loading and cache reuse, preventing failures in ComfyUI's dynamic VRAM manager.

**Perf**
- Discover compatible encoders from cached checkpoint headers and invalidate model/output reuse when files change.

**Docs**
- Document the new variants, linked selection across the Nova packs and group appearance.
- Document native encoder selection, supported files, sampling behavior and shared-file ownership.

**Changed files**
- `.defaults/themes/neutral-jade.json.example`, `.defaults/themes/neutral-ocean.json.example`
- `.defaults/themes/neutral-amethyst.json.example`, `.defaults/themes/neutral-amber.json.example`
- `Readme/Node_Design.md`, `Readme/Nova_Panels.md`, `Readme/Visual_Tour.md`
- `core/sml/backend_comfyui.py`, `core/sml/text_encoders.py`, `core/sml/loader_base.py`
- `core/sml/model_acquisition.py`, `core/sml/model_cache.py`, `core/sml/model_files.py`, `core/sml/model_registry.py`
- `core/sml/panel_configuration.py`, `py/RvLoader_SmartLMLoader.py`
- `js/nova-panel-controls.js`, `js/nova-panel-form.js`, `js/smartllm-registry-manager.js`
- `Readme/Smart_LM_Loader_Guide.md`, `pyproject.toml`

## 2026-10-16

**Version: 1.0.0 — Initial release**

- Five compact panel nodes for language/vision models, image tagging, detection, mask/bbox conversion and structured song prompts, with native and Docker backends.
- A shared panel design inspired by DaSiWa, adapted for classic ComfyUI and Nodes 2.0 with searchable model/task selectors, collapsible sections, pinned sections and resizable text fields.
- Controls update through editing and lifecycle events, reducing canvas work without native-widget visibility or recurring graph scans.
- Independent Nova node IDs, menu, routes, settings, configuration and container ownership allow the original SmartLLM pack to remain installed. Saved server settings hydrate the UI, and native/Docker backends share the current model folder.
- Connected prompt, detection and seed overrides, chained tasks, independent Standard/NSFW training selection per loader, exact 64-bit seed controls, and preserved image-batch and prompt-list processing.
- Registry Manager retains model selections and backend quantization across reopening and follows the selected theme, with grouped model/source/integrity sections, matching mode bars and controls, and Docker cards for setup diagnostics, immutable image references and runtime actions.
- Neutral as the default node design, plus Jade, Ocean, Amethyst, Amber and custom JSON themes. Theme selection persists in Nova’s config; bundled palettes are extracted once and preserve user edits and deletions.
- Installable dependency extras and the bundled Florence implementation with declared vision dependencies and actionable dependency messages.
- Guides and an illustrated visual tour covering the Nova panels, model selection, detection and converters, plus detailed model registration, file verification and Docker/Ollama runtime walkthroughs.
