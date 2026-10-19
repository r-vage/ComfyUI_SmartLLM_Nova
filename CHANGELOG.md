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
