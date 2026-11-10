## 2026-11-10

### Version: 1.1.14

- **Fix**
  - Update Scene Motion Analysis and Scene Description (T2V) instructions to lead generation paragraphs with framing and the visible starting pose, including facing direction and relevant hand placement. Describe stationary poses explicitly and remove conflicting guidance to omit resting details.
  - Upgrade recognized scene-task defaults in both Standard and NSFW profiles while preserving customized prompts.
- **Docs**
  - Explain that H3 receives the generation paragraph and that existing scene descriptions need to be regenerated or edited to include missing framing and pose.
- **Changed files**
  - .defaults/config/system_prompts.json.example
  - .defaults/config/system_prompts_nsfw.json.example
  - Readme/Smart_LM_Loader_Guide.md
  - core/migration.py
  - core/sml/scene_motion.py
  - pyproject.toml

## 2026-11-09

### Version: 1.1.13

- **Feat**
  - Add an opt-in Scene JSON mode for ordinary image-task descriptions. Build the JSON in code without changing model instructions or adding inference; retain the original text and measured request evidence, and use H3's visual field as the scene paragraph. Keep direct text output and existing structured scene tasks compatible.
- **Fix**
  - Accept lazy disk frames for every loader task that supports images, including MiniMax H3 and individual WD14 tagging. Preserve frame order, resizing, source cleanup and empty IMAGE output for disk inputs.
  - Apply Together consistently to H3: describe all selected frames as one observed scene, or process each image separately when off. Keep intermediate samples, the selected H3 output format and explicit user instructions; avoid automatic story invention for scene sequences. Retain Legacy reference counts and text-only T2VA behavior.
- **Docs**
  - Clarify interchangeable image sources, H3 scene sequences, optional image context for text tasks and individual processing with Florence/WD14.
- **Changed files**
  - README.md
  - Readme/Nova_Panels.md
  - Readme/Smart_LM_Loader_Guide.md
  - core/sml/image_inputs.py
  - core/sml/scene_motion.py
  - core/sml/tasks.py
  - js/nova-panel-form.js
  - py/RvLoader_SmartLMLoader.py
  - pyproject.toml

## 2026-11-07

### Version: 1.1.12

- **Feat**
  - Add a separate Image handling section with an independent Together on/off button, longest-side sizing, every-Nth-frame selection and the applicable legacy frame limit. Keep it accessible with Model collapsed, with its own summary and pin state. Show unchanged legacy grouping until the button is clicked. Retain sequence endpoints, bypass legacy tail truncation for explicit grouping, preserve old configurations and existing socket indices.
  - Add Scene Motion Analysis for image-to-video and Scene Description (T2V) for visible appearance, setting and motion. Focus motion instructions on transferable framing, existing poses, movement and expression, with no required appearance audit. Write concise generation paragraphs and observations in one response and record the selected description mode. Upgrade recognized task defaults while preserving custom prompts.
  - Filter fully black frames before scene description, audit sent and omitted positions, and skip inference for all-black selections while retaining a black-screen paragraph. Accept lazy disk-frame sources without retaining an IMAGE passthrough in analysis loops.
- **Fix**
  - Let text tasks such as refinement and tag conversion use supplied images as optional context on vision models. Preserve task instructions alongside user hints for all VLM families, leave image sequence handling inactive for text-only requests, and retain custom system instructions when no image or user text is supplied.
  - Clarify the Scene Motion Analysis scope in every response field, including the generation paragraph: transfer camera geometry, posture, motion and expression while leaving appearance and setting to the target reference. Keep full T2V descriptions separate and preserve usable responses with advisory warnings.
  - Keep Standard/NSFW profile selection independent of Training, including the Mode button and collapsed summary. Turning Training off now omits examples while retaining the selected system prompts.
  - Apply the existing NSFW profile guidance to active MiniMax H3 prompts. Remove retired H3 few-shot keys and preserve their richer railway example under T2VA; upgrade recognized defaults without replacing custom entries.
  - Preserve usable scene descriptions with review warnings for unwanted appearance, verbosity or questionable pose wording. Accept a single complete JSON object inside a prose/code-fence wrapper, and scalar action, gaze/expression and uncertainty text as list entries without rewording or corrective inference. Treat T2V appearance audits as optional metadata and retain unusual formats for review without corrective inference. Avoid retrying content complaints; reserve the single correction attempt for invalid JSON, missing required fields or mismatched IDs so users can copy and edit results manually.
  - Apply the Thinking switch to Ollama, Transformers, vLLM, SGLang and llama.cpp server requests as well as native ComfyUI encoders. Keep the choice off by default and use supported model-template controls for text, images and task chains. Hide it for local llama-cpp-python GGUF and non-chat detection models. Keep reasoning separate from final answers, including Transformers templates that prefill the opening thinking tag.
- **Docs**
  - Refresh the README and guides for image handling, text-plus-image tasks, motion/T2V descriptions, thinking controls and profile selection independent of Training. Clarify frame limits, lazy-frame inputs and optional Eclipse scene-analysis connections; describe music-node wiring without references to unshipped workflows.
- **Changed files**
  - .defaults/config/llm_few_shot_training_nsfw.json.example
  - .defaults/config/system_prompts.json.example
  - .defaults/config/system_prompts_nsfw.json.example
  - README.md
  - Readme/MiniMax_Music_3.md
  - Readme/Nova_Panels.md
  - Readme/Smart_LM_Loader_Guide.md
  - Readme/User_Data.md
  - Readme/Visual_Tour.md
  - Readme/YuE2.md
  - core/migration.py
  - core/sml/backend_llamacpp_docker.py
  - core/sml/backend_ollama_docker.py
  - core/sml/backend_sglang_docker.py
  - core/sml/backend_transformers.py
  - core/sml/backend_vllm_docker.py
  - core/sml/backend_vllm_native.py
  - core/sml/config_templates.py
  - core/sml/image_inputs.py
  - core/sml/panel_configuration.py
  - core/sml/scene_motion.py
  - core/sml/tasks.py
  - js/nova-panel-form.js
  - js/nova-panel.css
  - js/nova-panel-state.js
  - py/RvLoader_SmartLMLoader.py
  - pyproject.toml

## 2026-11-03

### Version: 1.1.11

- **Feat**
  - Add native Qwen3.5 dense GGUF text and vision loading, including compatible Qwen3.8-27B checkpoints and optional MTP weights. Restore hybrid-attention tensor layouts, retain quantized projections, and validate the model's matching vision projector.
  - Add a Thinking button to the LM Loader Mode bar for native ComfyUI encoders, defaulting to off. Save the choice per node, apply it throughout task chains, and default older workflows to off.
- **Fix**
  - Allow native encoders to use supported thinking templates and keep reasoning out of the caption. Bound native Qwen description thinking to 128 tokens or one quarter of the total, then continue the caption in the same pass; preserve natural completion and isolate cached generation state. Give Ultra Detailed fuller visual coverage and connected paragraphs while retaining example content.
  - Adjust LM Loader and native LM Settings token limits in steps of 128 while retaining manually entered integer limits. Refresh the bundled NSFW few-shot example from the current user configuration.
  - Simplify caption prompts and pose examples while retaining action, visible people and appearance, overall pose, camera angle, background, and lighting. Preserve full NSFW scene coverage, anatomical terms, size descriptions, and explicit actions across modes; remove rigid sentence limits and omit unshown details silently. Upgrade recognized bundled entries while preserving customizations.
  - Select NSFW system prompts together with NSFW training examples, including native ComfyUI encoders. Keep the added NSFW instructions in the separate prompt file, use Standard prompts when Training is off, and preserve custom system-prompt overrides.
  - Reserve generation workspace when loading native Qwen3.5 GGUF encoders so ComfyUI can leave more weights in system RAM, including for larger requests on cached models. Detach mapped offloaded weights directly on CPU to avoid temporary VRAM spikes.
- **Changed files**
  - .defaults/config/llm_few_shot_training.json.example
  - .defaults/config/llm_few_shot_training_nsfw.json.example
  - .defaults/config/system_prompts.json.example
  - .defaults/config/system_prompts_nsfw.json.example
  - Readme/Smart_LM_Loader_Guide.md
  - core/migration.py
  - core/sml/backend_comfyui.py
  - core/sml/config_templates.py
  - core/sml/gguf_memory.py
  - core/sml/gguf_metadata.py
  - core/sml/gguf_qwen35.py
  - core/sml/gguf_text_encoders.py
  - core/sml/native_thinking.py
  - core/sml/settings_controls.py
  - core/sml/tasks.py
  - extern/gguf/README.md
  - extern/gguf/dequant.py
  - extern/gguf/loader.py
  - extern/gguf/mappings.py
  - extern/gguf/ops.py
  - extern/gguf/patcher.py
  - extern/gguf/qwen35.py
  - js/nova-native-settings.js
  - js/nova-panel-form.js
  - py/RvLoader_SmartLMLoader.py
  - pyproject.toml

## 2026-11-01

### Version: 1.1.10

- **Fix**
  - Give Simple, Detailed and Ultra Detailed Description four example pairs each in both Standard and NSFW profiles, with clearer hand/leg placement, support points, actions and interactions.
  - Align examples and task instructions with their intended detail levels, distinguish visible anatomy from covered or cropped areas, and discourage invented contact or motion. Upgrade recognized bundled entries while preserving customized prompts and examples.
- **Docs**
  - Clarify description lengths, example counts and the role of text demonstrations in the loader guide.
- **Changed files**
  - .defaults/config/llm_few_shot_training.json.example
  - .defaults/config/llm_few_shot_training_nsfw.json.example
  - .defaults/config/system_prompts.json.example
  - Readme/Smart_LM_Loader_Guide.md
  - core/migration.py
  - pyproject.toml

## 2026-11-01

### Version: 1.1.9

- **Feat**
  - Add native Smart LM Settings and Smart Detection Settings helpers with optional `pipe_opt` inputs on their processing panels. Promote model, task, prompts and main controls onto subgraphs, or pass settings through a subgraph input. Preserve local fallback values, advanced controls and existing seed behavior.
  - Refresh native model-dependent choices independently for each subgraph instance, including nested hosts and workflow reloads. Validate settings payloads and track native text-encoder file changes for cached execution.
  - Carry LM user/system prompts through `pipe_opt` and add both native prompt widgets and a compact Smart LM Prompt Pipe for external strings. Use it alone or after native settings; preserve prompt lists, explicit empty values and Direct Chat behavior.
  - Carry Detection queries through `pipe_opt` and add Smart Detection Prompt Pipe for external query/class-filter strings, usable alone or after native settings. Preserve empty overrides, first-string behavior and promoted subgraph query widgets.
- **Breaking**
  - Remove the LM Loader's individual prompt sockets and Detection's direct `user_input` socket. Saved UI workflows automatically move their connections through the matching Prompt Pipe, including subgraph links; API templates should supply prompts through the pipe.
- **Docs**
  - Document native settings, prompt composition, subgraph wiring, override priority and saved-workflow migration.
- **Changed files**
  - README.md
  - Readme/Nova_Panels.md
  - Readme/Smart_LM_Loader_Guide.md
  - Readme/Smart_Detection_Guide.md
  - Readme/Visual_Tour.md
  - core/sml/panel_configuration.py
  - core/sml/settings_controls.py
  - core/sml/settings_pipe.py
  - js/nova-prompt-migration.js
  - js/nova-native-settings.js
  - js/nova-panel-form.js
  - js/nova-panel.js
  - js/smartllm-metadata.js
  - py/RvConfig_SmartSettings.py
  - py/RvLoader_SmartDetection.py
  - py/RvLoader_SmartLMLoader.py
  - py/RvPipe_SmartLMPrompt.py
  - py/RvPipe_SmartDetectionPrompt.py
  - pyproject.toml

## 2026-10-29

### Version: 1.1.8

- **Fix**
  - Restore middle-button canvas panning and wheel zoom over node panels in classic ComfyUI and Nodes 2.0. Preserve focused text scrolling, normal control clicks and workflow reloads.
- **Changed files**
  - js/nova-configuration-panel.js
  - js/nova-panel-wheel.js
  - pyproject.toml

## 2026-10-27

### Version: 1.1.7

- **Fix**
  - Support older YOLO segmentation checkpoints while retaining restricted loading and artifact verification. Report classification checkpoints clearly when they cannot supply detection boxes or masks.
  - Use Default/CPU device choices with compatible restoration of older workflows. Follow ComfyUI's selected device for local model placement, memory reservations, text-encoder offloading and device-aware caches. Keep CPU generation on CPU, use the selected CUDA index for YOLO/WD14 and native GGUF text loading, and report unsupported CPU backends instead of silently accelerating them.
  - Include native ComfyUI vision encoders in Smart Detection, reusing Smart LM Loader's capability discovery and loading for supported safetensors and GGUF files. Keep text-only encoders out of detection, expose the supported grounding tasks, and preserve per-image boxes, masks, SEGS, cleanup and checkpoint-aware caching.
  - Ground comma-, semicolon- or newline-separated detection targets independently with the same loaded model. Combine their results and retain overlapping targets instead of allowing one query to suppress another. Preserve the query as the category label when native encoders return boxes without labels, so category prompts remain usable downstream.
  - Keep confidence, NMS and maximum region area in the Detection panel without override sockets. Remove their retired workflow links safely, preserve image/prompt/seed connections and accept older API prompts using saved panel values.
- **Docs**
  - Explain native vision encoder selection and detection settings in the Smart Detection guide.
- **Changed files**
  - Readme/Smart_LM_Loader_Guide.md
  - core/sml/device.py
  - core/sml/backend_comfyui.py
  - core/sml/backend_transformers.py
  - core/sml/backend_wd14.py
  - core/sml/backend_yolo.py
  - core/sml/florence2_wrapper.py
  - core/sml/loader_base.py
  - core/sml/vlm_loader.py
  - js/nova-panel-state.js
  - Readme/Smart_Detection_Guide.md
  - core/sml/grounding_queries.py
  - core/sml/model_registry.py
  - core/sml/panel_configuration.py
  - core/sml/vlm_detection.py
  - js/nova-panel-controls.js
  - js/nova-panel-form.js
  - js/nova-panel.js
  - py/RvLoader_SmartDetection.py
  - pyproject.toml

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
