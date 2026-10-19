# SmartLLM Nova panels

The five nodes use one named JSON configuration widget, compact themed controls,
collapsible sections and searchable selectors in both classic ComfyUI and
Nodes 2.0. Their processing remains in the existing SmartLLM engines. The new
editors do not use native-widget visibility, chip bars or widget-performance
helpers. UI changes happen on editing, connections, registry events and restore;
there is no recurring canvas scan for these panels.

The design draws on DaSiWa and the Eclipse Nova panels. Each node keeps its
section state and resized text fields in the workflow. Open a section to edit
it; pin it to keep it open when opening another section. Collapsing controls
never disables their processing. The mode buttons control actual features.

| New editor | Controls | Connected overrides |
| --- | --- | --- |
| Smart LM Loader | Model, quantization where supported, tasks and chaining, prompt, generation, sampling, hardware, WD14 tagging, seed | `user_prompt`, `system_prompt`, `seed_input`; optional `images` |
| Smart Detection | Model, quantization where supported, task/classes, confidence and NMS, mask/crop adjustments, sampling, hardware, seed | `user_input`, `confidence`, `nms_iou_threshold`, `detection_filter`, `seed_input`; required `image` |
| Detection to Bboxes | Image/JSON detection, channel threshold, contour area, mask inversion/grow/blur, combined or selected masks | Required `image`, optional `data_opt` |
| Split YuE2 | Resizable song JSON field; unchanged style and lyrics outputs | Optional `song_json` |
| Split MiniMax Music 3 | Resizable song JSON field; unchanged caption and lyrics outputs | Optional `song_json` |

All names above end in `[SmartLLM Nova]`. These five panels are the registered
interfaces. Nova does not alias the original pack's IDs. See the
[visual tour](Visual_Tour.md) for the controls in use.

Connected non-None override values take priority over the corresponding local
setting. An empty prompt is a valid override. Smart LM Loader preserves prompt
lists and image batches, multi-task chaining, task templates, per-node Standard/NSFW training examples,
WD14, quantization, attention, cleanup and model retention. Smart Detection
keeps its IMAGE/MASK/SEGS/JSON list outputs. No model acquisition, trust, device,
or backend policies are replaced by the panels.

Choose **Settings → SmartLLM Nova → Appearance → Node design** for Neutral (default), Jade, Ocean, Amethyst, Amber, their Neutral color variants or a [custom JSON theme](Node_Design.md). The setting belongs only to this repo; it also updates saved node backgrounds.

## Seeds

Use the exact 64-bit seed field, small −/+ controls, random-each-run, last-seed
and fixed-random buttons. Enter `-1`, `-2` or `-3` for random, increment or
decrement policies. Browser queues resolve local random seeds before execution
and record the actual seed in saved workflow metadata. Only nodes included in a
partial queue advance. Rejected submissions do not advance the local seed.

**Local seed** ignores `seed_input` while enabled. With a connection, turn it off
to use the upstream seed. Without a connection the local seed is used. API
submissions can supply a fixed seed in the configuration or use `seed_input`;
API clients control their own increment/decrement progression between requests.

## Models and tasks

The mode bar sits at the top. Below it, the collapsible **Model** section contains
the model selector, quantization when supported, task selection (including chained
tasks), then Registry, Delete and Refresh controls. WD14 and YOLO omit tasks.

Click Model to search a freshly fetched registry list. Refresh reloads the
selected model's metadata; Registry opens Nova's own manager. A deliberate model
change selects compatible task/quantization defaults. Reloading a workflow
preserves its selections, including unavailable entries, instead of silently
substituting another model. Backend-specific sections follow the registry model.
Model lists retain the original grouped order with names sorted within each group:
Vision, Text, then WD14 for the loader; VLM, then YOLO for detection. Search hides
groups with no matching models, and group headings cannot be selected.
Model, quantization and task selections are stored in the panel configuration;
there are no model or task override sockets.

For supported language/vision models, nonempty connected system instructions
use Direct Chat. Florence and WD14 keep their specialized task handling.
Delete retains the existing confirmation and guarded deletion endpoint.

Nova reads its own config and registries. Startup materializes bundled defaults
without importing credentials or configuration from another installed pack.
