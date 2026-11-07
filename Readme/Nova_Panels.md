# SmartLLM Nova panels

The five panel nodes use one named JSON configuration widget, compact themed controls,
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
| Smart LM Loader | Model, quantization where supported, tasks and chaining, image handling, prompt, generation, sampling, hardware, WD14 tagging, seed | `pipe_opt` for settings and prompts, `seed_input`; optional `images` or lazy `frame_source` |
| Smart Detection | Model, quantization where supported, task/classes, confidence and NMS, mask/crop adjustments, sampling, hardware, seed | `pipe_opt` for settings and query text, `seed_input`; required `image` |
| Detection to Bboxes | Image/JSON detection, channel threshold, contour area, mask inversion/grow/blur, combined or selected masks | Required `image`, optional `data_opt` |
| Split YuE2 | Resizable song JSON field; unchanged style and lyrics outputs | Optional `song_json` |
| Split MiniMax Music 3 | Resizable song JSON field; unchanged caption and lyrics outputs | Optional `song_json` |

All names above end in `[SmartLLM Nova]`. Two native settings helpers and two
prompt-only IO helpers accompany these panels. Nova does not alias the original pack's IDs. See the
[visual tour](Visual_Tour.md) for the controls in use.

Connected non-None override values take priority over the corresponding local
setting. An empty prompt is a valid override. Smart LM Loader preserves prompt
lists and image batches, multi-task chaining, task templates, per-node Standard/NSFW profiles and optional examples,
WD14, quantization, attention, cleanup and model retention. Smart Detection
keeps its IMAGE/MASK/SEGS/JSON list outputs. No model acquisition, trust, device,
or backend policies are replaced by the panels.

Choose **Settings → SmartLLM Nova → Appearance → Node design** for Neutral (default), Jade, Ocean, Amethyst, Amber, their Neutral color variants or a [custom JSON theme](Node_Design.md). The setting belongs only to this repo; it also updates saved node backgrounds.

## Image handling and prompt modes

The loader's **Image handling** section sits below Model and has its own collapse,
pin and settings summary. **Together** switches between one request for an ordered
image selection and one request per image. **Longest side** limits input size;
`0` keeps automatic model limits. **Every Nth frame** samples Together selections
while retaining both endpoints. **Max frames** appears only for applicable legacy
configurations; explicit Together uses the supplied selection without that cap.

These controls remain available in Direct Chat and with Model collapsed. Text-only
requests ignore image handling. Vision models can also use images alongside text
for tasks such as refinement or tag conversion. Later multi-task stages consume
the previous text result. See the [loader guide](Smart_LM_Loader_Guide.md#image-sequences-and-scene-motion)
for scene tasks, bounded windows and model limits.

In the mode bar, **Standard/NSFW** selects system prompts independently of
**Training**; Training adds contextual examples without changing weights.
**Thinking** defaults off and appears for supported backend paths. The model's
template must support it, and reasoning shares **Max tokens** with the answer.

## Native settings and subgraphs

Add **Smart LM Settings** or **Smart Detection Settings** from
**🌒 SmartLLM Nova → Settings**. Connect its `settings` output to the matching
panel's optional `pipe_opt` input. These helpers use ordinary ComfyUI widgets:

| Helper | Controls |
| --- | --- |
| Smart LM Settings | Model, task, GGUF quantization, user prompt, context size, max tokens, system prompt |
| Smart Detection Settings | Model, task, GGUF quantization, query/classes, confidence, NMS IoU, maximum region area ratio, selected index |

Put the helper and processing panel inside a subgraph, then promote the helper's
widgets to edit them on the subgraph node. Promote model, task and quantization
together for model-dependent choices. Each subgraph instance keeps its own values.
You can also keep the helper outside and expose the panel's `pipe_opt` through
the subgraph boundary. Images still connect to the processing panel's image inputs.

Connected settings override the corresponding panel values. Advanced settings,
task chaining and seed controls remain on the panel. LM user/system prompts and
Detection's `user_input` query enter through `pipe_opt`.
Disconnecting `pipe_opt` restores the saved panel values. The panel
shows a connection note and keeps its local settings available for editing.

Native model lists follow the same category/name order as the panels, and task
choices use the same model-capability filters. Registry refreshes and workflow
reloads preserve saved values, including unavailable selections, without adding
incompatible tasks to the menu.

Controls stay visible when inactive, including when promoted to subgraphs.
Quantization is enabled for native GGUF and llama.cpp models and disabled for
other backends. WD14 and YOLO disable the task selector. On supported LM models,
nonblank system text or a connected system-prompt input selects **Direct Chat**
and disables the task selector; clear the text and disconnect the input to choose
a task again. Florence and WD14 retain their specialized behavior. Each subgraph
instance keeps its own enabled states, including through nested promotion and
workflow reloads.

For integrations, each pipe is one dictionary with `version: 1`, `kind: "loader"`
or `"detection"`, and a `parameters` mapping using the corresponding panel's field
names. Omitted fields retain panel values; `0`, `false` and empty text are explicit
values. Unknown fields, invalid ranges, mismatched kinds and multiple pipes are
rejected. Seed is managed separately and cannot be supplied in `parameters`.

### Prompt-only IO

Add **Smart LM Prompt Pipe** from **🌒 SmartLLM Nova → IO** to connect external
`user_prompt` and `system_prompt` strings without using native settings widgets.
Connect its `settings` output to **Smart LM Loader → pipe_opt**. With no incoming
settings pipe, model/task and all other controls continue to come from the panel.

To combine native settings with external prompts, connect **Smart LM Settings →
settings** to **Smart LM Prompt Pipe → pipe_opt**, then its output to the loader.
Each connected prompt overrides the corresponding incoming value. Unconnected
inputs preserve it. A connected empty user string clears the user prompt; an
empty system string restores the selected task's instructions. User prompt lists
are supported, with one shared system instruction for the whole list.

Nonblank system instructions select Direct Chat and disable task chaining for
supported models. Florence and WD14 retain their specialized handling. The
system text supplies the complete instructions, rather than extending a task's
template. Native Settings also has both prompt widgets for subgraph promotion.

For Detection, use **Smart Detection Prompt Pipe** from the same IO menu.
Connect query/class-filter text to its `user_input`, then its `settings` output
to **Smart Detection → pipe_opt**. It works alone or after **Smart Detection
Settings**, with connected text overriding the native query. Missing text keeps
the incoming query; an empty string clears it. Detection retains its existing
first-string behavior for prompt lists. Use commas, semicolons or newlines within
one grounding query to request multiple targets.

Saved workflows with the processing nodes' former prompt/query sockets are
rewired through the matching Prompt Pipe when loaded, preserving existing
settings, image and seed links. Promoted query widgets on settings subgraphs stay
available. Save the workflow to retain the updated wiring. API templates should
supply prompts through `pipe_opt` or the matching Prompt Pipe; browser rewiring applies to
saved UI workflows.

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
the matching native settings helper can override them through `pipe_opt`.

For supported language/vision models, nonempty connected system instructions
use Direct Chat. Florence and WD14 keep their specialized task handling.
Delete retains the existing confirmation and guarded deletion endpoint.

Nova reads its own config and registries. Startup materializes bundled defaults
without importing credentials or configuration from another installed pack.
