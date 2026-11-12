# Task library

Click **Tasks** below **Model** in Smart LM Loader or Smart Detection to open a
resizable panel. Tasks and search are on the left; editing fields are on the right.
The node shows a compact summary of its selection. Model selection, Registry,
refresh and quantization remain in Model.

## Select tasks

The loader keeps selected tasks at the top in execution order, including during
search. Available tasks have collapsible purpose categories: **Chat & Text**,
**Image Description**, **Video & Scenes**, **Tags**, **Stories & Music**, and
**OCR**. Each row shows whether images are required, optional or unused. Search
opens matching categories, and selected tasks remain visible. Check a task to select
it, then check compatible tasks to build a chain. Uncheck a selected task to remove
it. The numbered selection shows execution order;
arrows reorder it and × removes a selection. Only the first task receives original
images. Later stages receive the preceding text and run with their own instructions
and template. The loaded model is reused, and the last result uses the existing text
output. Up to 32 tasks can be selected. Existing workflows using Task 2–4 retain their
order. Structured scene, H3 and music outputs finish the chain; lyric sheets can feed
Music 3 or YuE2 conversion. Incompatible choices explain why they cannot be added.

Short Story and the song tasks can use optional images with a vision-language
model. Single images and ordinary Together requests compose directly. Grid and
VIDEO first collect visual notes across windows, then compose one complete result
using those notes and the original prompt. Existing song lyrics stay unchanged
during conversion. Choose **Story** behavior for custom story tasks to use this
same sequence handling; renaming or copying a task preserves its behavior.

Leave images, video or disk frames connected when switching to a text-only model.
It ignores that media and runs once per text request, regardless of Together/Grid
settings. Tasks configured for text only also ignore connected media on vision
models. IMAGE batches pass through unchanged; unused video/disk sources are not
decoded. Image-required tasks still need a vision model. Optional images remain
available as context when a vision model is selected.

Detection uses a single selection. Phrase grounding still accepts `face, eye, mouth`
and evaluates each target separately. Other tasks retain their full query, including
commas. YOLO and WD14 continue to use their native controls.

## System instructions and user guidance

The task file supplies system instructions. The Prompt field supplies additional
guidance or source text. A task's optional user template inserts that text at
`{prompt}`; it is applied once for text and vision requests. Each task has one instruction
set. There are no Training or NSFW switches, profile overrides or example conversations.
Older workflows can still load; their retired switch values are ignored.

General vision tasks use short instructions. Simple Description gives a compact paragraph; Detailed Description develops
one paragraph; Ultra Detailed Description uses several paragraphs with smaller visible
details. Cinematic, analysis, tags and video-summary tasks retain their own formats.
Visual task instructions use direct anatomical wording when supported by the input,
without censorship or euphemisms. Unshown or uncertain details are omitted silently;
nudity is not inferred beneath clothing. Detail level and output format still come
from the selected task. These instructions guide the model; they cannot guarantee
that it recognizes every detail correctly.

Open **System instructions · preview** in Prompt to inspect the selected task.
Direct Chat also shows an editable **System prompt** field. Edits there apply only
to that node; clearing it uses the task file again. A nonblank system prompt from
the settings pipe takes priority, selects Direct Chat and disables the local chain.
A connected task replaces the local selection. Connected user guidance retains it.

The preview shows task instructions. Grid windows, scene IDs, song conversion
and corrective retries can add context when the request executes. Chat-capable
backends send system instructions before user guidance. Native ComfyUI encoders use
one system/user pair. Older LLaVA and llama-cpp-python vision handlers combine instructions
and image text in the user turn. These adaptations preserve both texts.

## Edit, create and remove

Use **Edit** beside a task to open its fields on the right without selecting it.
Instructions and the user template are shown first; expand **Task behavior**
for **Category**, input policy and output handler. Category controls where the
task appears; it does not change its image requirements or output format.
Changing the selection keeps an
unsaved editor draft. Switching the edited task or closing the panel asks before
discarding unsaved changes. **Save** updates its file. **Save as new** creates
a new stable ID; give the copy a different name. **New task** starts a reusable custom
task. Choose an output behavior that matches the requested output: free-form text, story,
scene observation, H3, lyric sheet, music JSON or a supported detection operation.
The behavior selects application parsing and validation; instructions cannot redefine
a parser. A disabled draft remains editable but cannot execute.

Edit the task's instructions to change its focus, wording or detail. Retain the required
output fields and syntax when using a structured handler. Edits over the editor's
16 KiB request limit can be split into smaller saves or made in the JSON file.
Conflicting saves retain the draft and ask you to reload; they never overwrite a newer
file silently. Refresh Tasks after editing files externally.

**Remove** archives a task. Saved workflows keep its ID and show it as unavailable.
Restore it under **Removed tasks**, or explicitly select a replacement. Renaming a task
keeps its ID and old display-name aliases, so existing selections still resolve.
Manually deleted files also stay deleted on restart; restore those from your backup
or their matching `.defaults/tasks/*.json.example` file.

## Florence

Florence uses supported native operation tokens, such as `<CAPTION>` or
`<CAPTION_TO_PHRASE_GROUNDING>`. Its editor shows the native operation and query template.
Only operations supporting text receive query text; other operations receive only the
token. The application owns token mappings and spatial parsers. PromptGen-only
operations require a compatible checkpoint. Changing prose does not add Florence
capabilities, and Florence does not support text task chains.

## Files and upgrades

Each task lives in `tasks/<stable-id>.json`. A version 2 definition contains `id`, `name`,
`description`, `group`, `nodes`, `aliases`, `input`, `handler`, `florence_operation`,
`instructions`, `template`, `internal` and `enabled`. Input policy
declares whether text/images are required, optional or unused, whether ordered
image sequences are accepted, and whether grounding queries split into terms.
The editor creates IDs; keep them stable when editing files manually. Internal song
conversion tasks remain separate files and are not offered as ordinary selections.

The `group` field controls the purpose category only; input policy and task behavior
remain independent. Older `vision`/`text` values are still accepted. Bundled categories
update with unedited bundled tasks; customized tasks keep their saved category and
all other fields across restarts.

The initial library comes directly from `.defaults/tasks/`. Nova ignores the old
`config/system_prompts*.json`, `config/llm_few_shot_training*.json` and
`config/task_profiles.json` files. They stay on disk but do not override bundled
instructions, add examples or create imported drafts. Edit or duplicate tasks in the
Tasks panel to customize the current library.

Existing edits made directly to task files remain supported. Version 1 task files
are backed up under `tasks/.legacy/v1/` before conversion; their own main instructions
and templates are retained without reading retired shared files. Alternate full
prompts/templates and examples remain in that backup rather than being combined
with the active instructions.

`tasks/.state.json` records shipped revisions and removals; `tasks/.trash/` holds removed
definitions. Each new bundled task is extracted once. Deleted files stay deleted,
including files removed manually. Updates replace a task only when its complete
definition still matches its recorded bundled revision. Editing any field preserves
the entire file; existing files without an extraction record are also adopted untouched.
Unchanged startups do not rewrite task files or the extraction record.
Each execution takes a stable catalog snapshot, and task changes invalidate cached
node results for later runs.
Back up the entire `tasks/` directory, including hidden
records. Malformed tasks remain untouched and report an error in the Tasks panel;
other valid tasks remain available after restarting. Repair the affected file before
selecting it. Invalid shared migration state still requires repair before startup.

The former English Wan duration/beat presets consolidate into **Wan 2.2**, with old names and IDs
retained as workflow aliases. Only untouched variants are archived under
`tasks/.legacy/duration-presets/`; customized variants and explicit deletions remain
in effect. Wan, LTX and H3 instructions leave duration to the downstream generator
and do not add seconds or timestamps to the video prompt.
