# SmartLLM Nova user data

Nova reads its own `config.json`, `docker_config.json`, `config/` and `registry/`
folders. It does not import settings, credentials or registries from another pack.
Use **Settings → SmartLLM Nova** and **Registry…** to configure this installation.
The original SmartLLM and Eclipse packs can remain installed independently.

Bundled JSON defaults live in `.defaults/`. Startup extracts new files and records
them in its local manifest. Deleting an extracted registry or prompt file does
not cause it to be restored on every restart. Missing root configuration can be
recreated when the config service needs it. Updates add missing defaults and can
refresh recognized, unedited bundled prompt entries while preserving local edits.

Themes use their own once-only record, `.theme-defaults.json`. The bundled
`.defaults/themes/` palettes are extracted into `themes/`; your edits and deletions
remain yours. Put custom themes in `themes/custom/`, then use **Reload themes** in
Appearance. See [Node design](Node_Design.md).

Each task lives in `tasks/<stable-id>.json`, with its input policy, system instructions,
user template and output behavior. Use **Tasks → Edit** to edit, duplicate or remove it.
There is one instruction set per task, without profiles or example conversations.
An execution keeps a consistent snapshot; later executions read saved changes.

The initial task library uses the current `.defaults/tasks/` definitions. Old
`config/system_prompts*.json`, `config/llm_few_shot_training*.json` and
`config/task_profiles.json` files are ignored and left on disk. Their custom entries
are not imported as drafts. Customize tasks directly through the Tasks panel or their
individual files. Existing task-file edits remain yours; version 1 task files are
backed up under `tasks/.legacy/v1/` before upgrading their own fields to version 2,
without importing shared prompt/profile files. Task IDs preserve saved selections through renames.
Removed tasks stay removed; use **Removed tasks** to restore tasks deleted in the editor.
`tasks/.state.json` tracks bundled revisions and removals. Updates replace only unedited
bundled task files. Editing any task field preserves the whole file, including fields
that still contain older bundled values. New bundled tasks install once; unchanged
startups do not rewrite task files or their extraction record. Keep the hidden record
when backing up or moving tasks so manually deleted files stay deleted.
See [Task library](Task_Library.md).

Back up the root configuration, Docker configuration, `config/`, `registry/`,
`themes/`, `tasks/` (including its hidden records) and the extraction records before moving an installation. Root config
can contain credentials and uses private file permissions. Do not publish it.
Models and verification sidecars remain in the configured model directories.
