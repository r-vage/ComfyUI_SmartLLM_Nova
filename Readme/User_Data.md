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

The loader's **Standard / NSFW** button selects `config/llm_few_shot_training.json`
or `config/llm_few_shot_training_nsfw.json` for that node. Keep **Training** enabled
to include the examples. Files remain user-editable; a later execution reads
changes without changing another loader or the root configuration.

Back up the root configuration, Docker configuration, `config/`, `registry/`,
`themes/` and the extraction records before moving an installation. Root config
can contain credentials and uses private file permissions. Do not publish it.
Models and verification sidecars remain in the configured model directories.
