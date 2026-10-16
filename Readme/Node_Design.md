# Node designs

Choose **Settings → SmartLLM Nova → Appearance → Node design**. The setting belongs
only to this repository and works in classic canvas and Nodes 2.0. Neutral is
the default for new installations; existing saved theme choices are preserved.

| Bundled theme | Background | Controls |
| --- | --- | --- |
| Jade | RGB 26,26,26 (`#1a1a1a`) | Jade accents, purple selection glow |
| Neutral (default) | RGB 58,58,58 (`#3a3a3a`) | Gray controls, no glow |
| Ocean | RGB 26,26,26 (`#1a1a1a`) | Blue accents and glow |
| Amethyst | RGB 26,26,26 (`#1a1a1a`) | Purple accents and glow |
| Amber | RGB 26,26,26 (`#1a1a1a`) | Warm yellow-gold (`#e8b978`) accents and glow, matching the Eclipse README tour |

The palettes are JSON files in `themes/`. They style redesigned panels, searchable
menus and both Registry Manager tabs, including selected tabs, form controls,
Docker cards and the outer dialog glow. Changing or reloading the palette updates
an open manager while keeping its form values.
All nodes owned by this pack use the selected node background; nodes with native
ComfyUI controls keep those native controls. Other packs are unaffected.

## Existing workflows and colors

Selecting or reloading a design replaces **all saved/custom node backgrounds**
in this pack, including nodes inside subgraphs. Loading a workflow also applies
the current design. Manual background changes are temporary until one of those
actions happens. Custom title colors are preserved; automatic titles follow the
theme. Node values, connections, collapsed sections and outputs are unchanged.

The chosen theme ID is saved as `node_design` in this repo's private `config.json`.
It takes precedence over an older browser setting at startup and applies across
workflows. Changes are event driven: no file watcher, recurring canvas scan or
per-node theme request.

## Theme files and updates

Bundled palettes are shipped as `.defaults/themes/*.json.example`. Startup copies
each default into the user-owned `themes/` folder once. Existing files are kept
exactly as they are, including custom edits and malformed files. Updates do not
replace a theme or restore a deleted one, even if the bundled default changes.
A newly bundled filename can be installed once without restoring older deletions.

The extraction record is `.theme-defaults.json` in this repository. Theme reloads
only read files. To restore a removed palette, copy its `.json.example` from
`.defaults/themes/` into `themes/` and remove the `.example` suffix yourself.

## Create or edit a theme

1. Copy `themes/jade.json` (or another bundled theme) to
   **`themes/custom/my-theme.json`** in this repo.
2. Change `label`, the colors under `node`, and the values under `tokens`.
   Keep the complete structure and `version: 1`.
3. Click **Reload theme files → Reload themes** in this pack's Appearance
   settings. Select your theme under **Node design**.
4. After editing the selected JSON file again, use **Reload themes** to apply it.

The filename without `.json` is the saved ID. Use lowercase letters, digits,
hyphens and underscores, with at most 64 characters. `label` is the displayed
name. You can edit a theme directly in `themes/` or override it in `themes/custom/`
with its original filename; for example, `custom/jade.json` overrides
`themes/jade.json`. The whole `themes/` folder is ignored by Git. Reads never
rewrite theme files.

The same complete JSON format works in Eclipse Nova, SML2 and SmartLLM Nova.
Copy a theme into each repo where you want it available and select it separately.
There are no shared settings or runtime imports between the packs.

## Editable fields

| Field | Purpose |
| --- | --- |
| `node.background` | Node body and panel/dialog surface |
| `node.title` | Default automatic title color |
| `node.routing_background` | Eclipse Nova pipe/routing body color; black in bundled themes |
| `node.category_titles` | Eclipse Nova's type-based title colors; unused by the other packs |
| `tokens.accent` | Main control accent |
| `tokens.control-*` | Button backgrounds and borders |
| `tokens.input-*`, `tokens.select-*` | Text fields and selectors |
| `tokens.bar-*`, `tokens.pill-*`, `tokens.selected-*` | Mode bars and selected options |
| `tokens.menu-*`, `tokens.dialog-*`, `tokens.backdrop` | Menus, dialogs and shadows |
| `tokens.text`, `muted`, `label`, `error`, `warning`, `success`, `danger` | Text and status colors |
| `tokens.row-bg`, `divider`, `option-bg`, `ratio-bg`, `muted-accent` | Rows, separators and secondary accents |
| `tokens.color-scheme` | `dark` or `light` for native controls inside panels |

Node colors require six-digit hex values, such as `#243544`. Color tokens accept
hex colors with optional alpha (`#00ffcc55`) or `transparent`. `pill-radius` accepts
`0` or a pixel value. Shadow tokens accept `none` or up to four comma-separated
shadows, each with three nonnegative pixel lengths and a hex color, such as
`0 0 10px #975bff99`. Arbitrary CSS and URLs are not accepted.

Malformed, oversized or unsupported files are skipped with a visible message;
valid themes remain available. If the selected file is missing or invalid, Neutral
is shown temporarily without overwriting the saved selection. Restore the file
and reload, or explicitly choose another theme. A malformed custom override
leaves the bundled version available.
