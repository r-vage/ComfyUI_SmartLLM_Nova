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
