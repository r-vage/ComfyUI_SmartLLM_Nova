/** Jade/Director layout inspired by DaSiWa; DOM controls own their visibility. */
import { app, api } from './comfy/index.js';
import { element, button, SearchMenu, groupedModelOptions } from './nova-panel-controls.js';
import { TextFieldSizes } from './nova-panel-text-layout.js';
import { seedControls } from './nova-panel-seed-controls.js';
import { fetchModelEntry, fetchModelList, fetchDetectionModelList, expireMetadataCache } from './smartllm-metadata.js';
import { TaskLibrary, taskRequest } from './nova-task-library.js';
import { onSmartLLMRegistryChanged } from './smartllm-registry-events.js';

const LABELS = { image_stride: 'Every Nth frame', image_handling: 'Image handling', max_image_side: 'Longest side', memory_cleanup: 'Cleanup', cleanup: 'Cleanup', keep_model_loaded: 'Keep loaded', multi_task_mode: 'Multi-task', enable_thinking: 'Thinking', scene_json: 'Scene JSON', use_advanced: 'Sampling', enable_preview_boxes: 'Preview boxes', get_mask_from_image: 'From image', combine_masks: 'Combine', invert: 'Invert', user_prompt: 'Prompt', user_input: 'Prompt / classes', model_name: 'Model', song_json: 'Song JSON', nms_iou_threshold: 'NMS IoU', detection_filter: 'Max area ratio', replace_underscore: 'Remove underscores', char_threshold: 'Character threshold' };
const clean = values => (values ?? []).map(value => typeof value === 'string' ? value : value.display_name).filter(value => value && !value.startsWith('__SEP__') && !value.startsWith('─'));
Object.assign(LABELS, { image_grid: 'Grid', grid_frames: 'Frames per grid', grid_side: 'Grid longest side', grid_tile_side: 'Tile longest side', video_sample_fps: 'Sample FPS', video_cut_threshold: 'Cut threshold (0 = off)' });
const modeLabel = (key, value) => key === 'image_grid' ? (value ? 'Grid' : 'Frames') : LABELS[key];

export class PanelForm {
    constructor(controller) {
        this.controller = controller; this.menu = new SearchMenu(); this.textSizes = new TextFieldSizes(controller);
        this.sequence = 0; this.taskLibrary = new TaskLibrary(this);
        this.unsubscribe = onSmartLLMRegistryChanged(() => this.refresh(false));
        // One coalesced metadata request per selected model, no graph polling.
        queueMicrotask(() => this.refresh(false));
    }
    get state() { return this.controller.state; }
    get params() { return this.state.config.parameters; }
    get spec() { return this.state.spec; }
    get modelKey() { return this.spec.kind === 'loader' ? 'model' : this.spec.kind === 'detection' ? 'model_name' : null; }
    connected(key) { return this.controller.node.inputs?.some(input => input.name === key && input.link != null); }
    edit(fn, render = true) { this.controller.edit(fn, render); }
    async refresh(userDriven = false) {
        if (!this.modelKey || this.controller.disposed || this.controller.invalid) return;
        const model = this.params[this.modelKey], sequence = ++this.sequence;
        const entry = await fetchModelEntry(model);
        let catalog;
        try { catalog = await taskRequest(`?kind=${this.spec.kind}&model=${encodeURIComponent(model)}`); }
        catch (error) { if (sequence === this.sequence) this.status = error.message; }
        const tasks = catalog?.tasks.filter(task => !task.issue).map(task => task.name) ?? [];
        if (sequence !== this.sequence || this.controller.disposed || this.controller.invalid || this.params[this.modelKey] !== model) return;
        this.entry = entry; this.model = model; this.tasks = clean(tasks); this.catalog = catalog;
        // Only a deliberate model change chooses defaults; opening a workflow
        // never changes a saved selection, including unavailable registry entries.
        if (userDriven) this.edit(state => {
            const p = state.config.parameters;
            if (entry?.quantizations?.length && !entry.quantizations.includes(p.quantization)) p.quantization = entry.quantizations[0];
        });
        else this.render();
    }
    render() {
        const { root } = this.controller;
        const focused = root.contains(document.activeElement) ? document.activeElement.dataset.field : null;
        this.menu.close(); this.textSizes.disconnect();
        const content = element('div', 'sln-panel-content'), kind = this.spec.kind;
        if (this.connected('pipe_opt')) content.append(element('div', 'sln-panel-note',
            kind === 'loader'
                ? 'Connected settings and prompts override matching values below. Missing settings use these local values. Use Smart LM Prompt Pipe to connect external prompt text.'
                : 'Connected settings and query text override matching values below. Missing settings use these local values. Use Smart Detection Prompt Pipe to connect external query text.'));
        const entry = this.model === this.params[this.modelKey] ? this.entry : null;
        const backend = entry?.backend, family = entry?.family, wd = backend === 'wd14', yolo = backend === 'yolo', florence = family === 'Florence';
        const supportsThinkingControl = !florence && ['comfyui', 'ollama', 'transformers', 'vllm', 'vllm_native', 'sglang', 'llamacpp'].includes(backend);
        if (kind === 'loader') this.bar(content, ['memory_cleanup', 'keep_model_loaded', ...(!wd ? ['use_advanced', ...(supportsThinkingControl && this.spec.fields.enable_thinking ? ['enable_thinking'] : []), ...(this.spec.fields.scene_json ? ['scene_json'] : [])] : []), ...(this.spec.fields.image_grid && ((!wd && !florence && entry?.has_vision !== false) || this.params.image_grid) ? ['image_grid'] : [])], true);
        else if (kind === 'detection') this.bar(content, ['cleanup', 'keep_model_loaded', 'enable_preview_boxes'], true);
        if (this.modelKey) {
            this.section(content, 'model', 'Model', [], body => {
                this.field(body, this.modelKey);
                if (['gguf', 'llamacpp'].includes(backend)) this.field(body, 'quantization', entry.quantizations);
                this.modelActions(body);
            });
        }
        if (this.modelKey && !wd && !yolo) this.taskLibrary.render(content);
        if (kind === 'loader') {
            const legacyFrameLimit = !wd && this.params.image_handling === 'legacy' && entry?.has_vision
                && (backend === 'comfyui' || !['Florence', 'Mistral', 'LLaVA', 'VLM'].includes(family));
            const grid = this.params.image_grid && !wd && !florence;
            this.section(content, 'images', 'Image handling', grid
                ? ['grid_frames', ['grid_side', 'grid_tile_side'], 'image_stride']
                : ['image_handling', 'max_image_side', 'image_stride', ...(legacyFrameLimit ? ['frame_count'] : [])], body => {
                if (grid) body.append(element('div', 'sln-panel-note', 'Grid analyzes frames together. Supply one scene as images or disk frames; long scenes use overlapping windows. Switching back to Frames restores your Together setting.'));
            });
            if (!wd && !florence && this.spec.fields.video_sample_fps) this.section(content, 'video', 'Video input', ['video_sample_fps', 'video_cut_threshold'], body => {
                body.append(element('div', 'sln-panel-note', 'Connect VIDEO directly to image / video. It is sampled in chronological shot windows in either mode, then combined into one result. Set Cut threshold to 0 for continuous camera movement. Audio is not analyzed; VIDEO returns no image passthrough.'));
            });
            if (wd) this.section(content, 'tagging', 'Tagging', ['threshold', 'char_threshold', 'replace_underscore']);
            else {
                this.section(content, 'prompt', 'Prompt', [], body => { this.taskLibrary.prompt(body); this.field(body, 'user_prompt'); });
                this.section(content, 'generation', 'Generation', [backend === 'comfyui' ? 'max_tokens' : ['context_size', 'max_tokens']]);
                if (this.params.use_advanced) this.sampling(content, florence, backend, entry?.has_vision === false);
            }
            this.section(content, 'hardware', 'Hardware', ['device', ...(backend === 'transformers' ? ['attention_mode', ...(entry?.has_vision === false ? [] : ['use_torch_compile'])] : [])]);
        } else if (kind === 'detection') {
            this.section(content, 'detection', 'Detection', ['user_input', ['confidence', 'nms_iou_threshold'], ['detection_filter', 'select_index']], body => { if (!yolo) this.taskLibrary.prompt(body); });
            this.section(content, 'adjustments', 'Adjustments', [['drop_size', 'crop_factor'], 'dilation', ...(florence ? ['convert_to_bboxes'] : [])]);
            if (!yolo) this.sampling(content, florence, backend, false);
            this.section(content, 'hardware', 'Hardware', ['device', ...(!yolo && backend !== 'comfyui' ? ['use_torch_compile'] : [])]);
        } else if (kind === 'bboxes') {
            this.bar(content, ['get_mask_from_image', 'invert', 'combine_masks']);
            if (this.params.get_mask_from_image) this.section(content, 'detection', 'Image detection', ['detect_color', ['threshold', 'min_area']]);
            this.section(content, 'mask', 'Mask', [['grow', 'blur'], ...(!this.params.combine_masks ? ['indices'] : [])]);
        } else {
            this.field(content, 'song_json');
            content.append(element('div', 'sln-panel-note', 'Connect song_json to override the text. Outputs keep the native song format.'));
        }
        if (this.spec.seed) this.section(content, 'seed', 'Seed', [], body => {
            if (this.state.config.seed.source === 'local' || !this.connected('seed_input')) seedControls(body, this.controller);
            else body.append(element('div', 'sln-panel-note', 'Using the connected seed.'));
            if (this.state.editor.lastSeed != null) body.append(element('div', 'sln-panel-note', `Last run: ${this.state.editor.lastSeed}`));
        });
        if (kind === 'loader' && this.connected('pipe_opt')) content.append(element('div', 'sln-panel-note', 'Nonblank system instructions in the pipe use Direct Chat for supported models.'));
        if (this.status) content.append(element('div', 'sln-panel-note', this.status));
        root.replaceChildren(content);
        this.taskLibrary.refreshPicker();
        if (focused) [...root.querySelectorAll('[data-field]')].find(el => el.dataset.field === focused)?.focus({ preventScroll: true });
        this.controller.measure();
    }
    modelActions(parent) {
        const actions = element('div', 'sln-panel-tools');
        const refresh = button('↻', 'Refresh registry lists', () => { expireMetadataCache(); this.refresh(false); });
        const registry = button('Registry…', 'Open model and task registries', () => app.extensionManager.command.execute('SmartLLMNova.LMRegistry.Open'));
        const remove = button('Delete…', 'Delete the selected model files', async () => {
            const name = this.params[this.modelKey];
            if (!confirm(`Delete downloaded files for ${name}?`)) return;
            try {
                const response = await api.fetchApi('/smartllm-nova/model/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ display_name: name }) });
                const data = await response.json(); if (!response.ok || data.error) throw Error(data.error ?? 'Delete failed');
                this.status = data.message ?? 'Model files deleted.'; expireMetadataCache(); await this.refresh(false);
            } catch (error) { this.status = error.message; this.render(); }
        });
        actions.append(registry);
        if (this.model === this.params[this.modelKey] && this.entry?.can_delete !== false) actions.append(remove);
        actions.append(refresh); parent.append(actions);
        if (this.entry?.backend === 'comfyui') {
            parent.append(element('div', 'sln-panel-note', 'Local ComfyUI text encoder. File precision and memory management follow ComfyUI. Add files to models/text_encoders, then refresh the model list.'));
            if (this.entry.projector_note) parent.append(element('div', 'sln-panel-note', this.entry.projector_note));
        }
    }
    sampling(parent, florence, backend, textOnly) {
        const rows = [ ...(!florence ? [['temperature', 'top_p', 'top_k'], 'repetition_penalty'] : []), ...(backend === 'comfyui' ? ['do_sample'] : !textOnly ? [['num_beams', 'do_sample']] : []) ];
        if (this.spec.kind === 'loader' && !florence) {
            if (['comfyui', 'gguf', 'llamacpp', 'ollama', 'vllm', 'sglang'].includes(backend)) rows.push('min_p');
            rows.push('stop_sequences');
            if (['gguf', 'llamacpp', 'ollama'].includes(backend)) rows.push('mirostat', ['mirostat_eta', 'mirostat_tau'], 'repeat_last_n');
        }
        this.section(parent, 'sampling', 'Sampling', rows);
    }
    bar(parent, keys, seed = false) {
        const collapsed = this.state.editor.barCollapsed === true, bar = element('div', 'sln-panel-bar');
        bar.dataset.collapsed = String(collapsed); bar.setAttribute('aria-label', 'Mode');
        const toggle = button('', 'Mode', () => this.edit(state => { state.editor.barCollapsed = !collapsed; }));
        toggle.className = 'sln-panel-disclosure'; toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.append(element('span', '', `${collapsed ? '▸' : '▾'} Mode:`));
        if (collapsed) {
            const summary = keys.filter(key => key === 'image_grid' || this.params[key])
                .map(key => modeLabel(key, this.params[key]));
            toggle.append(element('span', 'sln-panel-summary', summary.join(' · ') || 'Off'));
        }
        bar.append(toggle);
        if (!collapsed) {
            for (const key of keys) {
                const label = modeLabel(key, this.params[key]);
                const control = button(label, this.spec.fields[key].tip || label, () => this.edit(state => { state.config.parameters[key] = !state.config.parameters[key]; }));
                control.dataset.field = key; control.setAttribute('aria-pressed', String(this.params[key])); bar.append(control);
            }
            if (seed) {
                const local = this.state.config.seed.source === 'local' || !this.connected('seed_input');
                const control = button('Local seed', 'Use the local seed instead of seed_input', () => this.edit(state => { state.config.seed.source = local ? 'input' : 'local'; }));
                control.dataset.field = 'seed.source'; control.setAttribute('aria-pressed', String(local)); control.disabled = !this.connected('seed_input'); bar.append(control);
            }
        }
        parent.append(bar);
    }
    section(parent, key, label, rows, extra) {
        const open = this.state.sectionOpen(key), section = element('section', 'sln-panel-section'), header = element('div', 'sln-panel-section-header');
        section.setAttribute('aria-label', label);
        const toggle = button('', label, () => this.edit(state => state.toggleSection(key)));
        toggle.className = 'sln-panel-section-toggle'; toggle.dataset.field = `section.${key}`; toggle.setAttribute('aria-expanded', String(open));
        const summary = element('span', 'sln-panel-summary', this.summary(key)); summary.dataset.summary = key;
        toggle.append(element('strong', '', `${open ? '▾' : '▸'} ${label}`), summary);
        const pin = button('⌖', `Keep ${label} open`, () => this.edit(state => {
            (state.editor.pins ??= {})[key] = !state.editor.pins?.[key];
            if (state.editor.pins[key]) (state.editor.sections ??= {})[key] = true;
        }));
        pin.setAttribute('aria-pressed', String(!!this.state.editor.pins?.[key])); header.append(toggle, pin); section.append(header);
        if (open) {
            const body = element('div', 'sln-panel-section-body');
            for (const row of rows) {
                if (Array.isArray(row)) { const grid = element('div', 'sln-panel-grid'); row.forEach(k => this.field(grid, k)); body.append(grid); }
                else this.field(body, row);
            }
            extra?.(body); section.append(body);
        }
        parent.append(section);
    }
    field(parent, key, choices) {
        const spec = this.spec.fields[key]; if (!spec) return;
        const label = LABELS[key] ?? spec.label, value = this.params[key];
        if (spec.type === 'COMBO') {
            let options = choices ?? spec.options;
            if (key === 'task' && this.tasks?.length) options = this.tasks;
            if (/^task_[234]$/.test(key) && this.textTasks?.length) options = ['None', ...this.textTasks];
            const change = next => { this.edit(state => { state.config.parameters[key] = next; }); if (key === this.modelKey) this.refresh(true); };
            options = key === this.modelKey ? this.modelOptions ?? groupedModelOptions(options, this.spec.kind) : clean(options);
            if (key === 'image_handling') {
                const together = value === 'frame_sequence', legacy = value === 'legacy';
                const wrap = element('div', 'sln-panel-field sln-panel-check');
                const tip = 'On: analyze the selected images together. Off: describe each image one by one. Independent of task and Direct Chat.';
                const input = button(`Together: ${legacy ? 'Legacy' : together ? 'On' : 'Off'}`,
                    legacy ? `Saved workflow behavior is retained. Click to enable Together. ${tip}` : tip,
                    () => change(together ? 'individual_images' : 'frame_sequence'));
                input.className = 'sln-panel-toggle'; input.dataset.field = key;
                input.setAttribute('aria-label', 'Together');
                input.setAttribute('aria-pressed', legacy ? 'mixed' : String(together));
                wrap.append(element('span', '', label), input); parent.append(wrap); return;
            }
            const input = this.menu.field(parent, label, value, options, change); input.dataset.field = key;
            if (key === this.modelKey) input.onclick = async () => {
                if (this.menu.trigger === input) { this.menu.close(true); return; }
                expireMetadataCache();
                const models = await (this.spec.kind === 'loader' ? fetchModelList() : fetchDetectionModelList());
                if (!input.isConnected || this.controller.disposed) return;
                this.modelOptions = groupedModelOptions(models, this.spec.kind);
                if (!this.modelOptions.length) this.modelOptions = groupedModelOptions(spec.options, this.spec.kind);
                this.menu.open(input, label, this.params[key], this.modelOptions, change);
            };
            input.title = `${value}\n${spec.tip}${this.connected(key) ? '\nConnected input overrides this value.' : ''}`; return;
        }
        const wrap = element('label', 'sln-panel-field'); wrap.append(element('span', '', `${label}${this.connected(key) ? ' · input' : ''}`));
        const input = element(spec.multiline ? 'textarea' : 'input'); input.dataset.field = key; input.title = spec.tip;
        if (spec.type === 'BOOLEAN') { input.type = 'checkbox'; input.checked = value; wrap.classList.add('sln-panel-check'); }
        else if (['INT', 'FLOAT'].includes(spec.type)) {
            input.type = 'number'; for (const name of ['min', 'max', 'step']) if (spec[name] != null) input[name] = spec[name]; input.value = value;
            // Keep every integer valid; HTML's min=1 step grid would turn 4096 into 4097.
            if (key === 'max_tokens') input.step = 1;
        } else input.value = value;
        if (spec.multiline) { input.rows = key === 'song_json' ? 8 : 4; this.textSizes.attach(input, key); }
        input.oninput = event => {
            // Native arrows/spinners emit Event; typing and pasting emit InputEvent.
            if (key === 'max_tokens' && event.isTrusted && !('inputType' in event)) {
                const previous = this.params[key], direction = Math.sign(input.valueAsNumber - previous);
                if (Number.isInteger(previous) && direction) input.value = Math.max(spec.min, Math.min(spec.max, previous + direction * spec.step));
            }
            this.edit(state => { state.config.parameters[key] = spec.type === 'BOOLEAN' ? input.checked : input.type === 'number' ? (Number.isFinite(input.valueAsNumber) ? input.valueAsNumber : null) : input.value; }, false);
        };
        wrap.append(input); parent.append(wrap);
    }
    summary(key) {
        const p = this.params;
        return ({model: p[this.modelKey], tasks: this.taskLibrary.selected.map(id => this.taskLibrary.find(id)?.name ?? id).join(' → '),
            images: p.image_grid ? `Grid · up to ${p.grid_frames} frames · ${p.grid_side}px · Every ${p.image_stride}`
                : `Together: ${p.image_handling === 'legacy' ? 'Legacy' : p.image_handling === 'frame_sequence' ? 'On' : 'Off'} · ${p.max_image_side ? `${p.max_image_side}px` : 'Auto size'}${p.image_handling === 'frame_sequence' ? ` · Every ${p.image_stride}` : ''}`,
            video: `${p.video_sample_fps} FPS · ${p.video_cut_threshold ? `Cut threshold ${p.video_cut_threshold}` : 'Continuous shot'}`,
            prompt: p.user_prompt ? `${p.user_prompt.length} characters` : '',
            generation: this.entry?.backend === 'comfyui' ? `${p.max_tokens} tokens` : `${p.max_tokens} tokens · ${p.context_size} context`,
            hardware: p.device, sampling: p.do_sample ? `Temperature ${p.temperature}` : 'Greedy',
            tagging: `${p.threshold} · ${p.char_threshold}`, detection: p.task,
            adjustments: `Crop ${p.crop_factor} · Dilation ${p.dilation}`, mask: `Grow ${p.grow} · Blur ${p.blur}`,
            seed: this.spec.seed ? (this.state.config.seed.source === 'input' ? 'Input' : this.state.config.seed.policy === 'fixed' ? this.state.config.seed.value : this.state.config.seed.policy) : '',
        })[key] ?? '';
    }
    syncSummaries() {
        for (const el of this.controller.root.querySelectorAll('[data-summary]')) el.textContent = this.summary(el.dataset.summary);
    }
    restore() { this.sequence++; this.model = null; this.refresh(false); }
    dispose() { this.sequence++; this.menu.close(); this.textSizes.disconnect(); this.taskLibrary.dispose(); }
    destroy() { this.dispose(); this.unsubscribe?.(); }
}
