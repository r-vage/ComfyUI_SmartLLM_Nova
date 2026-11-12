import { api } from './comfy/index.js';
import { button, element } from './nova-panel-controls.js';
import { emitSmartLLMRegistryChanged } from './smartllm-registry-events.js';

export async function taskRequest(path, body) {
    const response = await api.fetchApi(`/smartllm-nova/tasks${path}`, body === undefined ? {} : {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw Error(data.error || `Task request failed (${response.status}).`);
    return data;
}

export function selectedTasks(parameters, catalog) {
    let references;
    if (parameters.task_sequence) {
        try { references = JSON.parse(parameters.task_sequence).tasks; } catch { return [parameters.task_sequence]; }
        if (!Array.isArray(references)) return ['Invalid task selection'];
    } else {
        references = [parameters.task];
        if (parameters.multi_task_mode) for (const key of ['task_2', 'task_3', 'task_4']) {
            if (!parameters[key] || parameters[key] === 'None') break;
            references.push(parameters[key]);
        }
    }
    return references.map(reference => catalog?.tasks?.find(task => task.id === reference || task.name === reference || task.aliases.includes(reference))?.id ?? reference);
}

function chainIssue(task, previous, entry) {
    if (task.issue || !previous.length) return task.issue;
    if (entry?.family === 'Florence') return 'Florence supports one native task.';
    if (task.input.images === 'required') return 'Only the first task receives images.';
    if (task.input.text === 'none') return 'This task does not accept a previous text result.';
    if (!['text', 'lyrics'].includes(previous.at(-1)?.output)) return 'The previous task has a structured output. Use a separate node.';
    if (previous.at(-1)?.output === 'lyrics' && !['music3', 'yue2', 'chat', 'direct_chat', 'story'].includes(task.handler)) return 'This task cannot consume a lyric sheet.';
    return '';
}

export class TaskLibrary {
    constructor(form) { this.form = form; this.search = ''; this.request = 0; this.editorRequest = 0; this.collapsedGroups = new Set(); }
    get catalog() { return this.form.catalog; }
    get groups() { return this.catalog?.groups ?? { vision: 'Vision', text: 'Text', detection: 'Detection' }; }
    get selected() { return selectedTasks(this.form.params, this.catalog); }
    find(id) { return this.catalog?.tasks.find(task => task.id === id); }
    async select(ids) {
        if (this.selecting) return;
        const model = this.form.params[this.form.modelKey], request = ++this.request;
        this.selecting = true; this.form.render();
        try {
            if (ids.length) await taskRequest('/validate', { tasks: ids, model,
                kind: this.form.spec.kind, scene_json: !!this.form.params.scene_json });
            if (this.form.controller.disposed || request !== this.request || model !== this.form.params[this.form.modelKey]) return;
            this.status = '';
            this.form.edit(state => {
                const p = state.config.parameters;
                p.task = ids[0] ?? '';
                if (this.form.spec.kind === 'loader') {
                    p.task_sequence = JSON.stringify({ version: 1, tasks: ids });
                    p.multi_task_mode = ids.length > 1;
                    ['task_2', 'task_3', 'task_4'].forEach((key, index) => { p[key] = ids[index + 1] ?? 'None'; });
                }
            });
        } catch (error) {
            if (request === this.request && model === this.form.params[this.form.modelKey]) this.status = error.message;
        } finally {
            this.selecting = false;
            if (!this.form.controller.disposed) this.form.render();
        }
    }
    render(parent) {
        const section = element('section', 'sln-panel-section'), header = element('div', 'sln-panel-section-header');
        section.setAttribute('aria-label', 'Tasks');
        const trigger = button('', 'Tasks', () => this.openPicker());
        trigger.className = 'sln-panel-section-toggle sln-task-trigger'; trigger.dataset.field = 'section.tasks';
        trigger.setAttribute('aria-haspopup', 'dialog'); trigger.setAttribute('aria-expanded', String(!!this.picker?.open));
        const summary = this.selected.map(id => this.find(id)?.name ?? id).filter(Boolean).join(' → ') || 'Select a task';
        trigger.title = `Choose and edit tasks · ${summary}`;
        const value = element('span', 'sln-panel-summary', summary); value.dataset.summary = 'tasks';
        trigger.append(element('strong', '', 'Tasks'), value, element('span', '', '↗'));
        header.append(trigger); section.append(header); parent.append(section);
        if (!this.selected.length) section.append(element('div', 'sln-panel-error', 'Select a task.'));
    }
    openPicker(loadEditor = true) {
        if (this.picker?.open) { this.picker.querySelector('[data-field="tasks.search"]')?.focus(); return; }
        const picker = element('dialog', 'sln-panel sln-task-picker'); this.picker = picker;
        picker.setAttribute('aria-label', 'Tasks');
        Object.assign(picker.style, this.pickerSize ?? {});
        const heading = element('div', 'sln-panel-tools');
        this.pickerClose = button('×', 'Close tasks', () => this.closePicker());
        heading.append(element('strong', '', 'Tasks'), this.pickerClose);
        const columns = element('div', 'sln-task-columns');
        this.taskPane = element('div', 'sln-task-choices'); this.taskPane.setAttribute('aria-label', 'Task selection');
        this.editorPane = element('div', 'sln-task-editor'); this.editorPane.setAttribute('aria-label', 'Task editor');
        columns.append(this.taskPane, this.editorPane); picker.append(heading, columns);
        picker.oncancel = event => { event.preventDefault(); this.closePicker(); };
        picker.onkeydown = event => event.stopPropagation(); picker.onpointerdown = event => event.stopPropagation();
        document.body.append(picker); this.clearEditor(); picker.showModal(); this.form.render();
        this.taskPane.querySelector('[data-field="tasks.search"]')?.focus();
        if (loadEditor && this.find(this.selected[0])) void this.edit(this.selected[0], false);
    }
    refreshPicker() {
        if (!this.picker?.open) return;
        const active = this.taskPane.contains(document.activeElement) ? document.activeElement : null;
        const field = active?.dataset.field, caret = active?.type === 'search' ? [active.selectionStart, active.selectionEnd] : null;
        const key = JSON.stringify(this.selected), scroll = key === this.pickerSelection ? this.taskPane.querySelector('.sln-task-list')?.scrollTop ?? 0 : 0;
        this.pickerSelection = key; this.pickerClose.disabled = !!this.saving;
        this.taskPane.replaceChildren(); this.renderChoices(this.taskPane);
        this.taskPane.querySelector('.sln-task-list').scrollTop = scroll;
        if (field) {
            const restored = [...this.taskPane.querySelectorAll('[data-field]')].find(input => input.dataset.field === field);
            restored?.focus({ preventScroll: true }); if (caret) restored?.setSelectionRange(...caret);
        }
    }
    closePicker(force = false) {
        if (!force && (this.saving || this.canDiscard && !this.canDiscard())) return;
        this.editorRequest++;
        if (this.picker) this.pickerSize = { width: this.picker.style.width, height: this.picker.style.height };
        this.picker?.close(); this.picker?.remove(); this.picker = null;
        this.taskPane = null; this.editorPane = null; this.canDiscard = null; this.editingId = null;
        if (!force && !this.form.controller.disposed) {
            this.form.render(); this.form.controller.root.querySelector('[data-field="section.tasks"]')?.focus({ preventScroll: true });
        }
    }
    clearEditor() {
        this.editorRequest++; this.editingId = null; this.canDiscard = null;
        this.editorPane?.replaceChildren(element('strong', '', 'Task editor'),
            element('div', 'sln-panel-note', 'Choose Edit beside a task, or New task. Editing a task does not select it.'));
    }
    renderChoices(parent) {
        const form = this.form, detection = form.spec.kind === 'detection', ids = this.selected;
        const tools = element('div', 'sln-panel-tools');
        const create = button('New task', 'Create a task', () => this.edit(null)); create.disabled = !this.catalog || !!this.saving;
        tools.append(create, button('↻', 'Refresh task files', () => form.refresh(false)));
        parent.append(tools);
        const search = element('input'); search.type = 'search'; search.placeholder = 'Search tasks…'; search.value = this.search;
        search.setAttribute('aria-label', 'Search tasks'); search.dataset.field = 'tasks.search'; parent.append(search);
        if (!detection && form.entry?.has_vision === false) parent.append(element('div', 'sln-panel-note', 'Text-only model · connected images, video and disk frames are ignored.'));
        const list = element('div', 'sln-task-list'); list.tabIndex = 0; list.dataset.captureWheel = 'true';
        list.setAttribute('aria-label', detection ? 'Detection tasks' : 'Tasks by purpose');
        list.addEventListener('wheel', event => { if (!event.ctrlKey && !event.metaKey) event.stopPropagation(); }, { passive: true });
        const draw = () => {
            list.replaceChildren();
            const tasks = this.catalog?.tasks ?? [], previous = ids.map(id => this.find(id));
            const groups = detection ? [['detection', this.groups.detection]] : [['selected', 'Selected · run order'], ...Object.entries(this.groups).filter(([key]) => key !== 'detection')];
            for (const [group, heading] of groups) {
                // Keep the active chain visible even while searching for another task.
                const matches = group === 'selected' ? previous.filter(Boolean) : tasks.filter(task =>
                    task.group === group && (detection || !ids.includes(task.id)) &&
                    `${heading} ${task.name} ${task.description}`.toLowerCase().includes(this.search.toLowerCase())
                ).sort((a, b) => a.name.localeCompare(b.name));
                if (!matches.length) continue;
                const section = element(group === 'selected' ? 'section' : 'details', 'sln-task-group'); section.dataset.group = group;
                const title = element(group === 'selected' ? 'div' : 'summary', 'sln-panel-group-title', `${heading} · ${matches.length}`);
                section.append(title);
                if (group !== 'selected') {
                    section.open = !!this.search.trim() || !this.collapsedGroups.has(group);
                    title.onclick = () => {
                        if (this.search.trim()) return;
                        // Record the user's choice before the native toggle. Its
                        // queued event can arrive after search replaces the list.
                        if (section.open) this.collapsedGroups.add(group); else this.collapsedGroups.delete(group);
                    };
                }
                for (const task of matches) {
                    const selected = ids.includes(task.id), issue = detection ? task.issue : chainIssue(task, selected ? previous.slice(0, ids.indexOf(task.id)) : previous, form.entry);
                    const row = element('div', 'sln-task-row');
                    const label = element('label');
                    const input = element('input'); input.type = detection ? 'radio' : 'checkbox'; input.name = `sln-task-${form.controller.node.id}`;
                    input.dataset.field = `task.${task.id}`;
                    input.checked = selected; input.disabled = this.selecting || !selected && !!issue; input.setAttribute('aria-label', task.name);
                    input.onchange = () => this.select(detection ? [task.id] : selected ? ids.filter(id => id !== task.id) : [...ids, task.id]);
                    const title = element('span', '', task.name); label.title = issue || task.description;
                    title.append(element('small', 'sln-task-input-policy', { required: 'Images required', optional: 'Images optional', none: 'Text only' }[task.input.images]));
                    label.append(input, title); row.append(label);
                    if (selected && !detection) row.append(element('span', 'sln-task-step', String(ids.indexOf(task.id) + 1)));
                    const edit = button('Edit', `Edit ${task.name}`, () => this.edit(task.id)); edit.disabled = !!this.saving;
                    edit.setAttribute('aria-pressed', String(this.editingId === task.id)); edit.dataset.field = `edit.${task.id}`; row.append(edit);
                    if (issue) { row.dataset.unavailable = 'true'; title.append(element('small', '', issue)); }
                    section.append(row);
                }
                list.append(section);
            }
            if (!list.children.length) list.append(element('div', 'sln-panel-note', this.catalog ? 'No matching tasks.' : 'Loading tasks…'));
        };
        search.oninput = () => { this.search = search.value; draw(); }; draw(); parent.append(list);
        const selected = element('div', 'sln-task-chain');
        selected.append(element('div', 'sln-panel-note', detection ? 'One task; grounding accepts comma-separated targets.' : 'Run in this order. Each next task receives the previous text.'));
        ids.forEach((id, index) => {
            const task = this.find(id), row = element('div', 'sln-task-row');
            row.append(element('span', task ? '' : 'sln-panel-error', `${index + 1}. ${task?.name ?? `${id || 'No task selected'} (unavailable)`}`));
            if (!detection) {
                const up = button('↑', `Move task ${index + 1} earlier`, () => { const next = [...ids]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; this.select(next); });
                up.disabled = this.selecting || index === 0; row.append(up);
                const down = button('↓', `Move task ${index + 1} later`, () => { const next = [...ids]; [next[index + 1], next[index]] = [next[index], next[index + 1]]; this.select(next); });
                down.disabled = this.selecting || index === ids.length - 1; row.append(down);
                const remove = button('×', `Remove task ${index + 1} from selection`, () => this.select(ids.filter((_, i) => i !== index)));
                remove.disabled = this.selecting; row.append(remove);
            }
            selected.append(row);
        }); parent.append(selected);
        if (!ids.length) parent.append(element('div', 'sln-panel-error', 'Select a task.'));
        for (const message of [this.status, ...(this.catalog?.errors ?? [])].filter(Boolean)) parent.append(element('div', 'sln-panel-error', message));
        if (this.catalog?.deleted?.length) {
            const removed = element('details'); removed.append(element('summary', '', 'Removed tasks'));
            for (const task of this.catalog.deleted) removed.append(button(`Restore ${task.name}`, `Restore ${task.name}`, async () => {
                try { await taskRequest('/edit', { action: 'restore', id: task.id }); emitSmartLLMRegistryChanged(); }
                catch (error) { this.status = error.message; form.render(); }
            }));
            parent.append(removed);
        }
    }
    prompt(parent) {
        const ids = this.selected, task = this.find(ids[0]);
        if (!task) return;
        const form = this.form, native = form.entry?.family === 'Florence';
        const details = element('details', 'sln-task-preview'); details.append(element('summary', '', native ? 'Native task instructions' : 'System instructions · preview'));
        const text = element('textarea'); text.readOnly = true; text.rows = 5; text.setAttribute('aria-label', 'Effective system instructions');
        const note = element('div', 'sln-panel-note');
        const refresh = button('Refresh preview', 'Refresh prompt preview', update);
        details.append(text, note, refresh, button('Edit task file', 'Edit task file', () => this.edit(task.id)));
        let local;
        if (task.handler === 'direct_chat' && !native) {
            const label = element('label', 'sln-panel-field'); label.append(element('span', '', 'System prompt · Direct Chat'));
            local = element('textarea'); local.rows = 4; local.value = form.params.direct_system_prompt || '';
            local.setAttribute('aria-label', 'Direct Chat system prompt');
            form.textSizes.attach(local, 'direct_system_prompt');
            local.oninput = () => form.edit(state => { state.config.parameters.direct_system_prompt = local.value; }, false);
            label.append(local); parent.append(label);
            parent.append(element('div', 'sln-panel-note', 'Edit here for this node, or edit the task file for every use. Clearing this field uses the task file. A connected system prompt takes priority.'));
        }
        parent.append(details);
        async function update(defaultOnly = false) {
            try {
                const data = defaultOnly === true
                    ? await taskRequest(`/preview?id=${encodeURIComponent(task.id)}`)
                    : await taskRequest('/preview', { id: task.id,
                        user: form.params.user_prompt ?? form.params.user_input ?? '', direct_system: form.params.direct_system_prompt ?? '' });
                if (!details.isConnected) return;
                text.value = native ? `${data.operation || 'Unsupported native operation'}${form.catalog?.operations[task.florence_operation]?.text ? `\nQuery: ${data.user}` : '\nNo system prompt or guiding text is sent for this operation.'}` : data.system;
                if (local && !form.params.direct_system_prompt && document.activeElement !== local) local.value = data.system;
                note.textContent = native ? 'Florence uses a protected task token and supports query text only for specific operations.'
                    : 'The task supplies system instructions; user guidance stays in the user role. Scene, grid and song contracts may add request-specific instructions at execution.';
            } catch (error) { if (details.isConnected) note.textContent = error.message; }
        }
        details.ontoggle = () => { if (details.open) void update(); };
        if (local && !form.params.direct_system_prompt) void update(true);
    }
    async edit(id, focus = true) {
        if (this.saving || this.canDiscard && !this.canDiscard()) return;
        this.openPicker(false);
        const request = ++this.editorRequest;
        this.editorPane.inert = true; this.editorPane.setAttribute('aria-busy', 'true');
        try {
            const loaded = id ? await taskRequest(`/read?id=${encodeURIComponent(id)}`) : { task: structuredClone(this.catalog.template), revision: null, file: 'New task' };
            if (this.form.controller.disposed || !this.picker?.open || request !== this.editorRequest) return;
            this.openEditor(loaded, !id);
        } catch (error) { if (request === this.editorRequest) { this.status = error.message; this.form.render(); } }
        finally {
            if (request === this.editorRequest && this.editorPane) {
                this.editorPane.inert = false; this.editorPane.removeAttribute('aria-busy');
                if (focus) this.editorPane.querySelector('input')?.focus();
            }
        }
    }
    openEditor(loaded, create) {
        const draft = structuredClone(loaded.task), editor = this.editorPane;
        editor.replaceChildren(); this.editingId = create ? null : draft.id;
        const form = this.form, native = form.entry?.family === 'Florence' || draft.handler === 'florence';
        editor.append(element('strong', '', create ? 'New task' : `Edit · ${draft.name}`), element('div', 'sln-panel-note', `${loaded.file} · ${draft.id}`));
        const fields = element('div', 'sln-panel-content'); editor.append(fields);
        let dirty = false;
        this.canDiscard = () => !dirty || confirm('Discard unsaved task edits?');
        let fieldParent = fields;
        const bind = (label, value, set, multiline = false, options = null) => {
            const wrap = element('label', 'sln-panel-field'); wrap.append(element('span', '', label));
            const input = element(options ? 'select' : multiline ? 'textarea' : 'input'); input.setAttribute('aria-label', label);
            if (options) for (const [key, text] of options) { const option = element('option', '', text); option.value = key; input.append(option); }
            input.value = value; if (multiline) input.rows = label.includes('instructions') ? 9 : 3;
            input.oninput = () => { dirty = true; set(input.value); }; wrap.append(input); fieldParent.append(wrap); return input;
        };
        bind('Task name', draft.name, value => { draft.name = value; });
        bind('Description', draft.description, value => { draft.description = value; });
        if (!native) bind('System instructions', draft.instructions, value => { draft.instructions = value; }, true);
        bind(native ? 'Query template ({prompt})' : 'User template ({prompt})', draft.template, value => { draft.template = value; }, true);
        const behavior = element('details'); behavior.append(element('summary', '', native ? 'Native task behavior' : 'Task behavior'));
        behavior.open = native; fieldParent = element('div', 'sln-panel-content'); behavior.append(fieldParent); fields.append(behavior);
        const detection = draft.nodes.includes('detection');
        if (!detection) bind('Category', this.catalog.group_aliases?.[draft.group] ?? draft.group, value => { draft.group = value; }, false,
            Object.entries(this.groups).filter(([key]) => key !== 'detection'));
        const behaviors = { chat: 'Free-form text', direct_chat: 'Direct Chat', story: 'Story', tags_to_text: 'Tags to text', text_to_tags: 'Text to tags',
            song_lyrics: 'Lyric sheet', music3: 'MiniMax Music 3 JSON', yue2: 'YuE2 JSON', scene_motion: 'Scene motion JSON',
            scene_description: 'Full scene JSON', h3_scene: 'H3 scene', h3_t2va: 'H3 text timeline', h3_i2va: 'H3 first-image timeline',
            h3_fl2va: 'H3 first/last-image timeline', h3_l2va: 'H3 last-image timeline', grounding: 'Grounding targets', region_caption: 'Detect objects', florence: 'Florence native task' };
        bind('Output / behavior', draft.handler, value => { draft.handler = value; }, false,
            Object.keys(this.catalog.handlers).filter(key => !detection || ['grounding', 'region_caption', 'florence'].includes(key)).map(key => [key, behaviors[key] ?? key]));
        bind('Images', draft.input.images, value => { draft.input.images = value; }, false, ['required', 'optional', 'none'].map(key => [key, key]));
        if (!detection && !native) bind('Image sequences', String(draft.input.sequence), value => { draft.input.sequence = value === 'true'; }, false, [['true', 'Accept Together / Grid'], ['false', 'Individual images only']]);
        bind('User text', draft.input.text, value => { draft.input.text = value; }, false, ['required', 'optional', 'none'].map(key => [key, key]));
        bind('Enabled', String(draft.enabled), value => { draft.enabled = value === 'true'; }, false, [['true', 'Enabled'], ['false', 'Draft (disabled)']]);
        if (native) fieldParent.append(element('div', 'sln-panel-note', 'Native Florence codes and parsers are protected. Changing prose cannot add a model capability. The selected checkpoint must support the operation.'));
        bind('Florence operation', draft.florence_operation ?? '', value => { draft.florence_operation = value || null; }, false,
            [['', 'None'], ...Object.entries(this.catalog.operations).map(([key, op]) => [key, op.token])]);
        fieldParent = fields;
        const status = element('div', 'sln-panel-note'); status.setAttribute('role', 'status');
        const actions = element('div', 'sln-panel-tools'); editor.append(status, actions);
        const mutate = async operation => {
            if (this.saving) return;
            const request = this.editorRequest;
            this.saving = true;
            const controls = [...editor.querySelectorAll('input, select, textarea, button')];
            controls.forEach(control => { control.disabled = true; }); this.refreshPicker();
            try {
                await operation(() => !form.controller.disposed && this.picker?.open && request === this.editorRequest);
            } catch (error) { if (request === this.editorRequest) { status.textContent = error.message; status.className = 'sln-panel-error'; } }
            finally {
                this.saving = false; controls.forEach(control => { control.disabled = false; }); this.refreshPicker();
            }
        };
        const save = action => mutate(async active => {
            // The operation owns its query policy; never split document questions.
            draft.input.query = draft.handler === 'grounding' ? 'terms' : 'single';
            const protectedFields = ['id', 'version', 'aliases', 'internal'];
            const changes = Object.fromEntries(Object.entries(draft).filter(([key, value]) => !protectedFields.includes(key) &&
                (create || JSON.stringify(value) !== JSON.stringify(loaded.task[key]))));
            const body = { action: create ? 'create' : action, id: loaded.task.id, revision: loaded.revision, changes };
            if (new TextEncoder().encode(JSON.stringify(body)).length > 16384) throw Error('This edit exceeds 16 KiB. Save smaller changes or edit the task JSON file directly. Your draft is retained.');
            const result = await taskRequest('/edit', body);
            emitSmartLLMRegistryChanged();
            if (!active()) return;
            dirty = false; this.status = ''; this.openEditor(result, false);
            this.editorPane.querySelector('[role="status"]').textContent = `Saved ${result.task.name}.${this.selected.includes(result.task.id) ? '' : ' Select it on the left to use it.'}`;
        });
        actions.append(button('Save', 'Save task', () => save('save')));
        if (!create) {
            actions.append(button('Save as new', 'Save as a new task', () => save('duplicate')));
            actions.append(button('Remove', 'Remove task file', () => {
                if (!confirm(`Remove ${loaded.task.name}? Saved selections will show it as unavailable. It can be restored from Tasks.`)) return;
                return mutate(async active => {
                    await taskRequest('/edit', { action: 'remove', id: loaded.task.id, revision: loaded.revision });
                    emitSmartLLMRegistryChanged();
                    if (active()) { dirty = false; this.clearEditor(); }
                });
            }));
        }
        actions.append(button('Cancel', 'Cancel task edits', () => {
            if (!this.saving && this.canDiscard()) { this.clearEditor(); this.refreshPicker(); }
        }));
        this.refreshPicker();
    }
    dispose() { this.request++; this.closePicker(true); }
}
