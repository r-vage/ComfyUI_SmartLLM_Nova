import { app } from './comfy/index.js';
import { fetchModelEntry, fetchModelList, fetchDetectionModelList, fetchModelTasks } from './smartllm-metadata.js';
import { groupedModelOptions } from './nova-panel-controls.js';
import { onSmartLLMRegistryChanged } from './smartllm-registry-events.js';

const KINDS = new Map([
    ['Smart LM Settings [SmartLLM Nova]', 'loader'],
    ['Smart Detection Settings [SmartLLM Nova]', 'detection'],
]);
const clean = values => (values ?? []).map(value => typeof value === 'string' ? value : value.display_name)
    .filter(value => value && !value.startsWith('__SEP__') && !value.startsWith('─'));

// Resolve fields through actual boundary links, including nested hosts. Display
// labels and widget names on the host may have been renamed by the user.
export function settingsGroups(node, ancestors = new Set()) {
    const kind = KINDS.get(node.comfyClass ?? node.type);
    if (kind) return [{ kind, path: String(node.id), fields: new Map((node.widgets ?? []).map(widget => [widget.name, {
        widget, node, input: node.inputs?.find(input => input.widget?.name === widget.name),
    }])), own: new Set((node.widgets ?? []).map(widget => widget.name)) }];
    if (!node.subgraph || ancestors.has(node.subgraph)) return [];
    const seen = new Set(ancestors).add(node.subgraph), groups = [];
    for (const child of node.subgraph._nodes ?? []) {
        for (const group of settingsGroups(child, seen)) {
            const fields = new Map(group.fields), own = new Set();
            for (const input of node.inputs ?? []) {
                const widget = node.getWidgetFromSlot?.(input);
                if (!widget) continue;
                const boundary = node.subgraph.inputNode.slots.find(slot => slot.name === input.name);
                for (const id of boundary?.linkIds ?? []) {
                    const link = node.subgraph.getLink(id);
                    const target = link && node.subgraph.getNodeById(link.target_id);
                    const slot = target?.inputs?.[link.target_slot];
                    for (const [key, field] of group.fields) {
                        if (field.node === target && field.input === slot) {
                            fields.set(key, { widget, node, input }); own.add(key);
                        }
                    }
                }
            }
            if (own.size) groups.push({ ...group, path: `${node.id}/${group.path}`, fields, own });
        }
    }
    return groups;
}

const originalChoices = new WeakMap();
function initialChoices(widget) {
    if (!originalChoices.has(widget)) originalChoices.set(widget, clean(widget.options?.values));
    return originalChoices.get(widget);
}

// Partial promotion can make several hosts depend on the same interior widget.
// Share one callback wrapper and release it when the last binding goes away.
const widgetListeners = new WeakMap();
function watchWidget(widget, listener) {
    if (!widget) return () => {};
    let record = widgetListeners.get(widget);
    if (!record) {
        record = { previous: widget.callback, listeners: new Set() };
        record.callback = function (...args) {
            const result = record.previous?.apply(this, args);
            for (const notify of record.listeners) notify();
            return result;
        };
        widget.callback = record.callback; widgetListeners.set(widget, record);
    }
    record.listeners.add(listener);
    return () => {
        record.listeners.delete(listener);
        if (!record.listeners.size) {
            if (widget.callback === record.callback) widget.callback = record.previous;
            widgetListeners.delete(widget);
        }
    };
}

export function bindSettings(node, group) {
    const modelKey = group.kind === 'loader' ? 'model' : 'model_name';
    const model = group.fields.get(modelKey), system = group.fields.get('system_prompt');
    let disposed = false, sequence = 0;
    let entry, tasks, observedModel, observedLink, loadedModel, loadedLink;
    const disabledBefore = new Map();
    for (const field of group.fields.values()) initialChoices(field.widget);
    function choices(key, values, userDriven, fallback = true) {
        if (!group.own.has(key)) return;
        const field = group.fields.get(key), widget = field.widget;
        if (!values?.length && fallback) values = initialChoices(widget);
        if (!values) return;
        if (userDriven && values.length && field.input?.link == null && !values.includes(widget.value)) {
            widget.value = values[0]; widget.callback?.(widget.value);
        }
        // Preserve saved values without adding incompatible tasks to the menu.
        widget.options.values = [...values];
    }
    function disabled(key, value) {
        if (!group.own.has(key)) return;
        const widget = group.fields.get(key).widget;
        if (!disabledBefore.has(widget)) disabledBefore.set(widget, {
            value: widget.disabled, option: widget.options.disabled,
        });
        // Classic uses the widget flag; Nodes 2.0 and promoted controls use options.
        widget.disabled = value; widget.options.disabled = value;
    }
    function updateControls(userDriven = false) {
        if (disposed) return;
        const current = loadedModel === model?.widget.value && loadedLink === model?.input?.link ? entry : null;
        const specialized = ['wd14', 'yolo'].includes(current?.backend);
        const direct = group.kind === 'loader' && current && !specialized && current.family !== 'Florence'
            && (system?.input?.link != null || String(system?.widget.value ?? '').trim().length > 0);
        const gguf = ['gguf', 'llamacpp'].includes(current?.backend);
        const task = group.fields.get('task');
        // Inactive specialized-model tasks retain a display value, so Nodes 2.0
        // does not mark an intentionally disabled control as an invalid choice.
        choices('task', direct ? ['Direct Chat'] : specialized ? [task.widget.value] : current ? clean(tasks) : null, userDriven);
        if (direct && group.own.has('task') && task.input?.link == null && task.widget.value !== 'Direct Chat') {
            task.widget.value = 'Direct Chat'; task.widget.callback?.('Direct Chat');
        }
        choices('quantization', clean(current?.quantizations), userDriven && gguf);
        disabled('task', Boolean(direct || specialized));
        disabled('quantization', !gguf);
        node.setDirtyCanvas?.(true, true);
    }
    async function refresh(userDriven = false) {
        if (disposed) return;
        const selected = observedModel = model?.widget.value, link = observedLink = model?.input?.link, request = ++sequence;
        updateControls();
        const [models, fetchedEntry] = await Promise.all([
            group.kind === 'loader' ? fetchModelList() : fetchDetectionModelList(),
            link == null ? fetchModelEntry(selected) : null,
        ]);
        const fetchedTasks = fetchedEntry ? await fetchModelTasks(group.kind, fetchedEntry) : null;
        if (disposed || request !== sequence || selected !== model?.widget.value || link !== model?.input?.link) return;
        entry = fetchedEntry; tasks = fetchedTasks; loadedModel = selected; loadedLink = link;
        choices(modelKey, groupedModelOptions(models, group.kind).flatMap(group => group.options), false);
        updateControls(userDriven);
    }
    const unwatch = [watchWidget(model?.widget, () => { void refresh(true); }),
        watchWidget(system?.widget, () => updateControls())];
    void refresh();
    return {
        refresh,
        sync() {
            if (observedModel !== model?.widget.value || observedLink !== model?.input?.link) void refresh();
            else updateControls();
        },
        dispose() {
            disposed = true; sequence++;
            for (const release of unwatch) release();
            for (const [widget, before] of disabledBefore) {
                widget.disabled = before.value;
                if (before.option === undefined) delete widget.options.disabled;
                else widget.options.disabled = before.option;
            }
        },
    };
}

// Topology events rebuild bindings. Editing a value only refreshes that group's
// choices. There are no timers or draw-loop scans, and removed graphs release
// their listeners and callback wrappers.
export function installNativeSettings(root) {
    const graphs = new Map();
    const connections = new Map();
    let bindings = new Map();
    let pending = false, disposed = false;
    function schedule() {
        if (pending || disposed) return;
        pending = true;
        queueMicrotask(() => { pending = false; if (!disposed) rebuild(); });
    }
    function rebuild() {
        const visited = new Set(), next = new Map(), retained = new Set();
        function visit(graph) {
            if (!graph || visited.has(graph)) return;
            visited.add(graph);
            if (!graphs.has(graph)) {
                const controller = new AbortController();
                for (const event of ['node:added', 'node:removed', 'configured', 'subgraph-created',
                    'widget-promoted', 'widget-demoted', 'input-added', 'removing-input', 'inputs-reordered', 'renaming-input']) {
                    graph.events?.addEventListener(event, schedule, { signal: controller.signal });
                }
                graphs.set(graph, controller);
            }
            for (const node of graph._nodes ?? []) {
                const records = new Map();
                for (const group of settingsGroups(node)) {
                    const previous = bindings.get(node)?.get(group.path);
                    const same = previous && group.own.size === previous.group.own.size
                        && group.fields.size === previous.group.fields.size
                        && [...group.fields].every(([key, field]) => {
                            const old = previous.group.fields.get(key);
                            return old?.widget === field.widget && old.input === field.input
                                && group.own.has(key) === previous.group.own.has(key);
                        });
                    // An unrelated graph edit must not cancel an in-flight
                    // user model change or replace its widget callbacks.
                    const record = same ? previous : { group, node };
                    records.set(group.path, record); retained.add(record);
                }
                if (records.size) next.set(node, records);
                visit(node.subgraph);
            }
        }
        visit(root);
        for (const [graph, controller] of graphs) {
            if (!visited.has(graph)) { controller.abort(); graphs.delete(graph); }
        }
        for (const records of bindings.values()) for (const record of records.values()) {
            if (!retained.has(record)) record.binding.dispose();
        }
        for (const records of next.values()) for (const record of records.values()) {
            if (record.binding) record.binding.sync();
            else record.binding = bindSettings(record.node, record.group);
        }
        for (const [node, hook] of connections) if (!next.has(node)) {
            if (node.onConnectionsChange === hook.callback) node.onConnectionsChange = hook.previous;
            connections.delete(node);
        }
        for (const node of next.keys()) if (!connections.has(node)) {
            const previous = node.onConnectionsChange;
            const callback = function (...args) {
                const result = previous?.apply(this, args); schedule(); return result;
            };
            connections.set(node, { previous, callback }); node.onConnectionsChange = callback;
        }
        bindings = next;
    }
    const unsubscribe = onSmartLLMRegistryChanged(() => {
        for (const records of bindings.values()) for (const { binding } of records.values()) void binding.refresh();
    });
    rebuild();
    return () => {
        disposed = true; unsubscribe();
        for (const records of bindings.values()) for (const { binding } of records.values()) binding.dispose();
        for (const controller of graphs.values()) controller.abort();
        for (const [node, hook] of connections) if (node.onConnectionsChange === hook.callback) node.onConnectionsChange = hook.previous;
        connections.clear();
        bindings.clear(); graphs.clear();
    };
}

let dispose;
function bindActiveGraph() {
    dispose?.(); dispose = installNativeSettings(app.rootGraph ?? app.graph);
}
app.registerExtension({
    name: 'SmartLLMNova.NativeSettings',
    setup: bindActiveGraph,
    afterConfigureGraph: bindActiveGraph,
});
