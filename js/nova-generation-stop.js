/** Execution state is ephemeral: Stop never changes the saved workflow. */
import { app, api } from './comfy/index.js';
import { button, element } from './nova-panel-controls.js';

const listeners = new Set();
let snapshot = { revision: -1, runs: [] }, request, subscription = 0;

function publish(value) {
    if (!Array.isArray(value?.runs) || !Number.isInteger(value.revision) || value.revision < snapshot.revision) return;
    snapshot = value;
    for (const listener of listeners) listener(snapshot.runs);
}
function onState(event) { publish(event.detail); }
function disconnected() {
    request?.abort(); request = null;
    snapshot = { revision: -1, runs: [] };
    for (const listener of listeners) listener([]);
}
async function refresh() {
    request?.abort(); const pending = request = new AbortController(), current = subscription;
    try {
        const response = await api.fetchApi('/smartllm-nova/generation', { cache: 'no-store', signal: pending.signal });
        if (!response.ok) return;
        const value = await response.json();
        if (!pending.signal.aborted && current === subscription) publish(value);
    } catch { /* A disconnected server cannot accept Stop; reconnect refreshes state. */ }
    finally { if (request === pending) request = null; }
}
function subscribe(listener) {
    listeners.add(listener);
    if (listeners.size === 1) {
        subscription++;
        api.addEventListener('smartllm-nova/generation', onState);
        api.addEventListener('reconnecting', disconnected);
        api.addEventListener('reconnected', refresh);
        void refresh();
    }
    listener(snapshot.runs);
    return () => {
        listeners.delete(listener);
        if (listeners.size) return;
        subscription++;
        api.removeEventListener('smartllm-nova/generation', onState);
        api.removeEventListener('reconnecting', disconnected);
        api.removeEventListener('reconnected', refresh);
        disconnected();
    };
}
function graphNode(path) {
    let graph = app.graph, node;
    for (const id of String(path).split(':')) { node = graph?.getNodeById(id); graph = node?.subgraph; }
    return node;
}

export class GenerationStopControl {
    constructor(controller) {
        this.controller = controller;
        this.button = button('■', 'Stop LM', () => this.stop());
        this.button.className = 'sln-generation-stop';
        this.note = element('span', 'sln-panel-note'); this.note.setAttribute('role', 'status');
        this.unsubscribe = subscribe(runs => {
            const run = runs.find(run => graphNode(run.display_node_id) === controller.node || graphNode(run.node_id) === controller.node);
            if (run?.run_id !== this.run?.run_id) { this.note.textContent = ''; this.retry = false; }
            this.run = run; this.update();
        });
    }
    render(bar, parent) { bar.append(this.button); parent.append(this.note); this.update(); }
    update() {
        const stopping = this.run?.stopping && !this.retry;
        this.button.disabled = !this.run || this.pending || stopping;
        this.button.setAttribute('aria-busy', String(Boolean(this.pending || stopping)));
        this.button.title = this.pending || stopping ? 'Stopping LM… Waiting for cleanup.'
            : !this.run ? 'Stop LM — available while this loader is running.'
            : ['ollama', 'vllm', 'sglang', 'llamacpp'].includes(this.run.backend)
                ? 'Interrupt this workflow and force-stop its managed LM Docker backend, including with Keep loaded on.'
                : 'Interrupt this workflow. Native models stop at their next cancellation check; a blocked native call must return first.';
        this.note.hidden = !this.note.textContent;
        this.controller.measure();
    }
    async stop() {
        if (this.button.disabled || !this.run) return;
        const runId = this.run.run_id;
        this.pending = true; this.retry = false; this.note.textContent = ''; this.update();
        try {
            const response = await api.fetchApi('/smartllm-nova/generation/stop', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ run_id: runId }), timeoutMs: 30000,
            });
            const data = await response.json();
            if (!response.ok || !data.success) throw Error(data.error ?? 'LM stop failed.');
            if (!this.disposed && this.run?.run_id === runId) this.note.textContent = data.message;
        } catch (error) {
            if (!this.disposed && (!this.run || this.run.run_id === runId)) {
                this.retry = true; this.note.textContent = error.message;
            }
        } finally {
            this.pending = false;
            if (!this.disposed) { this.update(); void refresh(); }
        }
    }
    destroy() { this.disposed = true; this.unsubscribe(); }
}
