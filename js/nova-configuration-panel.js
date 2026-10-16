/** Lifecycle for one serialized DOM configuration panel in classic and Nodes 2.0. */
import { captureFocusedTextWheel } from './nova-panel-wheel.js';

const graphKey = Symbol('SmartLLMNova.ConfigurationGraph');
export class ConfigurationPanel {
    constructor(node, name, options, profile) {
        this.profile = profile;
        this.node = node; this.spec = options.editorSpec; this.raw = options.default; this.height = 0;
        this.state = new this.profile.State(this.raw, this.spec);
        this.root = document.createElement('div'); this.root.className = `sln-panel ${profile.className}`; this.root.setAttribute('aria-label', profile.label);
        if (profile.fillHeight) {
            this.root.classList.add('sln-panel-fill-height');
            this.root.style.setProperty('--comfy-widget-min-height', `${profile.minHeight}px`);
        }
        this.root.onpointerdown = event => event.stopPropagation(); this.root.onkeydown = event => event.stopPropagation();
        this.disposeTextWheel = captureFocusedTextWheel(this.root);
        // Keep the classic canvas resize corner outside an expanding DOM widget.
        const margin = profile.fillHeight ? 4 : 0;
        this.widget = node.addDOMWidget(name, profile.widgetType, this.root, {
            getValue: () => this.value(), setValue: value => this.restore(value),
            ...(profile.fillHeight ? { getMinHeight: () => profile.minHeight + margin * 2 } : {
                getMinHeight: () => this.height, getMaxHeight: () => this.height, getHeight: () => this.height,
            }),
            hideOnZoom: false, serialize: true, margin,
        });
        this.widget.serializeValue = () => this.value();
        // Flexible panels use ComfyUI's native height allocation. Content must
        // never turn the allocated height back into a minimum or resize the node.
        if (!profile.fillHeight) {
            this.widget.computeSize = () => [320, this.height];
            this.observer = new ResizeObserver(() => this.measure()); this.observer.observe(this.root);
        }
        this.form = new profile.Form(this);
        this.form.render();
        const remove = this.widget.onRemove.bind(this.widget);
        this.widget.onRemove = () => { this.dispose(); remove(); };
    }
    value() { return this.invalid ? this.raw : this.state.value(); }
    restore(value) {
        if (this.editing || this.disposed) return;
        this.raw = value; this.form?.dispose();
        try {
            this.state = new this.profile.State(value, this.spec, this.node.properties?.[this.profile.editorKey] ?? {});
            this.invalid = false; this.root.classList.remove('sln-panel-error'); this.form?.render(); this.form?.restore?.();
        } catch (error) {
            this.invalid = true; this.root.classList.add('sln-panel-error');
            const message = document.createElement('div'); message.textContent = `Cannot edit configuration: ${error.message}`;
            this.root.replaceChildren(message); this.measure();
        }
    }
    edit(fn, render = true) {
        if (this.disposed || this.invalid) return;
        const graph = this.node.graph; graph?.beforeChange(); graph?.canvasAction(canvas => canvas.emitBeforeChange()); this.editing = true;
        try {
            fn(this.state);
            this.node.properties ??= {}; this.node.properties[this.profile.editorKey] = JSON.parse(JSON.stringify(this.state.editor));
            this.raw = this.state.value(); this.widget.callback?.(this.raw);
        } finally { this.editing = false; graph?.afterChange(); graph?.canvasAction(canvas => canvas.emitAfterChange()); }
        if (render) this.form.render(); else this.form.syncSummaries();
        this.node.setDirtyCanvas(true, true);
    }
    bindHost() {
        if (!this.node.graph || this.disposed) return;
        const owner = `${this.node.graph[graphKey] ??= crypto.randomUUID()}/${this.node.id}`; this.root.dataset.slnOwner = owner;
        for (const previous of document.querySelectorAll(`.${this.profile.className}[data-sln-owner]`)) {
            if (previous !== this.root && previous.dataset.slnOwner === owner) previous.replaceWith(this.root);
        }
        this.measure();
    }
    measure() {
        if (this.disposed || this.profile.fillHeight || this.frame) return;
        this.frame = requestAnimationFrame(() => {
            this.frame = null; if (!this.root.isConnected || !this.root.offsetWidth) return;
            const height = Math.ceil((this.root.firstElementChild?.offsetHeight ?? this.root.scrollHeight) + 16);
            if (height <= 16 || height === this.height) return;
            this.height = height;
            for (const key of ['height', 'min-height', 'max-height']) this.root.style.setProperty(`--comfy-widget-${key}`, `${height}px`);
            const size = this.node.computeSize(); this.node.setSize([Math.max(320, this.node.size[0]), size[1]]); this.node.setDirtyCanvas(true, true);
        });
    }
    dispose() {
        if (this.disposed) return;
        this.disposed = true; this.observer?.disconnect(); this.disposeTextWheel(); this.form.dispose();
        if (this.frame) cancelAnimationFrame(this.frame); this.profile.onDispose?.(this.node);
    }
}
