import { SeedState } from './nova-panel-seed.js';
export class PanelState extends SeedState {
    constructor(raw, spec, editor = {}) {
        super(); this.spec = spec; this.config = JSON.parse(raw);
        if (this.config.version !== 1 || this.config.kind !== spec.kind || !this.config.parameters
            || Object.entries(spec.fields).some(([key, field]) => !Object.hasOwn(this.config.parameters, key) && !field.optional)) throw Error('Unsupported configuration.');
        for (const [key, field] of Object.entries(spec.fields)) {
            if (!Object.hasOwn(this.config.parameters, key)) this.config.parameters[key] = structuredClone(field.default);
        }
        this.editor = JSON.parse(JSON.stringify(editor));
    }
    toggleSection(key) {
        const open = !this.sectionOpen(key); this.editor.sections ??= {};
        if (open) for (const other of ["model", "prompt", "generation", "sampling", "tagging", "hardware", "detection", "adjustments", "mask", "seed"]) {
            if (!this.editor.pins?.[other]) this.editor.sections[other] = false;
        }
        this.editor.sections[key] = open;
    }
    value() { return JSON.stringify(this.config); }
    sectionOpen(key) { return this.editor.sections?.[key] ?? ['model', 'prompt', 'detection', 'json'].includes(key); }
}
