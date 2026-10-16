/** Persist native text-field resize handles without watching canvas geometry. */
export class TextFieldSizes {
    constructor(controller) {
        this.controller = controller;
        this.observer = new MutationObserver(records => {
            if (controller.disposed || controller.invalid) return;
            const heights = {};
            for (const { target } of records) {
                const height = target.style.height.endsWith('px') ? parseFloat(target.style.height) : NaN;
                const field = target.dataset.field;
                if (Number.isFinite(height) && height > 0 && height !== controller.state.editor.textHeights?.[field]) heights[field] = height;
            }
            if (Object.keys(heights).length) controller.edit(state => {
                state.editor.textHeights = { ...state.editor.textHeights, ...heights };
            }, false);
        });
    }
    attach(input, field) {
        input.dataset.field = field;
        const height = this.controller.state.editor.textHeights?.[field];
        input.style.height = Number.isFinite(height) && height > 0 ? `${height}px` : '';
        this.observer.observe(input, { attributes: true, attributeFilter: ['style'] });
    }
    disconnect() { this.observer.disconnect(); }
}
