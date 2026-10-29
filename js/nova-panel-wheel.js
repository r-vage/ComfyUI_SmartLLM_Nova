import { app } from './comfy/index.js';

/** Forward canvas navigation blocked by classic and Nodes 2.0 DOM widgets. */
export function forwardCanvasNavigation(root) {
    const pointer = event => {
        if (event.button !== 1 && !(event.buttons & 4)) return;
        const canvas = app.canvas?.canvas;
        if (!canvas || event.defaultPrevented) return;
        event.preventDefault(); event.stopPropagation();
        // Let ComfyUI own the pan gesture and pointer capture, even when the
        // pointer leaves this panel. Primary clicks still belong to controls.
        canvas.dispatchEvent(new PointerEvent(event.type, event));
    };
    const wheel = event => {
        const canvas = app.canvas?.canvas;
        if (!canvas || event.defaultPrevented) return;
        event.preventDefault(); event.stopPropagation();
        canvas.dispatchEvent(new WheelEvent('wheel', event));
    };
    const events = ['pointerdown', 'pointermove', 'pointerup'];
    for (const name of events) root.addEventListener(name, pointer, true);
    // Bubble only: focused fields and scrollable menus can keep their wheel.
    // Vue's ancestor handler already stops events it forwards to the canvas.
    root.addEventListener('wheel', wheel, { passive: false });
    return () => {
        for (const name of events) root.removeEventListener(name, pointer, true);
        root.removeEventListener('wheel', wheel);
    };
}

export function captureFocusedTextWheel(root) {
    let focused = null, previousCapture = null;
    const release = () => {
        if (!focused) return;
        if (previousCapture === null) focused.removeAttribute('data-capture-wheel');
        else focused.setAttribute('data-capture-wheel', previousCapture);
        focused = null;
    };
    const focus = event => {
        release();
        if (event.target.tagName !== 'TEXTAREA') return;
        focused = event.target;
        previousCapture = focused.getAttribute('data-capture-wheel');
        focused.setAttribute('data-capture-wheel', 'true');
    };
    const wheel = event => {
        if (event.target.tagName !== 'TEXTAREA' || root.ownerDocument.activeElement !== event.target) return;
        // Leave explicit zoom gestures to ComfyUI. Ordinary scrolling stays in
        // the focused field, including its edges; the browser scrolls the text.
        if (!event.ctrlKey && !event.metaKey) event.stopPropagation();
    };
    root.addEventListener('focusin', focus);
    root.addEventListener('focusout', release);
    root.addEventListener('wheel', wheel, { capture: true, passive: true });
    if (root.contains(root.ownerDocument.activeElement)) focus({ target: root.ownerDocument.activeElement });
    return () => {
        release();
        root.removeEventListener('focusin', focus);
        root.removeEventListener('focusout', release);
        root.removeEventListener('wheel', wheel, true);
    };
}
