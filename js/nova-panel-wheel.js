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
