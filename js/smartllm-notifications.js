import { app } from './comfy/index.js';

const recent = new Map();

export function showSmartLLMToast(summary, detail, severity = 'error', context = '') {
    const message = typeof detail === 'string' ? detail : detail?.message;
    if (typeof summary !== 'string' || typeof message !== 'string' || !message.trim()) return;
    if (!['error', 'warn', 'info', 'success'].includes(severity)) return;
    // Keep full local diagnostics even when the toast service is unavailable.
    try {
        console[severity === 'error' ? 'error' : severity === 'warn' ? 'warn' : 'info'](
            `[SmartLLM Nova] ${summary}:`, detail);
    } catch { /* Diagnostics cannot replace the original failure. */ }

    const text = message.length > 1200 ? `${message.slice(0, 1200)}…` : message;
    const title = summary.slice(0, 200);
    const key = JSON.stringify([context, severity, title, text]);
    const now = Date.now();
    // Execution notices are shown once per prompt/node; UI duplicates are brief.
    if (recent.has(key) && (context || now - recent.get(key) < 3000)) return;
    try {
        const toast = app.extensionManager?.toast;
        if (typeof toast?.add !== 'function') return;
        recent.delete(key);
        recent.set(key, now);
        if (recent.size > 128) recent.delete(recent.keys().next().value);
        toast.add({ severity, summary: title, detail: text,
            life: severity === 'error' ? 12000 : severity === 'warn' ? 10000 : 6000 });
    } catch {
        recent.delete(key);
    }
}

