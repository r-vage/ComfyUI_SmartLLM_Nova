import { app, api } from './comfy/index.js';
import { showSmartLLMToast } from './smartllm-notifications.js';

const ownedNodes = new Map();
let installed = false;
let validationAttempt = 0;

function showError(type, id, message, promptId = '') {
    const name = ownedNodes.get(type);
    if (!name) return;
    showSmartLLMToast(id == null ? name : `${name} · ${id}`, message, 'error', promptId);
}

function reportValidation(response, prompt, attempt) {
    for (const [id, failure] of Object.entries(response?.node_errors ?? {})) {
        const type = failure?.class_type ?? prompt?.[id]?.class_type;
        if (!ownedNodes.has(type)) continue;
        const errors = Array.isArray(failure?.errors) ? failure.errors : [];
        const message = errors.map(error => [error?.message, error?.details]
            .filter(value => typeof value === 'string' && value).join(': '))
            .filter(Boolean).join('\n');
        showError(type, id, message || 'Input validation failed. See the node error details.',
            `validation:${attempt}`);
    }
}

app.registerExtension({
    name: 'SmartLLMNova.ErrorNotifications',
    beforeRegisterNodeDef(_nodeType, nodeData) {
        // Ownership is scoped to the Nova Python module, independent of display names.
        if (/(?:^|\.)comfyui_smartllm_nova(?:\.|$)/i.test(nodeData.python_module ?? '')) {
            ownedNodes.set(nodeData.name, nodeData.display_name || nodeData.name);
        } else {
            ownedNodes.delete(nodeData.name);
        }
    },
    setup() {
        if (installed) return;
        installed = true;
        api.addEventListener('execution_error', ({ detail }) => {
            if (!detail) return;
            showError(detail.node_type, detail.node_id,
                detail.exception_message || detail.exception_type
                    || 'Execution failed. See the node error details.',
                detail.prompt_id);
        });
        api.addEventListener('smartllm-nova/notification', ({ detail }) => {
            if (!detail || !['warn', 'error'].includes(detail.severity)
                || typeof detail.summary !== 'string' || typeof detail.message !== 'string'
                || !detail.prompt_id || detail.node_id == null) return;
            showSmartLLMToast(`${detail.summary} · ${detail.node_id}`, detail.message,
                detail.severity, `backend:${detail.prompt_id}:${detail.node_id}`);
        });

        const originalQueue = api.queuePrompt;
        api.queuePrompt = async function (number, data, ...options) {
            const attempt = ++validationAttempt;
            try {
                const response = await originalQueue.call(this, number, data, ...options);
                reportValidation(response, data?.output, attempt);
                return response;
            } catch (error) {
                reportValidation(error?.response ?? error, data?.output, attempt);
                throw error;
            }
        };
    },
});

