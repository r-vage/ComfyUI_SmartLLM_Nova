import { onSmartLLMRegistryChanged } from './smartllm-registry-events.js';
import { showSmartLLMToast } from './smartllm-notifications.js';

// Pack-local request sharing. Entries are shared only while pending; task and
// model lists retain their existing successful-result cache until refresh.
let generation = 0;
const pending = new Map();
const cached = new Map();

export function invalidateMetadata() {
    generation++;
    pending.clear();
    cached.clear();
}

export function expireMetadataCache() { cached.clear(); }

async function request(path, fallback, cache = false) {
    if (cache && cached.has(path)) return cached.get(path);
    if (pending.has(path)) return pending.get(path);
    const started = generation;
    const promise = (async () => {
        try {
            const response = await fetch(path);
            if (!response.ok) throw new Error(`Metadata request failed (HTTP ${response.status}).`);
            const value = await response.json();
            if (started !== generation) return fallback;
            if (cache) cached.set(path, value);
            return value;
        } catch (error) {
            if (started === generation) showSmartLLMToast('Could not refresh model information', error, 'warn');
            return fallback;
        }
    })();
    pending.set(path, promise);
    try { return await promise; }
    finally { if (pending.get(path) === promise) pending.delete(path); }
}

export function fetchModelEntry(name) {
    return name ? request(`/smartllm-nova/model_entry?name=${encodeURIComponent(name)}`, null) : Promise.resolve(null);
}

export function fetchTaskList(hasVision, family = '') {
    const familyQuery = family ? `&family=${encodeURIComponent(family)}` : '';
    return request(`/smartllm-nova/task_list?has_vision=${hasVision}${familyQuery}`, [], true);
}

const DETECTION_TASKS = {
    Florence: ['Caption to Phrase Grounding', 'Region Caption', 'Dense Region Caption', 'Region Proposal', 'Referring Expression Segmentation', 'OCR With Region', 'DocVQA'],
    Qwen: ['Caption to Phrase Grounding', 'Region Caption'],
};

// Panels and native settings must offer the same tasks for the selected model.
export async function fetchModelTasks(kind, entry) {
    if (['wd14', 'yolo'].includes(entry?.backend)) return [];
    if (kind === 'loader') return fetchTaskList(entry?.has_vision ?? true, entry?.family ?? '');
    return entry?.backend === 'comfyui' && entry.has_vision
        ? DETECTION_TASKS.Qwen : DETECTION_TASKS[entry?.family];
}

export function fetchModelList() { return request('/smartllm-nova/model_list', [], true); }
export function fetchDetectionModelList() { return request('/smartllm-nova/detection/model_list', []); }

onSmartLLMRegistryChanged(invalidateMetadata);
