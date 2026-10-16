export const SMARTLLM_REGISTRY_CHANGED_EVENT = 'smartllm-nova:registry-changed';

export function emitSmartLLMRegistryChanged() {
    window.dispatchEvent(new CustomEvent(SMARTLLM_REGISTRY_CHANGED_EVENT));
}

export function onSmartLLMRegistryChanged(listener) {
    window.addEventListener(SMARTLLM_REGISTRY_CHANGED_EVENT, listener);
    return () => window.removeEventListener(SMARTLLM_REGISTRY_CHANGED_EVENT, listener);
}
