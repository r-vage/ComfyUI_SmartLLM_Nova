/** Submission-time dependency scope shared by Nova's queue hooks. */
const localSeedInputs = new Map();

// Match the seed_input removal performed by the panel hook, regardless of
// extension installation order. A locally ignored seed is not a dependency.
export function registerLocalSeedInput(nodeId, usesSeed) {
    const previous = localSeedInputs.get(nodeId);
    localSeedInputs.set(nodeId, usesSeed);
    return () => {
        if (localSeedInputs.get(nodeId) !== usesSeed) return;
        if (previous) localSeedInputs.set(nodeId, previous);
        else localSeedInputs.delete(nodeId);
    };
}

export function executionScope(prompt, options) {
    const targets = options?.partialExecutionTargets;
    if (!Array.isArray(targets)) return null; // Full queues retain their existing behavior.
    const output = prompt?.output ?? {}, selected = new Set(), pending = targets.map(String);
    while (pending.length) {
        const id = pending.pop(), node = output[id];
        if (!node || selected.has(id)) continue;
        selected.add(id);
        let localSeed = false;
        const usesSeed = localSeedInputs.get(node.class_type);
        if (usesSeed) {
            try {
                const config = JSON.parse(node.inputs?.configuration);
                localSeed = config.seed?.source === 'local' && usesSeed(config);
            } catch { /* Let ComfyUI report invalid configurations on the requested branch. */ }
        }
        for (const [name, value] of Object.entries(node.inputs ?? {})) {
            if (name === 'seed_input' && localSeed) continue;
            if (Array.isArray(value) && value.length === 2 && typeof value[0] === 'string'
                && Number.isInteger(value[1]) && value[1] >= 0 && Object.hasOwn(output, value[0])) pending.push(value[0]);
        }
    }
    return selected;
}

export function queueEntries(prompt, options, accepts) {
    const scope = executionScope(prompt, options);
    return Object.entries(prompt?.output ?? {}).filter(([id, node]) => (!scope || scope.has(id)) && accepts(node));
}

