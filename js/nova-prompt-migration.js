const PANELS = new Map([
    ['Smart LM Loader [SmartLLM Nova]', { pipe: 'Smart LM Prompt Pipe [SmartLLM Nova]', type: 'SMARTLLM_LM_SETTINGS', prompts: ['user_prompt', 'system_prompt'] }],
    ['Smart Detection [SmartLLM Nova]', { pipe: 'Smart Detection Prompt Pipe [SmartLLM Nova]', type: 'SMARTLLM_DETECTION_SETTINGS', prompts: ['user_input'] }],
]);
const linkId = link => Array.isArray(link) ? link[0] : link.id;
function target(link, id, slot) {
    if (Array.isArray(link)) { link[3] = id; link[4] = slot; }
    else { link.target_id = id; link.target_slot = slot; }
}
const numeric = value => Number.isSafeInteger(Number(value)) ? Number(value) : 0;

// Rewire saved data before ComfyUI constructs nodes. In particular, subgraph
// construction reconciles ports against the new schema before onConfigure;
// migrating later can pair old link indices with the wrong input names.
export function migrateSerializedPrompts(workflow) {
    const graphs = [workflow, ...(workflow.definitions?.subgraphs ?? [])];
    let nextNode = 0, nextLink = 0;
    for (const graph of graphs) {
        nextNode = Math.max(nextNode, numeric(graph.last_node_id), numeric(graph.state?.lastNodeId));
        nextLink = Math.max(nextLink, numeric(graph.last_link_id), numeric(graph.state?.lastLinkId));
        for (const node of graph.nodes ?? []) nextNode = Math.max(nextNode, numeric(node.id));
        for (const link of graph.links ?? []) nextLink = Math.max(nextLink, numeric(linkId(link)));
    }
    let changed = false;
    for (const graph of graphs) {
        const links = new Map((graph.links ?? []).map(link => [linkId(link), link]));
        for (const node of [...(graph.nodes ?? [])]) {
            const schema = PANELS.get(node.type);
            if (!schema || !node.inputs?.some(input => schema.prompts.includes(input.name))) continue;
            const retired = node.inputs.filter(input => schema.prompts.includes(input.name));
            // Let ComfyUI report a genuinely broken saved connection, without
            // silently discarding a wire we could not migrate.
            if (retired.some(input => input.link != null && !links.has(input.link))) continue;
            const prompts = retired.filter(input => input.link != null);
            const remaining = node.inputs.filter(input => !schema.prompts.includes(input.name));
            let pipe = remaining.find(input => input.name === 'pipe_opt');
            if (!pipe) {
                pipe = { name: 'pipe_opt', type: schema.type, link: null };
                const configuration = remaining.findIndex(input => input.name === 'configuration');
                remaining.splice(configuration < 0 ? remaining.length : configuration, 0, pipe);
            }
            if (prompts.length) {
                if (pipe.link != null && !links.has(pipe.link)) continue;
                const id = ++nextNode, outputLink = ++nextLink;
                const inputs = [
                    ...schema.prompts.map(name => ({ name, type: 'STRING', link: null })),
                    { name: 'pipe_opt', type: schema.type, link: pipe.link ?? null },
                ];
                for (const input of prompts) {
                    const index = schema.prompts.indexOf(input.name);
                    inputs[index].link = input.link; target(links.get(input.link), id, index);
                }
                if (pipe.link != null) target(links.get(pipe.link), id, schema.prompts.length);
                const slot = remaining.indexOf(pipe);
                const link = graph.inputNode
                    ? { id: outputLink, origin_id: id, origin_slot: 0, target_id: node.id, target_slot: slot, type: schema.type }
                    : [outputLink, id, 0, node.id, slot, schema.type];
                graph.links ??= []; graph.links.push(link); links.set(outputLink, link);
                pipe.link = outputLink;
                graph.nodes.push({
                    id, type: schema.pipe, pos: [(node.pos?.[0] ?? 0) - 290, node.pos?.[1] ?? 0],
                    size: [260, 100], flags: {}, order: 0, mode: 0, inputs,
                    outputs: [{ name: 'settings', type: schema.type, links: [outputLink] }],
                    properties: { 'Node name for S&R': schema.pipe }, widgets_values: [],
                });
            }
            node.inputs = remaining;
            for (const [slot, input] of remaining.entries()) {
                const link = links.get(input.link);
                if (link) target(link, node.id, slot);
            }
            changed = true;
        }
    }
    if (changed) {
        workflow.last_node_id = nextNode; workflow.last_link_id = nextLink;
        for (const graph of graphs) if (graph.state) {
            graph.state.lastNodeId = Math.max(numeric(graph.state.lastNodeId), nextNode);
            graph.state.lastLinkId = Math.max(numeric(graph.state.lastLinkId), nextLink);
        }
    }
    return changed;
}
