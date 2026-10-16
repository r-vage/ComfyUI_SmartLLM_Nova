/** SmartLLM Nova's independent, user-editable node designs. */
import { app, api } from './comfy/index.js';
import { showSmartLLMToast } from './smartllm-notifications.js';
import { createNodeDesign } from './node-design-theme.js';

export const DESIGN_SETTING = 'SmartLLMNova.NodeDesign';
const design = createNodeDesign({ app, api, settingId: DESIGN_SETTING,
    category: ['SmartLLM Nova', 'Appearance'], route: '/smartllm-nova', prefix: '--sln-design-',
    dataset: 'smartllmNovaDesign', suffix: ' [SmartLLM Nova]',
    notice: message => showSmartLLMToast('SmartLLM Nova design', message),
});
export const { applyNodeDesign, applyGraphDesign, applyDesign } = design;
export async function initializeNodeDesign(appRef, config) { await design.initialize(config); }
app.registerExtension({
    name: DESIGN_SETTING,
    setup() { design.installNodeDesignHooks(globalThis.LiteGraph?.registered_node_types); },
    nodeCreated(node) { applyNodeDesign(node, true); },
    loadedGraphNode(node) { applyNodeDesign(node); },
    afterConfigureGraph() { applyGraphDesign(app.graph); },
});
