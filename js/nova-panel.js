import { app, api } from './comfy/index.js';
import { ConfigurationPanel } from './nova-configuration-panel.js';
import { ensurePanelStyle } from './nova-panel-controls.js';
import { PanelState } from './nova-panel-state.js';
import { PanelForm } from './nova-panel-form.js';
import { installSeedQueueHook } from './nova-panel-seed-queue.js';
const WIDGET = 'SMARTLLM_NOVA_PANEL', EDITOR = 'SmartLLMNova.panel', controllers = new WeakMap();
const DETECTION_PANEL_ONLY = new Set(['confidence', 'nms_iou_threshold', 'detection_filter']);
function removeDetectionOverrides(node) {
    if (!node.graph) return;
    // Wait until every saved node has restored its links, including seed_input.
    for (let index = node.inputs.length - 1; index >= 0; index--) {
        if (DETECTION_PANEL_ONLY.has(node.inputs[index].name)) node.removeInput(index);
    }
}
app.registerExtension({
    name: 'SmartLLMNova.Panels',
    setup() {
        ensurePanelStyle();
        for (const name of ['Smart LM Loader', 'Smart Detection']) installSeedQueueHook(api, app, controllers, {
            nodeId: `${name} [SmartLLM Nova]`, editorKey: EDITOR,
        });
    },
    getCustomWidgets() {
        return { [WIDGET](node, name, data) {
            const controller = new ConfigurationPanel(node, name, data[1], {
                State: PanelState, Form: PanelForm, widgetType: WIDGET, editorKey: EDITOR,
                className: 'sln-editor', label: `${data[1].editorSpec.title} editor`,
                onDispose: node => { controllers.get(node)?.form.destroy(); controllers.delete(node); },
            });
            controllers.set(node, controller);
            return { widget: controller.widget, minWidth: 400 };
        } };
    },
    beforeRegisterNodeDef(type, data) {
        if (data.input?.required?.configuration?.[1]?.widgetType !== WIDGET) return;
        const configure = type.prototype.onConfigure, added = type.prototype.onAdded,
            connections = type.prototype.onConnectionsChange, executed = type.prototype.onExecuted;
        type.prototype.onConfigure = function (...args) {
            const result = configure?.apply(this, args), controller = controllers.get(this);
            if (controller) { controller.restore(controller.value()); controller.bindHost(); }
            if (data.input.required.configuration[1].editorSpec.kind === 'detection') {
                queueMicrotask(() => removeDetectionOverrides(this));
            }
            return result;
        };
        type.prototype.onAdded = function (...args) {
            const result = added?.apply(this, args);
            controllers.get(this)?.bindHost(); return result;
        };
        type.prototype.onConnectionsChange = function (...args) {
            const result = connections?.apply(this, args), controller = controllers.get(this);
            if (args[0] === 1 && controller && !controller.invalid) {
                if (this.inputs?.[args[1]]?.name === 'seed_input' && !args[2]) controller.edit(state => { state.config.seed.source = 'local'; });
                else controller.form.render();
            }
            return result;
        };
        type.prototype.onExecuted = function (output) {
            const result = executed?.apply(this, arguments), controller = controllers.get(this);
            if (controller && output.nova_seed?.length) controller.edit(state => { state.editor.lastSeed = String(output.nova_seed[0]); });
            return result;
        };
    },
});
