import { app, api } from './comfy/index.js';

app.registerExtension({
    name: 'SmartLLMNova.NodeTitleUpdater',
    async setup(appRef) {
        api.addEventListener('smartllm-nova/update_node_title', (event) => {
            const detail = event.detail;
            if (!detail) return;
            const node = appRef.graph.getNodeById(detail.node_id);
            if (node) {
                node.title = detail.title;
                appRef.canvas.draw(true, true);
            }
        });
    },
});
