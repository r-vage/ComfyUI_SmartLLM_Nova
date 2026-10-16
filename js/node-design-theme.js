/** Local theme catalog and event-driven application; no cross-pack state. */
export function createNodeDesign({ app, api, settingId, category, route, prefix, dataset,
    suffix, notice, nodeCategory = () => null, colorTitlesOnCreate = false }) {
    const defaultDesign = 'neutral';
    let designs = {}, active = null, saved = defaultDesign, ready = false, synchronizing = false;
    let revision = 0, pending = Promise.resolve();
    const appliedTitles = new WeakMap();
    const decoratedTypes = new WeakSet();
    const normalize = value => typeof value === 'string' ? value.toLowerCase().replace(/^#([0-9a-f])([0-9a-f])([0-9a-f])$/, '#$1$1$2$2$3$3') : value;
    const ownedId = node => [node.comfyClass, node.type, node.constructor?.type]
        .find(id => typeof id === 'string' && id.endsWith(suffix));

    function applyNodeDesign(node, created = false) {
        const id = ownedId(node);
        if (!id || !active) return false;
        const kind = nodeCategory(id);
        const titleFor = design => design.node.category_titles[kind] ?? design.node.title;
        const title = titleFor(active), current = normalize(node.color);
        // Keep explicit title overrides; backgrounds always follow the selected theme.
        if ((created && colorTitlesOnCreate) || !current || current === normalize(appliedTitles.get(node))
            || current === normalize(globalThis.LiteGraph?.NODE_DEFAULT_COLOR)
            || Object.values(designs).some(design => current === normalize(titleFor(design)))) {
            node.color = title;
            appliedTitles.set(node, title);
        }
        if (created && colorTitlesOnCreate) node.shape = 'default';
        node.bgcolor = kind === 'pipe' || kind === 'router' ? active.node.routing_background : active.node.background;
        return true;
    }

    function applyGraphDesign(graph) {
        if (!graph) return;
        const graphs = new Set(), nodes = new Set();
        const visit = current => {
            if (!current || graphs.has(current)) return;
            graphs.add(current);
            for (const node of current._nodes ?? []) {
                if (!nodes.has(node)) { nodes.add(node); applyNodeDesign(node); }
                if (node.subgraph) visit(node.subgraph);
            }
            for (const child of current.subgraphs?.values() ?? []) visit(child);
        };
        visit(graph.rootGraph ?? graph);
        graph.setDirtyCanvas?.(true, true);
    }

    function installNodeDesignHooks(nodeTypes) {
        // Frontend-only nodes do not receive ComfyUI's server-node creation hook.
        // Decorate owned classes once, instead of watching graphs for additions.
        for (const [id, type] of Object.entries(nodeTypes ?? {})) {
            const prototype = type?.prototype;
            if (!id.endsWith(suffix) || !prototype || decoratedTypes.has(prototype)) continue;
            decoratedTypes.add(prototype);
            const original = prototype.onAdded;
            prototype.onAdded = function (...args) {
                const result = original?.apply(this, args);
                applyNodeDesign(this);
                return result;
            };
        }
    }

    function applyDesign(id, graph = app.graph) {
        if (!Object.hasOwn(designs, id)) throw Error('Unknown node design.');
        active = designs[id];
        const root = document.documentElement;
        for (const [key, value] of Object.entries({ ...active.tokens, 'node-bg': active.node.background })) {
            root.style.setProperty(`${prefix}${key}`, value);
        }
        root.dataset[dataset] = id;
        applyGraphDesign(graph);
    }

    function applySaved() {
        if (Object.hasOwn(designs, saved)) return applyDesign(saved);
        if (Object.hasOwn(designs, defaultDesign)) applyDesign(defaultDesign);
        notice(`Theme “${saved}” is unavailable. Restore its JSON file or select another design. The saved selection has not been changed.`);
    }

    const options = () => {
        const choices = Object.entries(designs).sort(([a], [b]) => a === defaultDesign ? -1 : b === defaultDesign ? 1 : a.localeCompare(b))
            .map(([value, design]) => ({ text: design.label, value }));
        if (!Object.hasOwn(designs, saved)) choices.push({ text: `Unavailable: ${saved}`, value: saved });
        return choices;
    };
    async function showSaved() {
        synchronizing = true;
        try { await app.ui.settings.setSettingValueAsync(settingId, saved); }
        catch (error) { console.error(`[${category[0]}] Could not synchronize the design selector:`, error); }
        finally { synchronizing = false; }
    }
    async function readCatalog() {
        const response = await api.fetchApi(`${route}/themes`, { cache: 'no-store' });
        if (!response.ok) throw Error('Could not read theme files.');
        const catalog = await response.json();
        if (!catalog.themes || !Object.keys(catalog.themes).length) throw Error('No valid themes found. Restore the bundled themes folder.');
        designs = catalog.themes;
        if (catalog.errors?.length) notice(catalog.errors.join('\n'));
    }
    function reload() {
        // Serialize with saves so a reload cannot replace a newer user selection.
        pending = pending.then(async () => {
            try {
                await readCatalog();
                const definition = app.ui.settings.settingsLookup[settingId];
                definition.options = options();
                applySaved();
                await showSaved();
            } catch (error) { notice(error.message); }
        });
        return pending;
    }
    async function initialize(config = {}) {
        saved = typeof config.node_design === 'string' ? config.node_design : defaultDesign;
        try { await readCatalog(); applySaved(); }
        catch (error) { notice(error.message); }
        const settings = app.ui.settings;
        settings.addSetting({
            id: settingId, category: [...category, 'NodeDesign'], name: 'Node design',
            type: 'combo', defaultValue: saved, options: options(), sortOrder: 500,
            tooltip: 'Theme from this repo’s themes folder. Saved in its config.json. Selecting or reloading a theme replaces saved/custom node backgrounds, including nodes in subgraphs. Node titles keep their category or custom colors.',
            onChange(value) {
                if (!ready || synchronizing) return;
                if (!Object.hasOwn(designs, value)) return showSaved();
                const request = ++revision;
                pending = pending.then(async () => {
                    try {
                        const response = await api.fetchApi(`${route}/config/update`, {
                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ node_design: value }),
                        });
                        const result = await response.json();
                        if (!response.ok || !result.success) throw Error(result.error || 'Could not save node design.');
                        saved = value;
                        if (request === revision) applyDesign(saved);
                    } catch (error) {
                        if (request === revision) {
                            applySaved(); await showSaved();
                            notice(`Could not save the design: ${error.message}`);
                        }
                    }
                });
                return pending;
            },
        });
        settings.addSetting({
            id: `${settingId}.Reload`, category: [...category, 'ReloadThemes'],
            name: 'Reload theme files', defaultValue: '', sortOrder: 499,
            tooltip: 'Read edits and new JSON files in this repo’s themes/custom folder, then reapply the selected theme.',
            type() {
                const button = document.createElement('button');
                button.className = 'comfy-menu-btn'; button.textContent = 'Reload themes';
                button.type = 'button';
                button.onclick = async () => {
                    button.disabled = true;
                    try { await reload(); } finally { button.disabled = false; }
                };
                return button;
            },
        });
        // The repo config wins over stale ComfyUI/browser settings on startup.
        await showSaved(); ready = true;
    }
    return { initialize, reload, applyDesign, applyNodeDesign, applyGraphDesign, installNodeDesignHooks };
}
