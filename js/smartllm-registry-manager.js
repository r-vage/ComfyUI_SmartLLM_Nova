import { showSmartLLMToast } from './smartllm-notifications.js';
import { app, api } from './comfy/index.js';
import { emitSmartLLMRegistryChanged } from './smartllm-registry-events.js';

const COMMAND_ID = 'SmartLLMNova.LMRegistry.Open';
const SIDEBAR_TAB_ID = 'smartllm-nova-lm-registry';
const CSS_ID = 'smartllm-nova-registry-manager-css';
const BACKENDS = ['transformers', 'gguf', 'llamacpp', 'ollama', 'vllm', 'vllm_native', 'sglang', 'wd14', 'yolo'];
const FAMILIES = ['Qwen', 'Mistral', 'Florence', 'LLaVA', 'LLM_TEXT', 'VLM', 'WD14', 'YOLO'];
const SERVER_QUANTIZATIONS = {
    vllm: ['auto', 'fp8', 'awq', 'gptq', 'bitsandbytes'],
    vllm_native: ['auto', 'fp8', 'awq', 'gptq', 'bitsandbytes', 'squeezellm'],
    sglang: ['auto', 'fp8', 'awq', 'gptq'],
};

function injectCSS() {
    if (document.getElementById(CSS_ID)) return;
    const stylesheet = document.createElement('link');
    stylesheet.id = CSS_ID;
    stylesheet.rel = 'stylesheet';
    stylesheet.href = new URL('./smartllm-registry-manager.css', import.meta.url).href;
    document.head.appendChild(stylesheet);
}

async function request(path, body = null) {
    const response = await api.fetchApi(path, body === null ? undefined : {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    let payload;
    try { payload = await response.json(); } catch (_) { payload = {}; }
    if (!response.ok || payload?.success === false) {
        throw new Error(payload?.error || `Request failed (${response.status})`);
    }
    return payload;
}

function element(tag, className = '', text = '') {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
}

function section(form, name, title, description, open = true) {
    const wrapper = element('details', 'smartllm-nova-registry-section');
    wrapper.dataset.section = name;
    wrapper.open = open;
    const summary = element('summary', '', title);
    const body = element('div', 'smartllm-nova-registry-section-body');
    if (description) body.appendChild(element('p', 'smartllm-nova-registry-section-note', description));
    wrapper.append(summary, body);
    form.appendChild(wrapper);
    return body;
}

function badge(text, kind = '') {
    const node = element('span', 'smartllm-nova-registry-badge', text);
    node.dataset.kind = kind;
    return node;
}

function field(form, id, labelText, type = 'text', options = {}) {
    const wrap = element('div', `smartllm-nova-registry-field${options.wide ? ' smartllm-nova-registry-field--wide' : ''}${type === 'checkbox' ? ' smartllm-nova-registry-field--check' : ''}`);
    wrap.dataset.field = id;
    if (options.backends) wrap.dataset.backends = options.backends.join(',');
    const label = element('label', '', labelText);
    label.htmlFor = `smartllm-nova-registry-${id}`;
    let input;
    if (type === 'select') {
        input = document.createElement('select');
        for (const value of options.values || []) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            input.appendChild(option);
        }
    } else if (type === 'textarea') {
        input = document.createElement('textarea');
    } else {
        input = document.createElement('input');
        input.type = type;
    }
    input.id = `smartllm-nova-registry-${id}`;
    input.name = id;
    input.dataset.testid = `smartllm-nova-registry-${id}`;
    if (options.placeholder) input.placeholder = options.placeholder;
    if (type === 'checkbox') {
        wrap.append(input, label);
    } else {
        wrap.append(label, input);
    }
    if (options.help) {
        const help = element('p', 'smartllm-nova-registry-field-help', options.help);
        help.id = `${input.id}-help`;
        input.setAttribute('aria-describedby', help.id);
        wrap.appendChild(help);
    }
    form.appendChild(wrap);
    return input;
}

class RegistryManager {
    constructor() {
        this.models = [];
        this.selected = null;
        this.originalFocus = null;
        this.busy = false;
        this.inputs = {};
        this.activeView = 'models';
        this.dockerLoaded = false;
        this.dockerState = null;
        this.viewStatus = {};
    }

    open() {
        if (this.backdrop?.isConnected) return;
        injectCSS();
        this.originalFocus = document.activeElement;
        this.build();
        document.body.appendChild(this.backdrop);
        if (this.activeView === 'models') this.filter.focus();
        this.refreshActive();
    }

    close() {
        if (this.busy) return;
        this.backdrop?.remove();
        this.backdrop = null;
        if (this.originalFocus instanceof HTMLElement) this.originalFocus.focus();
    }

    build() {
        this.pendingSelection = this.selected;
        this.selected = null;
        this.modelsLoaded = false;
        const backdrop = element('div', 'smartllm-nova-registry-backdrop');
        backdrop.dataset.testid = 'smartllm-nova-registry-manager';
        const dialog = element('section', 'smartllm-nova-registry-dialog');
        dialog.setAttribute('role', 'dialog');
        dialog.setAttribute('aria-modal', 'true');
        dialog.setAttribute('aria-labelledby', 'smartllm-nova-registry-title');
        this.dialog = dialog;
        backdrop.appendChild(dialog);
        backdrop.addEventListener('mousedown', event => {
            if (event.target === backdrop && !this.busy) this.close();
        });
        backdrop.addEventListener('keydown', event => {
            // Keep graph shortcuts outside the modal while editing or navigating it.
            event.stopPropagation();
            if (event.key === 'Escape' && !this.busy) this.close();
            if (event.key === 'Tab') {
                const controls = [...dialog.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, a[href]')]
                    .filter(control => control.getClientRects().length && control.tabIndex >= 0);
                const target = event.shiftKey ? controls.at(-1) : controls[0];
                if (target && document.activeElement === (event.shiftKey ? controls[0] : controls.at(-1))) {
                    event.preventDefault();
                    target.focus();
                }
            }
        });

        const header = element('header', 'smartllm-nova-registry-header');
        const heading = element('div', 'smartllm-nova-registry-heading');
        const title = element('h2', '', 'SmartLLM Nova Manager (Beta)');
        title.id = 'smartllm-nova-registry-title';
        heading.append(title, element('p', 'smartllm-nova-registry-caption', 'Model sources, local files and Docker runtimes.'));
        this.refreshButton = this.button('Refresh', () => this.refreshActive(), 'smartllm-nova-registry-refresh');
        this.addButton = this.button('New Model', () => this.newEntry(), 'smartllm-nova-registry-new');
        const close = this.button('Close', () => this.close(), 'smartllm-nova-registry-close');
        header.append(heading, this.refreshButton, this.addButton, close);

        const tabs = element('div', 'smartllm-nova-manager-tabs');
        tabs.setAttribute('role', 'tablist');
        this.modelsTab = this.button('Models', () => this.switchView('models'), 'smartllm-nova-manager-tab-models');
        this.modelsTab.classList.add('smartllm-nova-manager-tab');
        this.modelsTab.setAttribute('role', 'tab');
        this.dockerTab = this.button('Docker Images', () => this.switchView('docker'), 'smartllm-nova-manager-tab-docker');
        this.dockerTab.classList.add('smartllm-nova-manager-tab');
        this.dockerTab.setAttribute('role', 'tab');
        for (const [tab, view] of [[this.modelsTab, 'models'], [this.dockerTab, 'docker']]) {
            tab.id = `smartllm-nova-tab-${view}`;
            tab.setAttribute('aria-controls', `smartllm-nova-view-${view}`);
        }
        tabs.addEventListener('keydown', event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || this.busy) return;
            event.preventDefault();
            const view = event.key === 'Home' ? 'models' : event.key === 'End' ? 'docker' : this.activeView === 'models' ? 'docker' : 'models';
            this.switchView(view);
            (view === 'models' ? this.modelsTab : this.dockerTab).focus();
        });
        tabs.append(this.modelsTab, this.dockerTab);

        const body = element('div', 'smartllm-nova-registry-body');
        body.id = 'smartllm-nova-view-models';
        body.setAttribute('role', 'tabpanel');
        body.setAttribute('aria-labelledby', this.modelsTab.id);
        this.registryBody = body;
        const sidebar = element('aside', 'smartllm-nova-registry-sidebar');
        this.filter = document.createElement('input');
        this.filter.type = 'search';
        this.filter.placeholder = 'Filter registry models';
        this.filter.setAttribute('aria-label', 'Filter registry models');
        this.filter.addEventListener('input', () => this.renderList());
        this.list = element('div', 'smartllm-nova-registry-list');
        this.list.setAttribute('role', 'listbox');
        this.list.setAttribute('aria-label', 'Registry models');
        this.listCount = element('div', 'smartllm-nova-registry-count');
        sidebar.append(this.filter, this.listCount, this.list);

        this.form = element('form', 'smartllm-nova-registry-form');
        this.form.addEventListener('submit', event => { event.preventDefault(); this.save(); });
        const identity = section(this.form, 'model', 'Model', 'Choose the backend that will run this model. A saved entry keeps its backend.');
        const source = section(this.form, 'source', 'Source', 'Inspect resolves the source without saving or downloading. Save Entry stores the configuration.');
        const files = section(this.form, 'files', 'Files & runtime', 'Only settings supported by the selected backend are shown.');
        this.filesSection = files.parentElement;
        const integrity = section(this.form, 'integrity', 'Integrity', 'Optional expected hashes strengthen verification. Leave this empty to use available upstream hashes.', false);
        this.integritySection = integrity.parentElement;
        const notes = section(this.form, 'notes', 'Notes', '', false);
        this.inputs.name = field(identity, 'name', 'Registry name');
        this.inputs.backend = field(identity, 'backend', 'Backend', 'select', { values: BACKENDS });
        this.inputs.family = field(identity, 'family', 'Family', 'select', { values: FAMILIES });
        this.inputs.repo_id = field(source, 'repo_id', 'Repository / model ID', 'text', { wide: true, placeholder: 'owner/model or ollama/model:tag', help: 'Use the repository ID, or the exact Ollama model name and tag.' });
        this.inputs.source = field(source, 'source', 'Download source', 'select', { values: ['huggingface', 'modelscope'] });
        this.inputs.revision = field(source, 'revision', 'Revision', 'text', { placeholder: 'Resolved to an immutable commit on Inspect/Save' });
        this.inputs.local_only = field(source, 'local_only', 'Use an existing local model only', 'checkbox');
        this.inputs.local_path = field(source, 'local_path', 'Local path below configured LLM folder', 'text', { wide: true, placeholder: 'ModelFolder or ModelFolder/model.gguf' });
        this.inputs.has_vision = field(identity, 'has_vision', 'Vision-capable model', 'checkbox');
        this.inputs.filename = field(files, 'filename', 'YOLO checkpoint filename', 'text', { backends: ['yolo'], placeholder: 'model.pt' });
        this.inputs.detection_type = field(files, 'detection_type', 'YOLO detection type', 'select', { backends: ['yolo'], values: ['bbox', 'segm'] });
        this.inputs.file_pattern = field(files, 'file_pattern', 'GGUF file pattern', 'text', { backends: ['gguf', 'llamacpp'], placeholder: 'Model.{quant}.gguf' });
        this.inputs.mmproj = field(files, 'mmproj', 'Vision mmproj filename', 'text', { backends: ['gguf', 'llamacpp'] });
        this.inputs.quantizations = field(files, 'quantizations', 'Quantizations (comma-separated)', 'text', { wide: true, backends: ['gguf', 'llamacpp'] });
        this.inputs.download_quantization = field(files, 'download_quantization', 'Download quantization', 'select', { backends: ['gguf', 'llamacpp'], values: [], help: 'Used by Download and Verify Local Files. Save quantization-list changes first.' });
        this.inputs.quantization = field(files, 'quantization', 'Server quantization', 'select', { backends: ['vllm', 'vllm_native', 'sglang'], values: ['auto', 'fp8', 'awq', 'gptq', 'bitsandbytes', 'squeezellm'] });
        this.inputs.tensor_parallel = field(files, 'tensor_parallel', 'Tensor parallel', 'number', { backends: ['vllm', 'sglang'] });
        this.inputs.data_parallel = field(files, 'data_parallel', 'Data parallel', 'number', { backends: ['sglang'] });
        this.inputs.expected_sha256 = field(integrity, 'expected_sha256', 'Expected SHA-256', 'textarea', { wide: true, placeholder: 'model.safetensors=<64-character SHA-256>', help: 'One filename=digest per line. Use exact repository-relative filenames, including subfolders.' });
        this.inputs.description = field(notes, 'description', 'Description', 'textarea', { wide: true });
        this.inputs.backend.addEventListener('change', () => this.updateVisibility());
        this.inputs.local_only.addEventListener('change', () => this.updateVisibility());
        this.inputs.quantizations.addEventListener('input', () => this.syncDownloadQuantizations());
        body.append(sidebar, this.form);

        this.dockerBody = element('section', 'smartllm-nova-docker-body');
        this.dockerBody.id = 'smartllm-nova-view-docker';
        this.dockerBody.setAttribute('role', 'tabpanel');
        this.dockerBody.setAttribute('aria-labelledby', this.dockerTab.id);
        this.dockerBody.dataset.testid = 'smartllm-nova-docker-manager';
        const dockerToolbar = element('div', 'smartllm-nova-docker-toolbar');
        const vendorLabel = element('label', '', 'Image platform');
        this.dockerVendor = document.createElement('select');
        this.dockerVendor.dataset.testid = 'smartllm-nova-docker-vendor';
        for (const [value, label] of [['auto', 'Auto-detect'], ['nvidia', 'NVIDIA / CUDA'], ['amd', 'AMD / ROCm'], ['cpu', 'CPU only']]) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = label;
            this.dockerVendor.appendChild(option);
        }
        vendorLabel.htmlFor = 'smartllm-nova-docker-vendor';
        this.dockerVendor.id = 'smartllm-nova-docker-vendor';
        this.dockerVendor.addEventListener('change', () => this.refreshDocker());
        dockerToolbar.append(vendorLabel, this.dockerVendor, element('p', 'smartllm-nova-docker-note', 'Select which runtime images to manage. Model weights are managed in Models.'));
        this.dockerContent = element('div', 'smartllm-nova-docker-content');
        this.dockerBody.append(dockerToolbar, this.dockerContent);

        const footer = element('footer', 'smartllm-nova-registry-footer');
        this.status = element('div', 'smartllm-nova-registry-status', 'Select a model or create a new entry.');
        this.status.setAttribute('role', 'status');
        this.status.setAttribute('aria-live', 'polite');
        this.status.setAttribute('aria-atomic', 'true');
        this.inspectButton = this.button('Inspect', () => this.inspect(), 'smartllm-nova-registry-inspect');
        this.saveButton = this.button('Save Entry', () => this.save(), 'smartllm-nova-registry-save');
        this.downloadButton = this.button('Download', () => this.download(), 'smartllm-nova-registry-download');
        this.verifyButton = this.button('Verify Local Files', () => this.verify(), 'smartllm-nova-registry-verify');
        this.deleteButton = this.button('Delete Local Files', () => this.deleteLocal(), 'smartllm-nova-registry-delete-local', true);
        this.removeButton = this.button('Remove Registry Entry', () => this.removeEntry(), 'smartllm-nova-registry-remove-entry', true);
        this.saveButton.classList.add('smartllm-nova-registry-button--primary');
        this.inspectButton.title = 'Validate the form and resolve its source. Does not save an entry or download weights.';
        this.saveButton.title = 'Save this configuration to the registry. Does not download model weights.';
        this.downloadButton.title = 'Download the saved entry and selected quantization. Save edited fields first.';
        this.verifyButton.title = 'Check files for the saved entry without downloading them.';
        this.deleteButton.title = 'Delete this model’s local files while keeping the registry entry.';
        this.removeButton.title = 'Remove this registry entry while keeping local model files.';
        const primaryActions = element('div', 'smartllm-nova-registry-action-group');
        const removalActions = element('div', 'smartllm-nova-registry-action-group');
        primaryActions.append(this.inspectButton, this.saveButton, this.downloadButton, this.verifyButton);
        removalActions.append(this.deleteButton, this.removeButton);
        this.modelActions = element('div', 'smartllm-nova-registry-actions');
        this.modelActions.append(primaryActions, removalActions);
        footer.append(this.status, this.modelActions);
        this.registryActionButtons = [this.inspectButton, this.saveButton, this.downloadButton, this.verifyButton, this.deleteButton, this.removeButton];
        dialog.append(header, tabs, body, this.dockerBody, footer);
        this.backdrop = backdrop;
        this.setForm({});
        this.updateView();
    }

    button(label, handler, testid, danger = false) {
        const button = element('button', `smartllm-nova-registry-button${danger ? ' smartllm-nova-registry-danger' : ''}`, label);
        button.type = 'button';
        button.dataset.testid = testid;
        button.addEventListener('click', handler);
        return button;
    }

    switchView(view) {
        if (!['models', 'docker'].includes(view) || this.busy) return;
        this.activeView = view;
        this.updateView();
        if (view === 'docker' && !this.dockerLoaded) this.refreshDocker();
        else if (view === 'models') {
            this.filter.focus();
            if (!this.modelsLoaded) this.refresh();
        }
    }

    updateView() {
        const showModels = this.activeView === 'models';
        this.registryBody.hidden = !showModels;
        this.dockerBody.hidden = showModels;
        this.addButton.hidden = !showModels;
        this.modelsTab.setAttribute('aria-selected', String(showModels));
        this.dockerTab.setAttribute('aria-selected', String(!showModels));
        this.modelsTab.tabIndex = showModels ? 0 : -1;
        this.dockerTab.tabIndex = showModels ? -1 : 0;
        this.modelActions.hidden = !showModels;
        for (const button of this.registryActionButtons) button.hidden = !showModels;
        const status = this.viewStatus[this.activeView];
        this.status.textContent = status?.message || (showModels ? 'Select a model or create a new entry.' : 'Refresh to inspect Docker setup and runtime images.');
        this.status.dataset.kind = status?.kind || '';
    }

    refreshActive() {
        return this.activeView === 'docker' ? this.refreshDocker() : this.refresh();
    }

    dockerDetail(list, label, value) {
        list.append(element('dt', '', label), element('dd', '', value || '—'));
    }

    renderDocker() {
        this.dockerContent.replaceChildren();
        if (!this.dockerState) {
            this.dockerContent.appendChild(element('p', 'smartllm-nova-docker-note', 'Loading Docker status…'));
            return;
        }

        const installation = this.dockerState.installation || {};
        const docker = installation.docker || {};
        const setup = installation.setup || {};
        const gpu = installation.gpu || {};
        const platform = installation.platform || {};
        const setupCard = element('section', 'smartllm-nova-docker-card');
        setupCard.dataset.testid = 'smartllm-nova-docker-setup';
        const setupHeading = element('div', 'smartllm-nova-docker-card-heading');
        setupHeading.append(element('h3', '', 'Docker Engine'), badge(docker.daemon_accessible ? 'Daemon accessible' : 'Setup required', docker.daemon_accessible ? 'success' : 'warn'));
        setupCard.appendChild(setupHeading);
        setupCard.appendChild(element('p', '', setup.message || 'Docker setup status is unavailable.'));
        const diagnostics = element('details', 'smartllm-nova-docker-diagnostics');
        diagnostics.open = setup.state !== 'ready';
        diagnostics.appendChild(element('summary', '', 'System & GPU details'));
        const details = element('dl', 'smartllm-nova-docker-detail');
        this.dockerDetail(details, 'Platform', [platform.system, platform.release, platform.machine].filter(Boolean).join(' '));
        this.dockerDetail(details, 'Docker CLI', docker.installed ? (docker.version || 'Installed') : 'Not installed');
        this.dockerDetail(details, 'Docker daemon', docker.daemon_accessible ? `Accessible${docker.daemon_version ? ` · ${docker.daemon_version}` : ''}` : 'Not accessible');
        this.dockerDetail(details, 'Selected images', this.dockerState.selected_vendor || 'auto');
        this.dockerDetail(details, 'Detected GPU', gpu.vendor || 'unknown');
        if (gpu.vendor === 'nvidia') {
            this.dockerDetail(details, 'NVIDIA toolkit', gpu.nvidia_container_toolkit ? 'Installed' : 'Not detected');
            this.dockerDetail(details, 'NVIDIA driver', gpu.driver_accessible ? (gpu.devices || []).join('; ') || 'Accessible' : 'Not accessible');
        } else if (gpu.vendor === 'amd') {
            this.dockerDetail(details, 'ROCm devices', gpu.kfd_available && gpu.dri_available ? 'Available' : 'Incomplete');
        }
        diagnostics.appendChild(details);
        setupCard.appendChild(diagnostics);

        if (setup.restart_required) {
            setupCard.appendChild(element('p', 'smartllm-nova-docker-note', 'After changing group membership, log out or reboot and then restart ComfyUI.'));
        }
        if (setup.guide_url) {
            const guide = element('a', 'smartllm-nova-docker-guide', 'Open Linux Docker installation guide');
            guide.href = setup.guide_url;
            guide.target = '_blank';
            guide.rel = 'noopener noreferrer';
            setupCard.appendChild(guide);
        }
        setupCard.appendChild(element('p', 'smartllm-nova-docker-note', 'Installation and privileged service administration remain terminal-only.'));
        this.dockerContent.appendChild(setupCard);

        this.dockerContent.appendChild(element('h3', 'smartllm-nova-docker-images-heading', `Managed images · ${this.dockerState.selected_vendor || 'auto'}`));
        this.dockerContent.appendChild(element('p', 'smartllm-nova-docker-note', 'Install a runtime for its matching model backend. Update / Repair pulls the configured immutable image; it does not follow the latest release.'));
        const imageGrid = element('div', 'smartllm-nova-docker-images');
        for (const image of this.dockerState.images || []) {
            const card = element('article', 'smartllm-nova-docker-image');
            card.dataset.backend = image.backend;
            const cardHeading = element('div', 'smartllm-nova-docker-card-heading');
            cardHeading.append(element('h3', '', image.label), badge(image.installed ? 'Installed' : 'Not installed', image.installed ? 'success' : 'warn'));
            card.appendChild(cardHeading);
            card.appendChild(element('p', 'smartllm-nova-docker-note', image.description));
            const reference = element('details', 'smartllm-nova-docker-reference');
            const referenceSummary = element('summary', '', image.image.split('@')[0]);
            referenceSummary.title = 'Expand to see the complete image reference and immutable digest.';
            const imageReference = element('code', '', image.image);
            reference.append(referenceSummary, imageReference);
            card.appendChild(reference);
            let runtimeVersion = '';
            let runtimeSelect = null;
            let removeButton = null;
            if (image.backend === 'ollama' && (image.runtime_versions || []).length) {
                const runtime = element('div', 'smartllm-nova-docker-runtime');
                const runtimeLabel = element('label', '', 'Ollama runtime version');
                runtimeSelect = document.createElement('select');
                runtimeSelect.dataset.testid = 'smartllm-nova-docker-ollama-version';
                runtimeLabel.htmlFor = 'smartllm-nova-docker-ollama-version';
                runtimeSelect.id = 'smartllm-nova-docker-ollama-version';
                if (!image.runtime_version) {
                    const customOption = document.createElement('option');
                    customOption.value = '';
                    customOption.textContent = 'Custom configured pin';
                    runtimeSelect.appendChild(customOption);
                }
                for (const option of image.runtime_versions) {
                    const versionOption = document.createElement('option');
                    versionOption.value = option.version;
                    versionOption.textContent = option.label;
                    versionOption.dataset.image = option.image;
                    runtimeSelect.appendChild(versionOption);
                }
                runtimeSelect.value = image.runtime_version || '';
                runtimeVersion = runtimeSelect.value;
                runtimeSelect.disabled = this.busy;
                runtimeSelect.addEventListener('change', () => {
                    runtimeVersion = runtimeSelect.value;
                    const selected = Array.from(runtimeSelect.children).find(option => option.value === runtimeVersion);
                    imageReference.textContent = selected?.dataset?.image || image.image;
                    referenceSummary.textContent = imageReference.textContent.split('@')[0];
                    if (removeButton) removeButton.disabled = runtimeVersion !== (image.runtime_version || '') || !docker.daemon_accessible || this.busy;
                });
                runtime.append(runtimeLabel, runtimeSelect);
                card.appendChild(runtime);
                card.appendChild(element('small', 'smartllm-nova-docker-note', 'Install & Select stores the immutable version pin. The managed Ollama container is recreated on its next start; model data remains in its persistent store.'));
            }
            const metadata = [image.size, image.short_id ? `ID ${image.short_id}` : '', image.created ? image.created.slice(0, 10) : ''].filter(Boolean).join(' · ');
            if (metadata) card.appendChild(element('small', 'smartllm-nova-docker-note', metadata));
            const actions = element('div', 'smartllm-nova-docker-image-actions');
            const installLabel = runtimeSelect ? 'Install & Select' : (image.installed ? 'Update / Repair' : 'Install');
            const install = this.button(installLabel, () => this.dockerImageAction('pull', image, runtimeVersion), `smartllm-nova-docker-pull-${image.backend}`);
            install.classList.add('smartllm-nova-registry-button--primary');
            install.disabled = !docker.daemon_accessible || this.busy;
            actions.appendChild(install);
            const stop = this.button('Stop', () => this.dockerImageAction('stop', image), `smartllm-nova-docker-stop-${image.backend}`);
            stop.title = 'Stop SmartLLM-managed containers for this backend. Model data stays on disk.';
            stop.disabled = !docker.daemon_accessible || this.busy;
            actions.appendChild(stop);
            if (image.installed) {
                removeButton = this.button('Remove', () => this.dockerImageAction('remove', image), `smartllm-nova-docker-remove-${image.backend}`, true);
                removeButton.disabled = !docker.daemon_accessible || this.busy;
                actions.appendChild(removeButton);
            }
            card.appendChild(actions);
            imageGrid.appendChild(card);
        }
        if (!imageGrid.childElementCount) imageGrid.appendChild(element('p', '', 'No managed images are available for this platform.'));
        this.dockerContent.appendChild(imageGrid);
    }

    async refreshDocker() {
        const vendor = this.dockerVendor.value || 'auto';
        const result = await this.action('Refreshing Docker overview', () => request(`/smartllm-nova/docker/images?vendor=${encodeURIComponent(vendor)}`));
        if (!result) return;
        this.dockerState = result;
        this.dockerLoaded = true;
        this.renderDocker();
        this.setStatus('Docker overview refreshed.', 'success');
    }

    async dockerImageAction(action, image, runtimeVersion = '') {
        if (action === 'remove' && !window.confirm(`Remove the managed Docker image for ${image.label}?\n\nSmartLLM will refuse if any container still uses it.`)) return;
        if (action === 'stop' && !window.confirm(`Stop SmartLLM-managed ${image.label} container(s)?\n\nSmartLLM will refuse while a model execution is active.`)) return;
        const verb = action === 'pull' ? 'Installing' : (action === 'stop' ? 'Stopping' : 'Removing');
        const payload = {
            backend: image.backend,
            vendor: this.dockerVendor.value || 'auto',
        };
        if (action === 'pull' && runtimeVersion) payload.runtime_version = runtimeVersion;
        const target = action === 'stop' ? 'container(s)' : 'image';
        const result = await this.action(`${verb} ${image.label} ${target}`, () => request(`/smartllm-nova/docker/images/${action}`, {
            ...payload,
        }));
        if (result) await this.refreshDocker();
    }

    setStatus(message, kind = '') {
        this.viewStatus[this.activeView] = { message, kind };
        this.status.textContent = message;
        this.status.dataset.kind = kind;
        if (kind === 'error' || kind === 'warn') showSmartLLMToast('Registry Manager', message, kind);
    }

    setBusy(value) {
        this.busy = value;
        this.dialog.setAttribute('aria-busy', String(value));
        for (const control of this.backdrop.querySelectorAll('button, input, select, textarea')) control.disabled = value;
        if (!value) {
            this.updateVisibility();
            this.updateView();
            if (this.activeView === 'docker') this.renderDocker();
        }
    }

    async action(label, callback) {
        if (this.busy) return;
        this.setBusy(true);
        this.setStatus(label);
        try {
            const result = await callback();
            this.setStatus(`${label} complete.`, 'success');
            return result;
        } catch (error) {
            this.setStatus(error?.message || `${label} failed.`, 'error');
            return null;
        } finally {
            this.setBusy(false);
        }
    }

    async refresh(preferred = this.selected || this.pendingSelection) {
        const result = await this.action('Refreshing registry', () => request('/smartllm-nova/model_list'));
        if (!result) return;
        this.models = Array.isArray(result) ? result.filter(model => model.editable !== false) : [];
        this.modelsLoaded = true;
        this.pendingSelection = null;
        if (!this.models.some(model => model.display_name === this.selected)) {
            this.selected = null;
            this.setForm({});
        }
        this.renderList();
        if (preferred && this.models.some(model => model.display_name === preferred)) {
            await this.select(preferred);
        } else if (!this.selected && this.models.length) {
            await this.select(this.models[0].display_name);
        }
    }

    renderList() {
        const filter = this.filter.value.trim().toLowerCase();
        this.list.replaceChildren();
        for (const model of this.models) {
            if (filter && !`${model.display_name} ${model.backend} ${model.family}`.toLowerCase().includes(filter)) continue;
            const item = element('button', 'smartllm-nova-registry-item');
            item.appendChild(element('strong', '', model.display_name));
            item.type = 'button';
            item.setAttribute('role', 'option');
            item.setAttribute('aria-selected', String(model.display_name === this.selected));
            item.dataset.testid = 'smartllm-nova-registry-list-item';
            item.title = model.display_name;
            item.disabled = this.busy;
            const detail = element('small', '', `${model.backend} · ${model.origin} · ${model.local_status}`);
            item.appendChild(detail);
            item.addEventListener('click', () => this.select(model.display_name));
            this.list.appendChild(item);
        }
        this.listCount.textContent = `${this.list.childElementCount} of ${this.models.length} models`;
        if (!this.list.childElementCount) this.list.appendChild(element('p', '', 'No matching models.'));
    }

    async select(displayName) {
        const entry = await this.action('Loading registry entry', () => request(`/smartllm-nova/model_entry?name=${encodeURIComponent(displayName)}`));
        if (!entry) return;
        this.selected = displayName;
        this.setForm(entry);
        this.renderList();
        this.setStatus(`Editing ${displayName}.`, 'success');
    }

    newEntry() {
        this.selected = null;
        this.setForm({ backend: 'transformers', family: 'Qwen', source: 'huggingface', quantization: 'auto', tensor_parallel: 1, data_parallel: 1 });
        this.renderList();
        this.inputs.name.focus();
        this.setStatus('Creating a user registry entry.');
    }

    setForm(entry) {
        this.inputs.backend.value = entry.backend || "transformers";
        this.syncServerQuantizations();
        for (const [name, input] of Object.entries(this.inputs)) {
            let value = name === 'backend' ? (entry.backend || 'transformers') : entry[name];
            if (name === 'expected_sha256') {
                value = Object.entries(value || {}).map(([file, digest]) => `${file}=${digest}`).join('\n');
            } else if (name === 'quantizations') {
                value = Array.isArray(value) ? value.join(', ') : '';
            } else if (name === 'download_quantization') {
                continue;
            }
            if (input.type === 'checkbox') input.checked = Boolean(value);
            else input.value = value ?? (name === 'source' ? 'huggingface' : '');
        }
        this.inputs.backend.disabled = Boolean(this.selected);
        this.syncDownloadQuantizations();
        this.updateVisibility();
    }

    syncDownloadQuantizations() {
        const select = this.inputs.download_quantization;
        const previous = select.value;
        const values = this.inputs.quantizations.value.split(',').map(value => value.trim()).filter(Boolean);
        select.replaceChildren();
        for (const value of values) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            select.appendChild(option);
        }
        if (values.includes(previous)) select.value = previous;
    }

    syncServerQuantizations() {
        const select = this.inputs.quantization;
        const values = SERVER_QUANTIZATIONS[this.inputs.backend.value] || ['auto'];
        const previous = select.value;
        select.replaceChildren();
        for (const value of values) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            select.appendChild(option);
        }
        select.value = values.includes(previous) ? previous : 'auto';
    }

    updateVisibility() {
        const backend = this.inputs.backend.value;
        const localOnly = this.inputs.local_only.checked;
        const ollama = backend === 'ollama';
        const yolo = backend === 'yolo';
        this.syncServerQuantizations();
        for (const wrap of this.form.querySelectorAll('[data-backends]')) {
            wrap.hidden = !wrap.dataset.backends.split(',').includes(backend);
        }
        this.filesSection.hidden = ![...this.filesSection.querySelectorAll('[data-backends]')].some(wrap => !wrap.hidden);
        this.integritySection.hidden = localOnly || ollama;
        this.inputs.repo_id.closest('.smartllm-nova-registry-field').hidden = localOnly;
        for (const name of ['source', 'revision', 'expected_sha256']) {
            this.inputs[name].closest('.smartllm-nova-registry-field').hidden = localOnly || ollama;
        }
        this.inputs.local_path.closest('.smartllm-nova-registry-field').hidden = !localOnly || yolo;
        if (backend === 'wd14' || yolo) {
            this.inputs.family.value = yolo ? 'YOLO' : 'WD14';
            this.inputs.has_vision.checked = true;
        }
        this.inputs.backend.disabled = Boolean(this.selected) || this.busy;
        this.inputs.family.disabled = yolo || backend === 'wd14' || this.busy;
        this.inputs.has_vision.disabled = yolo || backend === 'wd14' || this.busy;
        if (yolo) {
            this.inputs.source.value = 'huggingface';
        }
        this.downloadButton.disabled = localOnly || this.busy || !this.selected;
        // Ollama owns and verifies its content-addressed model store; there are
        // no registry-managed local files for this endpoint to verify.
        this.verifyButton.disabled = !this.selected || this.busy || ollama;
        this.deleteButton.disabled = !this.selected || this.busy;
        this.removeButton.disabled = !this.selected || this.busy;
    }

    collect() {
        const hashes = {};
        for (const line of this.inputs.expected_sha256.value.split(/\r?\n/)) {
            if (!line.trim()) continue;
            const index = line.indexOf('=');
            if (index < 1) throw new Error('Expected hashes must use filename=digest lines.');
            hashes[line.slice(0, index).trim()] = line.slice(index + 1).trim();
        }
        const entry = {
            name: this.inputs.name.value.trim(), backend: this.inputs.backend.value,
            family: this.inputs.family.value, repo_id: this.inputs.repo_id.value.trim(),
            source: this.inputs.source.value, revision: this.inputs.revision.value.trim(),
            local_only: this.inputs.local_only.checked, local_path: this.inputs.local_path.value.trim(),
            has_vision: this.inputs.has_vision.checked,
            filename: this.inputs.filename.value.trim(), detection_type: this.inputs.detection_type.value,
            file_pattern: this.inputs.file_pattern.value.trim(), mmproj: this.inputs.mmproj.value.trim(),
            quantizations: this.inputs.quantizations.value.split(',').map(value => value.trim()).filter(Boolean),
            quantization: this.inputs.quantization.value,
            tensor_parallel: Number(this.inputs.tensor_parallel.value || 1), data_parallel: Number(this.inputs.data_parallel.value || 1),
            expected_sha256: hashes, description: this.inputs.description.value.trim(),
        };
        if (entry.backend === 'ollama') {
            delete entry.source;
            delete entry.revision;
            delete entry.expected_sha256;
        }
        return entry;
    }

    async inspect() {
        let entry;
        try { entry = this.collect(); } catch (error) { this.setStatus(error.message, 'error'); return; }
        const result = await this.action('Inspecting model source', () => request('/smartllm-nova/registry/inspect', { entry, original_display_name: this.selected }));
        if (result?.entry) {
            const selected = this.selected;
            this.setForm(result.entry);
            this.selected = selected;
            this.inputs.backend.disabled = Boolean(selected);
            this.setStatus(`Validated as ${result.entry.display_name}.`, 'success');
        }
    }

    async save() {
        let entry;
        try { entry = this.collect(); } catch (error) { this.setStatus(error.message, 'error'); return; }
        const result = await this.action('Saving registry entry', () => request('/smartllm-nova/registry/upsert', { entry, original_display_name: this.selected }));
        const displayName = result?.entry?.display_name;
        if (!displayName) return;
        this.selected = displayName;
        emitSmartLLMRegistryChanged();
        await this.refresh(displayName);
    }

    selectedQuantization() {
        return this.inputs.download_quantization.value || null;
    }

    async download() {
        if (!this.selected) return;
        const result = await this.action('Downloading model', () => request('/smartllm-nova/model/download', { display_name: this.selected, quantization: this.selectedQuantization() }));
        if (result) await this.refresh(this.selected);
    }

    async verify() {
        if (!this.selected) return;
        await this.action('Verifying local model files', () => request('/smartllm-nova/model/verify', { display_name: this.selected, quantization: this.selectedQuantization() }));
    }

    async deleteLocal() {
        if (!this.selected) return;
        if (!window.confirm(`Delete local model files for "${this.selected}"?\n\nThe registry entry will remain. This cannot be undone.`)) return;
        const result = await this.action('Deleting local model files', () => request('/smartllm-nova/model/delete', { display_name: this.selected }));
        if (result) await this.refresh(this.selected);
    }

    async removeEntry() {
        if (!this.selected) return;
        const selected = this.selected;
        if (!window.confirm(`Remove registry entry "${selected}"?\n\nLocal model files will not be deleted. This cannot be undone.`)) return;
        const result = await this.action('Removing registry entry', () => request('/smartllm-nova/registry/remove', { display_name: selected }));
        if (!result) return;
        this.selected = null;
        emitSmartLLMRegistryChanged();
        this.newEntry();
        await this.refresh();
    }
}

const manager = new RegistryManager();
let sidebarLauncherRegistered = false;

function isNewMenuActive() {
    try {
        return app.ui?.settings?.getSettingValue?.('Comfy.UseNewMenu') !== 'Disabled';
    } catch (_) { return true; }
}

function injectClassicButton() {
    const existing = document.querySelector('[data-smartllm-nova-registry-classic]');
    if (isNewMenuActive()) { existing?.remove(); return; }
    if (existing?.isConnected) return;
    const host = app.ui?.menuContainer;
    if (!host) return;
    const button = element('button', 'smartllm-nova-registry-classic', 'Open SmartLLM Nova Manager (Beta)');
    button.type = 'button';
    button.dataset.smartllmNovaRegistryClassic = 'true';
    button.addEventListener('click', () => manager.open());
    host.appendChild(button);
}

function registerSidebarLauncher() {
    if (sidebarLauncherRegistered) return true;
    const extensionManager = app.extensionManager;
    if (!extensionManager || typeof extensionManager.registerSidebarTab !== 'function') return false;
    extensionManager.registerSidebarTab({
        id: SIDEBAR_TAB_ID,
        icon: 'pi pi-database',
        title: 'SmartLLM Nova Manager (Beta)',
        tooltip: 'Manage Smart LM models and Docker images (Beta)',
        type: 'custom',
        render: (host) => {
            host.replaceChildren();
            manager.open();
            // This tab is a left-toolbar launcher, not a persistent panel.
            // Close the empty sidebar immediately while leaving the modal open.
            queueMicrotask(() => {
                try {
                    const closeRequest = extensionManager.command?.execute?.(
                        `Workspace.ToggleSidebarTab.${SIDEBAR_TAB_ID}`
                    );
                    Promise.resolve(closeRequest).catch(() => {});
                } catch (_) { /* noop */ }
            });
        },
    });
    sidebarLauncherRegistered = true;
    return true;
}

app.registerExtension({
    name: 'SmartLLMNova.RegistryManager',
    commands: [{
        id: COMMAND_ID,
        label: 'Open SmartLLM Nova Manager (Beta)',
        icon: 'pi pi-database',
        tooltip: 'Manage Smart LM registry models, Docker setup, and backend images (Beta)',
        function: () => manager.open(),
    }],
    menuCommands: [{ path: ['SmartLLM Nova'], commands: [COMMAND_ID] }],
    async init() { injectCSS(); },
    async setup() {
        if (!registerSidebarLauncher()) {
            let tries = 0;
            const timer = setInterval(() => {
                tries += 1;
                if (registerSidebarLauncher() || tries > 20) clearInterval(timer);
            }, 100);
        }
        injectClassicButton();
        queueMicrotask(injectClassicButton);
        setTimeout(injectClassicButton, 250);
        const observer = new MutationObserver(injectClassicButton);
        observer.observe(document.body, { childList: true, subtree: true });
    },
});
