/** Small DOM controls for SmartLLM Nova panels. Jade/Director design adapted from DaSiWa. */
export function ensurePanelStyle() {
    if (document.getElementById('smartllm-nova-panel-style')) return;
    const style = document.createElement('link'); style.id = 'smartllm-nova-panel-style'; style.rel = 'stylesheet';
    style.href = new URL('./nova-panel.css', import.meta.url).href; document.head.append(style);
}
export function element(tag, className = '', text = '') {
    const item = document.createElement(tag); item.className = className; item.textContent = text; return item;
}
export function button(text, title, action) {
    const item = element('button', '', text); item.type = 'button'; item.title = title;
    item.setAttribute('aria-label', title); item.onclick = action; return item;
}

const MODEL_GROUPS = {
    loader: { __SEP__VISION_MODELS__: 'Vision models', __SEP__TEXT_MODELS__: 'Text models', __SEP__WD14_MODELS__: 'WD14 models' },
    detection: { __SEP__DETECTION_VLM__: 'VLM models', __SEP__YOLO__: 'YOLO models' },
};

// Schema lists carry separator tokens; the loader refresh endpoint carries
// registry objects. Keep both paths in the original category/name order.
export function groupedModelOptions(values, kind) {
    const labels = MODEL_GROUPS[kind], groups = new Map(Object.keys(labels).map(key => [key, []]));
    groups.set('', []);
    let group = '';
    for (const value of values ?? []) {
        if (typeof value === 'string') {
            if (Object.hasOwn(labels, value)) { group = value; continue; }
            if (value && !value.startsWith('__SEP__') && !value.startsWith('─')) groups.get(group).push(value);
        } else if (value?.display_name) {
            if (kind === 'loader' && value.backend === 'yolo') continue;
            const key = kind === 'detection' ? (value.backend === 'yolo' ? '__SEP__YOLO__' : '__SEP__DETECTION_VLM__')
                : value.backend === 'wd14' ? '__SEP__WD14_MODELS__' : value.has_vision ? '__SEP__VISION_MODELS__' : '__SEP__TEXT_MODELS__';
            groups.get(key).push(value.display_name);
        }
    }
    return [...groups].filter(([, names]) => names.length).map(([key, names]) => ({ label: labels[key] ?? '', options: [...new Set(names)].sort() }));
}

export class SearchMenu {
    field(parent, name, current, options, change) {
        const field = element('div', 'sln-panel-field'); field.append(element('span', '', name));
        const available = options.some(option => typeof option === 'string' ? option === current : option.options.includes(current));
        const label = available ? current : `${current} (unavailable)`;
        const trigger = button('', name, () => this.trigger === trigger ? this.close(true) : this.open(trigger, name, current, options, change));
        trigger.className = 'sln-panel-select'; trigger.setAttribute('role', 'combobox');
        trigger.setAttribute('aria-expanded', 'false'); trigger.setAttribute('aria-haspopup', 'listbox'); trigger.title = label;
        trigger.append(element('span', '', label), element('span', '', '▾'));
        trigger.onkeydown = event => {
            if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); this.open(trigger, name, current, options, change); }
        };
        field.append(trigger); parent.append(field); return trigger;
    }
    open(trigger, name, current, options, change, label = value => value) {
        this.close();
        const menu = element('div', 'sln-panel-menu'); menu.popover = 'manual';
        const search = element('input'); search.type = 'search'; search.placeholder = `Search ${name.toLowerCase()}…`;
        search.setAttribute('aria-label', `Search ${name.toLowerCase()}`); search.autocomplete = 'off';
        const list = element('div', 'sln-panel-results'); list.id = `sln-options-${crypto.randomUUID()}`;
        list.setAttribute('role', 'listbox'); list.setAttribute('aria-label', name);
        const choices = [], groups = [];
        const addChoice = (parent, value) => {
            const choice = button(label(value), label(value), () => { this.close(true); change(value); });
            choice.setAttribute('role', 'option'); choice.setAttribute('aria-selected', String(value === current));
            choice.tabIndex = -1; parent.append(choice); choices.push(choice); return choice;
        };
        for (const option of options) {
            if (typeof option === 'string') { addChoice(list, option); continue; }
            const group = element('div', 'sln-panel-option-group');
            if (option.label) {
                group.setAttribute('role', 'group'); group.setAttribute('aria-label', option.label);
                const heading = element('div', 'sln-panel-group-title', option.label); heading.setAttribute('aria-hidden', 'true'); group.append(heading);
            }
            const members = option.options.map(value => addChoice(group, value));
            groups.push({ group, members }); list.append(group);
        }
        const empty = element('div', 'sln-panel-empty', 'No matching options'); empty.setAttribute('role', 'status'); empty.hidden = true;
        menu.append(search, list, empty); document.body.append(menu); menu.showPopover();
        this.menu = menu; this.trigger = trigger;
        trigger.setAttribute('aria-expanded', 'true'); trigger.setAttribute('aria-controls', list.id);
        let visible = choices;
        const position = () => {
            const rect = trigger.getBoundingClientRect(), scale = rect.width / trigger.offsetWidth || 1;
            menu.style.width = `${Math.min(Math.max(trigger.offsetWidth, 200), (innerWidth - 16) / scale)}px`;
            menu.style.maxHeight = `${Math.min(340, (innerHeight - 16) / scale)}px`;
            menu.style.transform = `scale(${scale})`;
            const height = menu.offsetHeight * scale, width = menu.offsetWidth * scale;
            menu.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - width - 8))}px`;
            menu.style.top = `${Math.max(8, Math.min(rect.bottom + height + 4 < innerHeight ? rect.bottom + 4 : rect.top - height - 4, innerHeight - height - 8))}px`;
        };
        search.oninput = () => {
            const query = search.value.trim().toLowerCase();
            visible = choices.filter(choice => { choice.hidden = !choice.textContent.toLowerCase().includes(query); return !choice.hidden; });
            for (const { group, members } of groups) group.hidden = members.every(choice => choice.hidden);
            empty.hidden = visible.length > 0; list.scrollTop = 0; position();
        };
        menu.onkeydown = event => {
            event.stopPropagation(); if (event.isComposing) return;
            if (event.key === 'Escape') { event.preventDefault(); this.close(true); return; }
            if (event.key === 'Tab') { this.close(true); return; }
            if (event.target === search) {
                if (event.key === 'Enter') { event.preventDefault(); visible[0]?.click(); }
                else if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
                    event.preventDefault(); (event.key === 'ArrowDown' ? visible[0] : visible.at(-1))?.focus();
                }
                return;
            }
            if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key) || !visible.length) return;
            event.preventDefault(); const index = visible.indexOf(document.activeElement);
            if (event.key === 'ArrowUp' && index === 0) { search.focus(); return; }
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? visible.length - 1
                : (index + (event.key === 'ArrowDown' ? 1 : -1) + visible.length) % visible.length;
            visible[next].focus();
        };
        menu.onpointerdown = event => event.stopPropagation();
        menu.addEventListener('wheel', event => event.stopPropagation(), { passive: true });
        position(); choices.find(choice => choice.getAttribute('aria-selected') === 'true')?.scrollIntoView({ block: 'nearest' });
        search.focus({ preventScroll: true });
        this.events = new AbortController(); const signal = this.events.signal;
        document.addEventListener('pointerdown', event => { if (!menu.contains(event.target) && !trigger.contains(event.target)) this.close(); }, { capture: true, signal });
        document.addEventListener('wheel', event => { if (!menu.contains(event.target)) this.close(); }, { passive: true, capture: true, signal });
        window.addEventListener('resize', () => this.close(), { signal });
        this.observer = new MutationObserver(() => this.close());
        for (let parent = trigger.parentElement; parent; parent = parent.parentElement) {
            this.observer.observe(parent, { childList: true, attributes: true, attributeFilter: ['style', 'class'] });
        }
    }
    close(focus = false) {
        this.events?.abort(); this.events = null; this.observer?.disconnect(); this.observer = null;
        if (this.menu) { this.menu.hidePopover(); this.menu.remove(); this.menu = null; }
        if (this.trigger) {
            this.trigger.setAttribute('aria-expanded', 'false'); this.trigger.removeAttribute('aria-controls');
            if (focus && this.trigger.isConnected) this.trigger.focus({ preventScroll: true });
        }
        this.trigger = null;
    }
}
