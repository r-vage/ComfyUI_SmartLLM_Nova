import { app } from './comfy/index.js';
import { initializeNodeDesign } from './nova-node-design.js';
import { showSmartLLMToast } from './smartllm-notifications.js';

const SETTINGS_CATEGORY = ['SmartLLM Nova', 'Configuration'];
const TOKEN_MASK = '••••••••';

async function updateConfig(values) {
    const response = await fetch('/smartllm-nova/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
    });
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || 'Configuration update failed');
    return data;
}

async function registerConfigSetting(setting) {
    let hydrating = true;
    const onChange = setting.onChange;
    app.ui.settings.addSetting({
        ...setting,
        onChange(value) {
            if (!hydrating) return onChange?.call(this, value);
        },
    });
    try {
        if (app.ui.settings.setSettingValueAsync) {
            await app.ui.settings.setSettingValueAsync(setting.id, setting.defaultValue);
        } else {
            app.ui.settings.setSettingValue(setting.id, setting.defaultValue);
        }
    } finally {
        hydrating = false;
    }
}

async function registerCredentialSetting({ id, name, key, configured, tooltip, sortOrder }) {
    await registerConfigSetting({
        id,
        category: [...SETTINGS_CATEGORY, key],
        name,
        type: 'text',
        tooltip,
        defaultValue: configured ? TOKEN_MASK : '',
        sortOrder,
        onChange: async (value) => {
            if (value === TOKEN_MASK) return;
            try {
                await updateConfig({ [key]: value });
                app.ui.settings.setSettingValue?.(id, value ? TOKEN_MASK : '');
            } catch (error) {
                showSmartLLMToast('Credential update failed', `${name} could not be saved. Check the server log.`);
            }
        },
    });
}

app.registerExtension({
    name: 'SmartLLMNova.Settings',
    async init() {
        let config = {
            log_level: 'warning',
            llm_models_path: 'LLM',
            retry_download_attempts: 2,
            hf_token_configured: false,
            modelscope_token_configured: false,
        };
        try {
            const response = await fetch('/smartllm-nova/config/all');
            if (!response.ok) throw new Error(`Configuration request failed (HTTP ${response.status}).`);
            config = { ...config, ...(await response.json()) };
        } catch (error) {
            showSmartLLMToast('Could not read settings', error, 'warn');
        }
        await initializeNodeDesign(app, config);

        await registerConfigSetting({
            id: 'SmartLLMNova.ModelsPath',
            category: [...SETTINGS_CATEGORY, 'ModelsPath'],
            name: '📁 LLM Models Path',
            type: 'text',
            tooltip: 'Path to the LLM model folder, relative to ComfyUI models or absolute.',
            defaultValue: config.llm_models_path || 'LLM',
            sortOrder: 400,
            onChange: async (value) => {
                try { await updateConfig({ llm_models_path: value }); }
                catch (error) { showSmartLLMToast('Model path update failed', error); }
            },
        });
        await registerConfigSetting({
            id: 'SmartLLMNova.RetryDownloadAttempts',
            category: [...SETTINGS_CATEGORY, 'RetryDownloadAttempts'],
            name: '🔄 Retry Download Attempts',
            type: 'number',
            tooltip: 'Retries after a failed integrity verification (0-20).',
            defaultValue: config.retry_download_attempts ?? 2,
            sortOrder: 300,
            onChange: async (value) => {
                const attempts = Number.parseInt(value, 10);
                if (!Number.isInteger(attempts) || attempts < 0 || attempts > 20) {
                    showSmartLLMToast('Invalid retry count', 'Choose a value from 0 through 20.', 'warn');
                    return;
                }
                try { await updateConfig({ retry_download_attempts: attempts }); }
                catch (error) { showSmartLLMToast('Retry setting update failed', error); }
            },
        });
        await registerConfigSetting({
            id: 'SmartLLMNova.LogLevel',
            category: [...SETTINGS_CATEGORY, 'LogLevel'],
            name: 'Log Level',
            type: 'combo',
            options: ['error', 'warning', 'info', 'debug'],
            defaultValue: config.log_level || 'warning',
            sortOrder: 200,
            onChange: async (value) => {
                try {
                    const response = await fetch('/smartllm-nova/config/log_level', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ log_level: value }),
                    });
                    if (!response.ok) throw new Error('Log-level update failed');
                } catch (error) { showSmartLLMToast('Log level update failed', error); }
            },
        });
        await registerCredentialSetting({
            id: 'SmartLLMNova.HFToken',
            name: '🔑 Hugging Face Token',
            key: 'hf_token',
            configured: config.hf_token_configured === true,
            tooltip: 'Write-only Hugging Face credential. The server never returns token bytes.',
            sortOrder: 100,
        });
        await registerCredentialSetting({
            id: 'SmartLLMNova.ModelScopeToken',
            name: '🔑 ModelScope Token',
            key: 'modelscope_token',
            configured: config.modelscope_token_configured === true,
            tooltip: 'Write-only ModelScope credential. The server never returns token bytes.',
            sortOrder: 90,
        });
    },
});
