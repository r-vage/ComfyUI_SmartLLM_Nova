"""Settings envelopes and prompt composition shared by Nova pipe producers."""

LM_PROMPT_KEYS = ("user_prompt", "system_prompt")
SYSTEM_PROMPT_SPEC = {
    "type": "STRING", "label": "System prompt", "default": "", "optional": True,
    "multiline": True,
    "tip": "Complete system instructions. Nonblank text uses Direct Chat for supported models; leave empty to use the selected task's instructions.",
}


def settings_pipe_parameters(raw, kind):
    """Read one envelope; callers validate fields against their own schema."""
    while isinstance(raw, (list, tuple)):
        if len(raw) > 1:
            raise ValueError("pipe_opt accepts one settings pipe, not a list of configurations.")
        raw = raw[0] if raw else None
    if raw is None:
        return {}
    if (not isinstance(raw, dict) or set(raw) != {"version", "kind", "parameters"}
            or type(raw["version"]) is not int or raw["version"] != 1
            or raw["kind"] != kind or not isinstance(raw["parameters"], dict)):
        raise ValueError(f"pipe_opt requires version 1 {kind} settings.")
    return dict(raw["parameters"])


def validate_pipe_prompt(value, key):
    if key == "user_prompt" and isinstance(value, (list, tuple)):
        if not value or any(not isinstance(item, str) for item in value):
            raise ValueError("User prompt must be text or a nonempty list of text prompts.")
        return list(value)
    if not isinstance(value, str):
        raise TypeError(f"{key.replace('_', ' ').capitalize()} must be text.")
    return value


def merge_lm_prompt_pipe(pipe_opt=None, user_prompt=None, system_prompt=None):
    """Connected prompts override a settings pipe without changing its producer."""
    params = settings_pipe_parameters(pipe_opt, "loader")
    for key, value in (("user_prompt", user_prompt), ("system_prompt", system_prompt)):
        if value is None or isinstance(value, (list, tuple)) and not value:
            continue
        if key == "system_prompt" and isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("Supply one system prompt shared by all user prompts.")
            value = value[0]
        params[key] = value
    for key in LM_PROMPT_KEYS:
        if key in params:
            params[key] = validate_pipe_prompt(params[key], key)
    return {"version": 1, "kind": "loader", "parameters": params}


def merge_detection_prompt_pipe(pipe_opt=None, user_input=None):
    """Compose one detection query, retaining the engine's first-string behavior."""
    params = settings_pipe_parameters(pipe_opt, "detection")
    while isinstance(user_input, (list, tuple)):
        user_input = user_input[0] if user_input else None
    if user_input is not None:
        params["user_input"] = user_input
    if "user_input" in params:
        params["user_input"] = validate_pipe_prompt(params["user_input"], "user_input")
    return {"version": 1, "kind": "detection", "parameters": params}
