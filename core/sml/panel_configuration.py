"""Structured controls for Nova nodes; execution stays in the existing engines."""
import copy
import json
import math
import secrets

from comfy_api.latest import io

from .device import normalize_device_selection

WIDGET_TYPE = "SMARTLLM_NOVA_PANEL"
SCALARS = {"INT", "FLOAT", "BOOLEAN", "STRING", "COMBO"}
OVERRIDES = {
    "loader": ("user_prompt",),
    "detection": ("user_input",),
    "bboxes": (), "yue2": ("song_json",), "music3": ("song_json",),
}
DYNAMIC = {"model", "model_name", "task", "task_2", "task_3", "task_4", "quantization"}
MAX_SEED = (1 << 64) - 1


def scalar(value):
    while isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value


def editor_spec(schema, kind):
    fields = {}
    for field in schema.inputs:
        if field.io_type not in SCALARS or getattr(field, "force_input", False) or field.id == "seed":
            continue
        data = {"type": field.io_type, "label": field.id.replace("_", " ").capitalize(),
                "tip": field.tooltip or "", "default": getattr(field, "default", None)}
        if getattr(field, "optional", False):
            data["optional"] = True
        if field.io_type == "COMBO":
            data["options"] = list(field.options)
            if data["default"] is None:
                data["default"] = data["options"][0] if data["options"] else ""
        elif field.io_type == "STRING" and data["default"] is None:
            data["default"] = ""
        for key in ("min", "max", "step", "multiline"):
            if (value := getattr(field, key, None)) is not None:
                data[key] = value
        if field.id in {"model", "model_name"} and str(data["default"]).startswith("__SEP__"):
            data["default"] = next((name for name in data["options"] if not name.startswith("__SEP__")), "")
        if field.id == 'device':
            data.update(options=['default', 'cpu'], default=normalize_device_selection(data['default']),
                        tip='Default follows ComfyUI’s configured device; CPU forces CPU processing. Backend accelerator support still applies.')
        fields[field.id] = data
    return {"kind": kind, "title": schema.display_name, "fields": fields,
            "seed": any(field.id == "seed" for field in schema.inputs)}


def default_configuration(spec):
    config = {"version": 1, "kind": spec["kind"],
              "parameters": {key: copy.deepcopy(field["default"]) for key, field in spec["fields"].items()}}
    if spec["seed"]:
        config["seed"] = {"value": "0", "policy": "random", "source": "local"}
    return config


def validate_value(value, key, field):
    if key == 'device':
        value = normalize_device_selection(value)
    kind = field["type"]
    if kind in {"INT", "FLOAT"}:
        if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{field['label']} must be a finite number.")
        if kind == "INT" and int(value) != value:
            raise ValueError(f"{field['label']} must be an integer.")
        if ("min" in field and value < field["min"]) or ("max" in field and value > field["max"]):
            raise ValueError(f"{field['label']} is outside its supported range.")
        return int(value) if kind == "INT" else value
    if kind == "BOOLEAN":
        if type(value) is not bool:
            raise ValueError(f"{field['label']} must be enabled or disabled.")
    elif not isinstance(value, str):
        raise ValueError(f"{field['label']} must be text.")
    elif kind == "COMBO" and key not in DYNAMIC and value not in field["options"]:
        raise ValueError(f"Unknown {field['label']}: {value!r}.")
    return value


def seed_number(value):
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 20:
        value = int(value)
    if type(value) is not int or not 0 <= value <= MAX_SEED:
        raise ValueError("Seed must be an unsigned 64-bit integer.")
    return value


def parse_configuration(raw, spec):
    raw = scalar(raw)
    if not isinstance(raw, str) or len(raw) > 1048576:
        raise ValueError("Configuration must be JSON text smaller than 1 MiB.")
    try:
        config = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid configuration JSON.") from error
    keys = {"version", "kind", "parameters"} | ({"seed"} if spec["seed"] else set())
    if (not isinstance(config, dict) or set(config) != keys or type(config["version"]) is not int
            or config["version"] != 1 or config["kind"] != spec["kind"]
            or not isinstance(config["parameters"], dict)):
        raise ValueError(f"Expected version 1 {spec['title']} configuration.")
    fields = spec["fields"]
    provided = set(config["parameters"])
    if provided - set(fields) or any(not fields[key].get("optional") for key in set(fields) - provided):
        raise ValueError(f"Expected version 1 {spec['title']} configuration.")
    config["parameters"] = {key: validate_value(config["parameters"].get(key, copy.deepcopy(field["default"])), key, field)
                            for key, field in spec["fields"].items()}
    if spec["seed"]:
        seed = config["seed"]
        if (not isinstance(seed, dict) or set(seed) != {"value", "policy", "source"}
                or seed["policy"] not in {"fixed", "random", "increment", "decrement"}
                or seed["source"] not in {"local", "input"}):
            raise ValueError("Invalid seed configuration.")
        seed_number(seed["value"])
    return config


class SmartLLMPanel:
    PANEL_KIND = ""

    @classmethod
    def _panel_spec(cls):
        return editor_spec(cls._engine_schema(), cls.PANEL_KIND)

    @classmethod
    def define_schema(cls):
        schema = cls._engine_schema()
        spec = editor_spec(schema, cls.PANEL_KIND)
        inputs = [field for field in schema.inputs
                  if field.id not in spec["fields"] and field.id != "seed"]
        for key in OVERRIDES[cls.PANEL_KIND]:
            field = spec["fields"][key]
            constructor = {"FLOAT": io.Float, "INT": io.Int}.get(field["type"], io.String)
            inputs.append(constructor.Input(key, optional=True, force_input=True,
                                            tooltip=field["tip"] + " Connected non-None values override the panel."))
        if spec["seed"]:
            inputs.append(io.Int.Input("seed_input", optional=True, force_input=True, min=0, max=MAX_SEED,
                                       tooltip="Used when Local seed is off. Disconnecting restores the local seed."))
        inputs.append(io.String.Input("configuration", socketless=True, default=json.dumps(default_configuration(spec)),
                                      extra_dict={"widgetType": WIDGET_TYPE, "editorSpec": spec}))
        schema.inputs = inputs
        return schema

    @classmethod
    def validate_inputs(cls, configuration):
        try:
            parse_configuration(configuration, cls._panel_spec())
        except (TypeError, ValueError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, configuration, **_inputs):
        config = parse_configuration(configuration, cls._panel_spec())
        seed = config.get("seed")
        if seed and (seed["source"] == "local" or scalar(_inputs.get("seed_input")) is None) and seed["policy"] != "fixed":
            return float("nan")
        if cls.PANEL_KIND in {"loader", "detection"}:
            from .text_encoders import (
                DISPLAY_PREFIX,
                resolve_text_encoder_path,
                text_encoder_signature,
            )

            model = config["parameters"]["model" if cls.PANEL_KIND == "loader" else "model_name"]
            if model.startswith(DISPLAY_PREFIX):
                path = resolve_text_encoder_path(model[len(DISPLAY_PREFIX):])
                return (configuration, text_encoder_signature(path))
        return configuration

    @classmethod
    def execute(cls, configuration, **inputs):
        spec = cls._panel_spec()
        config = parse_configuration(configuration, spec)
        params = config["parameters"]
        if cls.PANEL_KIND == "detection":
            # Older API prompts can still supply these removed sockets. The
            # saved panel values now control filtering; discard obsolete kwargs.
            for key in ("confidence", "nms_iou_threshold", "detection_filter"):
                inputs.pop(key, None)
        list_overrides = {}
        for key in OVERRIDES[cls.PANEL_KIND]:
            value = inputs.pop(key, None)
            if scalar(value) is None:
                continue
            if key == "user_prompt" and isinstance(value, (tuple, list)):
                list_overrides[key] = [validate_value(item, key, spec["fields"][key]) for item in value]
            else:
                params[key] = validate_value(scalar(value), key, spec["fields"][key])
        seed = config.get("seed")
        external_seed = scalar(inputs.pop("seed_input", None))
        if seed:
            params["seed"] = (seed_number(external_seed) if seed["source"] == "input" and external_seed is not None
                              else secrets.randbits(64) if seed["policy"] == "random" else seed_number(seed["value"]))
        if cls.PANEL_KIND == "loader" and scalar(inputs.get("system_prompt")):
            from .model_registry import get_model_entry
            entry = get_model_entry(params["model"]) or {}
            if entry.get("family") != "Florence" and entry.get("backend") != "wd14":
                params["task"] = "Direct Chat"
                params["multi_task_mode"] = False
        if cls._engine_schema().is_input_list:
            params = {key: [value] for key, value in params.items()}
            params.update(list_overrides)
        result = cls._execute_engine(**params, **inputs)
        if seed:
            result.ui = {**(result.ui or {}), "nova_seed": [str(scalar(params["seed"]))]}
        return result
