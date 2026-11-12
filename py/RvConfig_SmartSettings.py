# ruff: noqa: N999
"""Native main-setting widgets that can be promoted onto ComfyUI subgraphs."""

from comfy_api.latest import io

from ..core import CATEGORY
from ..core.sml.panel_configuration import editor_spec, parse_settings_pipe
from ..core.sml.settings_controls import SETTINGS_TYPES, settings_fields
from ..core.sml.text_encoders import (
    DISPLAY_PREFIX,
    resolve_text_encoder_path,
    text_encoder_signature,
)


class _NativeSettings(io.ComfyNode):
    KIND = ""
    TITLE = ""

    @classmethod
    def define_schema(cls):
        fields = settings_fields(cls.KIND)
        model_key = "model" if cls.KIND == "loader" else "model_name"
        fields[model_key].options = [name for name in fields[model_key].options
                                     if not name.startswith("__SEP__")]
        if fields[model_key].default not in fields[model_key].options:
            fields[model_key].default = next(iter(fields[model_key].options), "")
        return io.Schema(
            node_id=f"{cls.TITLE} [SmartLLM Nova]", display_name=cls.TITLE,
            category=CATEGORY.MAIN.value + "/ Settings",
            description="Native controls for subgraphs. Connect settings to the matching panel's pipe_opt; promote these widgets to edit them on the subgraph.",
            inputs=list(fields.values()),
            outputs=[io.Custom(SETTINGS_TYPES[cls.KIND]).Output("settings")],
        )

    @classmethod
    def validate_inputs(cls, quantization, task=None):
        # Catalog IDs and saved name aliases can outlive the current combo list.
        # The consuming loader/detector validates task existence and compatibility.
        if task is not None and (not isinstance(task, str) or not task):
            return "Task must be text."
        # Registry-specific GGUF choices extend the initial placeholder list.
        return True if quantization is None or isinstance(quantization, str) and quantization else "Quantization must be text."

    @classmethod
    def fingerprint_inputs(cls, model=None, model_name=None, **_inputs):
        selected = model if cls.KIND == "loader" else model_name
        if selected is None:
            # A linked model has no constant value during fingerprinting.
            return float("nan")
        if selected.startswith(DISPLAY_PREFIX):
            return text_encoder_signature(resolve_text_encoder_path(selected[len(DISPLAY_PREFIX):]))
        return selected

    @classmethod
    def execute(cls, **values):
        spec = editor_spec(cls.define_schema(), cls.KIND)
        payload = {"version": 1, "kind": cls.KIND, "parameters": values}
        payload["parameters"] = parse_settings_pipe(payload, spec)
        return io.NodeOutput(payload)


class RvConfig_SmartLMSettings(_NativeSettings):
    KIND = "loader"
    TITLE = "Smart LM Settings"


class RvConfig_SmartDetectionSettings(_NativeSettings):
    KIND = "detection"
    TITLE = "Smart Detection Settings"
