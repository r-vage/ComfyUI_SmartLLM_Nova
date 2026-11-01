# ruff: noqa: N999
"""Socket-only query adapter for Smart Detection's settings pipe."""

from comfy_api.latest import io

from ..core import CATEGORY
from ..core.sml.settings_controls import SETTINGS_TYPES
from ..core.sml.settings_pipe import merge_detection_prompt_pipe


class RvPipe_SmartDetectionPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Smart Detection Prompt Pipe [SmartLLM Nova]", display_name="Smart Detection Prompt Pipe",
            category=CATEGORY.MAIN.value + "/ IO",
            description="Send query or class-filter text to Smart Detection through pipe_opt, with or without native settings.",
            inputs=[
                io.String.Input("user_input", optional=True, force_input=True,
                                tooltip="Detection query or class-filter text. Empty text clears the incoming query. Like Smart Detection, a prompt list uses its first string."),
                io.Custom(SETTINGS_TYPES["detection"]).Input("pipe_opt", optional=True,
                                                          tooltip="Optional native settings or another detection prompt pipe. An unconnected query preserves its values."),
            ],
            outputs=[io.Custom(SETTINGS_TYPES["detection"]).Output("settings")],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, user_input=None, pipe_opt=None):
        return io.NodeOutput(merge_detection_prompt_pipe(pipe_opt, user_input))
