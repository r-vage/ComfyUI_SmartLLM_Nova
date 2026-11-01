# ruff: noqa: N999
"""Small socket-only LM prompt adapter, usable with or without native settings."""

from comfy_api.latest import io

from ..core import CATEGORY
from ..core.sml.settings_controls import SETTINGS_TYPES
from ..core.sml.settings_pipe import merge_lm_prompt_pipe


class RvPipe_SmartLMPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Smart LM Prompt Pipe [SmartLLM Nova]", display_name="Smart LM Prompt Pipe",
            category=CATEGORY.MAIN.value + "/ IO",
            description="Send prompts to Smart LM Loader through pipe_opt. Optionally pass native settings through this node; connected prompts override them.",
            inputs=[
                io.String.Input("user_prompt", optional=True, force_input=True,
                                tooltip="User text or a list of prompts. A connected empty string clears the pipe's user prompt."),
                io.String.Input("system_prompt", optional=True, force_input=True,
                                tooltip="One shared system instruction. Nonblank text uses Direct Chat for supported models; empty text restores task instructions."),
                io.Custom(SETTINGS_TYPES["loader"]).Input("pipe_opt", optional=True,
                                                       tooltip="Optional native settings or another prompt pipe. Unconnected prompt inputs preserve its values."),
            ],
            outputs=[io.Custom(SETTINGS_TYPES["loader"]).Output("settings")],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, user_prompt=None, system_prompt=None, pipe_opt=None):
        return io.NodeOutput(merge_lm_prompt_pipe(pipe_opt, user_prompt, system_prompt))
