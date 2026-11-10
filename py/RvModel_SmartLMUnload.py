# ruff: noqa: N999
"""A socket-only unload barrier; bypass preserves normal data flow."""

from comfy_api.latest import io

from ..core import CATEGORY
from ..core.sml.model_unload import unload_retained_models


class RvModel_SmartLMUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Smart LM Unload [SmartLLM Nova]",
            display_name="Smart LM Unload",
            category=CATEGORY.MAIN.value + CATEGORY.LOADER.value,
            description=(
                "Release retained SmartLLM models and stop its managed inference containers, "
                "then pass the input through unchanged. Place after the final LM task or "
                "Loop End, before downstream generation. Bypass to keep models loaded. "
                "Does not unload unrelated ComfyUI models."
            ),
            inputs=[io.AnyType.Input("value", tooltip="Completed upstream result; accepts any type or list.")],
            outputs=[io.AnyType.Output("value", is_output_list=True)],
            is_input_list=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, value):
        unload_retained_models()
        return io.NodeOutput(value)
