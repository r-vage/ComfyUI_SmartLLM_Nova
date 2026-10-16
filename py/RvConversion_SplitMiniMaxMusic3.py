"""Split validated Music 3 JSON into the native encoder's text inputs."""

# ruff: noqa: N999 - preserve the repository's node module naming convention

from comfy_api.latest import io  # type: ignore

from ..core import CATEGORY
from ..core.sml.music3 import parse_song_json
from ..core.sml.panel_configuration import SmartLLMPanel


class RvConversion_SplitMiniMaxMusic3(SmartLLMPanel, io.ComfyNode):
    PANEL_KIND = "music3"
    @classmethod
    def _engine_schema(cls):
        return io.Schema(
            node_id="Split MiniMax Music 3 [SmartLLM Nova]",
            display_name="Split MiniMax Music 3",
            category=CATEGORY.MAIN.value + CATEGORY.CONVERSION.value,
            description="Split Music 3 song JSON into caption and lyrics for the native encoder.",
            inputs=[
                io.String.Input(
                    "song_json",
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip="Connect Smart LM Loader's result or paste caption/lyrics JSON.",
                ),
            ],
            outputs=[io.String.Output("caption"), io.String.Output("lyrics")],
        )

    @classmethod
    def _execute_engine(cls, song_json):
        song = parse_song_json(song_json)
        return io.NodeOutput(song["caption"], song["lyrics"])
