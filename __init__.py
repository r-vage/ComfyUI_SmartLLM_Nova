import os
from pathlib import Path

from .core import version
from .core.compatibility import find_conflicts
from .core.logger import log

_REPO_ROOT = Path(__file__).resolve().parent
_CONFLICTS = find_conflicts(_REPO_ROOT)

if _CONFLICTS:
    for conflict in _CONFLICTS:
        log.error(
            "Startup",
            f"SmartLLM Nova disabled to prevent duplicate nodes/routes: {conflict}",
        )
    log.error(
        "Startup",
        "Disable the duplicate SmartLLM Nova installation, then restart ComfyUI.",
    )
else:
    WEB_DIRECTORY = "./js"
    log.msg("", f"Version: {version}")

    from .core.migration import materialize_defaults

    materialize_defaults(_REPO_ROOT)

    try:
        from .core.sml.model_registry import sync_yolo_registry

        sync_yolo_registry()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        log.warning("Registry", f"Could not sync YOLO registry: {error}")

    try:
        from .core.sml.config_templates import (
            ensure_config_exists,
            initialize_llm_paths,
        )

        ensure_config_exists()
        initialize_llm_paths()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        log.warning("Config", f"Could not initialize LLM paths: {error}")

    try:
        from .core.sml import florence2_wrapper

        if not florence2_wrapper.FLORENCE2_CUSTOM_AVAILABLE:
            log.warning("Florence-2", florence2_wrapper.florence_unavailable_message())
    except (ImportError, RuntimeError, ValueError) as error:
        log.warning("Florence-2", f"Compatibility check failed: {error}")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    try:
        from .core.sml.docker_utils import get_docker_version, is_docker_installed

        if is_docker_installed():
            log.msg("Docker", f"Available: {get_docker_version()}")
    except (OSError, RuntimeError, ValueError):
        pass

    from .core.sml.server_endpoints import initialize_endpoints

    initialize_endpoints()

from comfy_api.latest import ComfyExtension, io  # type: ignore


class SmartLLMExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        if _CONFLICTS:
            return []
        from .py.RvConversion_DetectionToBboxes import (
            RvConversion_DetectionToBboxes,
        )
        from .py.RvConversion_SplitMiniMaxMusic3 import (
            RvConversion_SplitMiniMaxMusic3,
        )
        from .py.RvConversion_SplitYuE2 import (
            RvConversion_SplitYuE2,
        )
        from .py.RvLoader_SmartDetection import (
            RvLoader_Detection,
        )
        from .py.RvLoader_SmartLMLoader import (
            RvLoader_SmartModelLoader_LM,
        )

        return [
            RvLoader_SmartModelLoader_LM,
            RvLoader_Detection,
            RvConversion_DetectionToBboxes,
            RvConversion_SplitMiniMaxMusic3,
            RvConversion_SplitYuE2,
        ]


async def comfy_entrypoint() -> SmartLLMExtension:
    return SmartLLMExtension()
