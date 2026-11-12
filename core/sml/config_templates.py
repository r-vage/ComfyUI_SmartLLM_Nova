# SmartLM Templates - Configuration Management
#
# Handles config and context operations:
# - Config value loading/saving
# - TemplateContext for loader pipelines
# - Prompt configuration loading
#
# Used by the SmartLM core loader and related modules (shared utilities).

from pathlib import Path
from typing import Any, Dict

from ..config_store import (
    ensure_config_exists as ensure_config_exists,
)
from ..config_store import (
    get_config_snapshot as get_config_snapshot,
)
from ..config_store import (
    get_config_value as get_config_value,
)
from ..config_store import (
    invalidate_config_cache as invalidate_config_cache,
)
from ..config_store import (
    update_config_value as update_config_value,
)
from ..config_store import (
    update_config_values as update_config_values,
)
from .logger import log

_LOG_PREFIX = ""


# ============================================================================
# Template Context Class
# ============================================================================


class TemplateContext:
    # Context object for template-related operations.
    #
    # Holds all template/widget values in one place, avoiding the need to
    # pass many individual parameters through call chains.
    #
    # Thread-safe: Each execution creates its own context instance.
    #
    # Usage:
    #     # From widget values
    #     ctx = TemplateContext.from_widgets(
    #         model_family="LLaVA",
    #         model_type="llava",
    #         loading_method="Ollama (Docker)",
    #     )
    #
    #     # Update values
    #     ctx.update(has_vision=True, ollama_model="local_llava:latest")

    def __init__(self):
        # Core template fields
        self.model_family: str = ""
        self.model_type: str = ""
        self.loading_method: str = ""

        # Model source
        self.repo_id: str = ""
        self.local_path: str = ""
        self.ollama_model: str = ""
        self.model_source: str = ""  # "huggingface", "local", "ollama"

        # Vision support
        self.mmproj_path: str = ""
        self.mmproj_url: str = ""
        self.has_vision: bool = False

        # Configuration
        self.quantization: str = "auto"
        self.attention_mode: str = "auto"
        self.context_size: int = 8192
        self.max_tokens: int = 1024

        # Metadata
        self.original_filename: str = ""
        self.quantized: bool = False
        self.default_task: str = ""
        self.default_text_input: str = ""


    @classmethod
    def from_widgets(
        cls,
        model_family: str = "",
        model_type: str = "",
        loading_method: str = "",
        **kwargs,
    ) -> "TemplateContext":
        # Create context from widget values
        ctx = cls()
        ctx.model_family = model_family
        ctx.model_type = model_type
        ctx.loading_method = loading_method

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(ctx, key):
                setattr(ctx, key, value)

        return ctx

    def update(self, **kwargs) -> "TemplateContext":
        # Update context values. Returns self for chaining
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        # Convert context to dictionary (for compatibility)
        return {
            "model_family": self.model_family,
            "model_type": self.model_type,
            "loading_method": self.loading_method,
            "repo_id": self.repo_id,
            "local_path": self.local_path,
            "ollama_model": self.ollama_model,
            "model_source": self.model_source,
            "mmproj_path": self.mmproj_path,
            "mmproj_url": self.mmproj_url,
            "quantization": self.quantization,
            "attention_mode": self.attention_mode,
            "context_size": self.context_size,
            "max_tokens": self.max_tokens,
            "quantized": self.quantized,
            "default_task": self.default_task,
            "default_text_input": self.default_text_input,
            "has_vision": self.has_vision,
        }


# ============================================================================
# Directory Constants and Paths
# ============================================================================

NODE_DIR = Path(__file__).parent.parent.parent

# Config directories
REPO_CONFIG_DIR = NODE_DIR / "config"

def resolve_llm_models_path(value: str) -> Path:
    import folder_paths  # type: ignore

    path = Path(value if value.strip() else "LLM").expanduser()
    return path if path.is_absolute() else Path(folder_paths.models_dir) / path


def get_llm_models_path() -> Path:
    return resolve_llm_models_path(get_config_value("llm_models_path", "LLM"))


def get_llm_models_absolute_path() -> str:
    # Derived on every read so native acquisition and Docker cannot diverge.
    return str(get_llm_models_path().resolve())


def initialize_llm_paths() -> bool:
    current = get_config_snapshot()
    relative = current.get("llm_models_path", "LLM")
    if not relative or not relative.strip():
        relative = "LLM"
    path = resolve_llm_models_path(relative)
    absolute = str(path.resolve())
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        log.warning(_LOG_PREFIX, f"Could not create LLM models folder: {error}")
    updates = {"llm_models_path": relative, "llm_models_absolute_path": absolute}
    if all(current.get(key) == value for key, value in updates.items()):
        return False
    return update_config_values(updates)


# ============================================================================
# Task prompt compatibility for backend adapters
# ============================================================================


def get_llm_few_shot_examples():
    """Legacy adapter name: supplies task instructions/templates, never examples."""
    from .task_library import task_configs
    return task_configs()


def reload_task_configs():
    from .task_library import invalidate_catalog, load_catalog
    invalidate_catalog()
    catalog = load_catalog()
    return {"tasks": len(catalog.tasks), "path": str(NODE_DIR / "tasks"), "errors": list(catalog.errors)}


def get_vision_few_shot_messages(task_name: str) -> list:
    """Older adapters may request demonstrations; tasks no longer contain them."""
    return []
