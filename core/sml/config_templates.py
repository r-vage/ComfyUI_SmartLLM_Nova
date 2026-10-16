# SmartLM Templates - Configuration Management
#
# Handles config and context operations:
# - Config value loading/saving
# - TemplateContext for loader pipelines
# - Prompt configuration loading
#
# Used by the SmartLM core loader and related modules (shared utilities).

import copy
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional

from ..config_store import (
    ensure_config_exists as ensure_config_exists,
    get_config_snapshot as get_config_snapshot,
    get_config_value as get_config_value,
    invalidate_config_cache as invalidate_config_cache,
    update_config_value as update_config_value,
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
# Few-Shot Example Loading
# ============================================================================

LLM_FEW_SHOT_EXAMPLES: Dict[str, Any] = {}
_few_shot_mtime: float = 0.0
_few_shot_path: Optional[Path] = None


@dataclass
class _TrainingProfile:
    filename: str
    examples: dict[str, Any] | None = None


_training_profile: ContextVar[_TrainingProfile | None] = ContextVar(
    "smartllm_nova_training_profile", default=None,
)
_training_cache: dict[Path, tuple[tuple[int, int, int, int], dict[str, Any]]] = {}
_training_cache_lock = threading.RLock()


@contextmanager
def training_profile(nsfw: bool = False):
    """Select a lazy, execution-local snapshot without changing root config."""
    if type(nsfw) is not bool:
        raise ValueError("NSFW training must be enabled or disabled.")
    filename = "llm_few_shot_training_nsfw.json" if nsfw else "llm_few_shot_training.json"
    token = _training_profile.set(_TrainingProfile(filename))
    try:
        yield
    finally:
        _training_profile.reset(token)


def with_training_profile(function):
    """Consume the panel's profile option and scope all backend/task lookups."""
    @wraps(function)
    def wrapped(*args, nsfw_training=False, **kwargs):
        def scalar(value):
            while isinstance(value, (tuple, list)):
                value = value[0] if value else False
            return value

        nsfw = scalar(nsfw_training)
        if type(nsfw) is not bool:
            raise ValueError("NSFW training must be enabled or disabled.")
        enabled = scalar(kwargs.get("use_few_shot_training", True))
        with training_profile(nsfw and enabled is True):
            return function(*args, **kwargs)
    return wrapped


def _profile_examples(profile: _TrainingProfile) -> dict[str, Any]:
    if profile.examples is None:
        path = REPO_CONFIG_DIR / profile.filename
        try:
            with _training_cache_lock:
                info = path.stat()
                stamp = (info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino)
                cached = _training_cache.get(path)
                if cached is None or cached[0] != stamp:
                    cached = (stamp, _read_few_shot_configs(path))
                    _training_cache[path] = cached
                profile.examples = copy.deepcopy(cached[1])
        except (OSError, ValueError, TypeError, RecursionError) as error:
            raise ValueError(
                f"Could not read training examples from config/{profile.filename}: {error}"
            ) from error
    return profile.examples


def get_llm_few_shot_config_path() -> Path:
    # Get path to LLM few-shot config file.
    # Reads the filename from config.json "few_shot_training_file" key.
    # Falls back to "llm_few_shot_training.json" if not set or invalid.
    default_name = "llm_few_shot_training.json"
    configured_name = get_config_value("few_shot_training_file", default_name)

    # Sanitize: strip path separators to prevent path traversal
    if not configured_name or not isinstance(configured_name, str):
        configured_name = default_name
    configured_name = configured_name.strip()
    safe_name = Path(configured_name).name  # strip any directory components

    # Must end with .json
    if not safe_name.endswith(".json"):
        log.warning(
            _LOG_PREFIX,
            f"Invalid few-shot file name '{safe_name}', must end with .json. Using default.",
        )
        safe_name = default_name

    target = REPO_CONFIG_DIR / safe_name
    if not target.exists():
        log.warning(
            _LOG_PREFIX,
            f"Few-shot file '{safe_name}' not found in {REPO_CONFIG_DIR}, falling back to default",
        )
        target = REPO_CONFIG_DIR / default_name

    return target


def _read_few_shot_configs(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    if not isinstance(loaded_data, dict):
        raise TypeError("Training data must be a JSON object.")

    # Normalize all entries: the JSON stores everything as list-of-pairs
    # [["key", val], ...] at every level.  Convert recursively so callers
    # can safely call .get() and Ollama/vLLM receive plain dicts.
    def _normalize(obj: Any) -> Any:
        if (
            isinstance(obj, list)
            and obj
            and isinstance(obj[0], list)
            and len(obj[0]) == 2
        ):
            # Looks like list-of-pairs → dict, then recurse into values
            try:
                return {pair[0]: _normalize(pair[1]) for pair in obj}
            except (TypeError, IndexError):
                pass
        if isinstance(obj, list):
            return [_normalize(item) for item in obj]
        return obj

    normalized: Dict[str, Any] = {}
    for k, v in loaded_data.items():
        if k.startswith("_"):
            # Metadata keys (e.g. _description) — keep as-is
            normalized[k] = v
        else:
            normalized[k] = _normalize(v)
    # Runtime fallback for installations that still contain only retired H3
    # task keys. Startup migration normally materializes the canonical keys.
    h3_legacy_keys = {
        "minimax_h3_scene": "minimax_h3_scene_5s",
        "minimax_h3_t2va_timeline": "minimax_h3_t2va_timeline_15s",
        "minimax_h3_i2va_timeline": "minimax_h3_i2va_timeline_15s",
        "minimax_h3_fl2va_timeline": "minimax_h3_fl2va_timeline_15s",
        "minimax_h3_l2va_timeline": "minimax_h3_l2va_timeline_15s",
    }
    for canonical_key, legacy_key in h3_legacy_keys.items():
        if canonical_key not in normalized and legacy_key in normalized:
            normalized[canonical_key] = normalized[legacy_key]
    return normalized


def _load_few_shot_configs():
    # Load LLM few-shot training examples and show transformers version.
    global LLM_FEW_SHOT_EXAMPLES, _few_shot_mtime, _few_shot_path

    llm_config_path = get_llm_few_shot_config_path()
    _few_shot_path = llm_config_path
    try:
        normalized = _read_few_shot_configs(llm_config_path)
        LLM_FEW_SHOT_EXAMPLES.clear()
        LLM_FEW_SHOT_EXAMPLES.update(normalized)
        try:
            _few_shot_mtime = llm_config_path.stat().st_mtime
        except OSError:
            _few_shot_mtime = 0.0
        log.msg(
            _LOG_PREFIX,
            f"Loaded LLM few-shot training examples ({len(normalized)} modes) from {llm_config_path.name}",
        )
    except Exception as exc:
        log.warning(_LOG_PREFIX, f"LLM few-shot config load failed: {exc}")
        LLM_FEW_SHOT_EXAMPLES.clear()
        LLM_FEW_SHOT_EXAMPLES.update(
            {
                "prompt_generation": {
                    "system_prompt": "You are a helpful assistant.",
                    "examples": [],
                },
                "direct_chat": {
                    "system_prompt": "You are a helpful assistant. Try your best to give the best response possible to the user.",
                    "examples": [],
                },
            }
        )

    # Show transformers version
    try:
        import transformers  # type: ignore

        log.msg(_LOG_PREFIX, f"Transformers version: {transformers.__version__}")
    except Exception:
        log.warning(_LOG_PREFIX, "Transformers not found")


def get_llm_few_shot_examples() -> Dict[str, Any]:
    # Get loaded LLM few-shot examples.
    # Auto-reloads if the source JSON file has changed on disk OR the
    # configured `few_shot_training_file` (NSFW vs. SFW switch) was changed.
    global _few_shot_mtime, _few_shot_path
    profile = _training_profile.get()
    if profile is not None:
        return _profile_examples(profile)
    try:
        current_path = get_llm_few_shot_config_path()
        if _few_shot_path != current_path:
            # Configured file changed (e.g. SFW <-> NSFW toggle) → full reload
            _load_few_shot_configs()
        else:
            try:
                mtime = current_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if mtime and mtime != _few_shot_mtime:
                _load_few_shot_configs()
    except Exception:
        # Never fail callers because of an mtime check
        pass
    return LLM_FEW_SHOT_EXAMPLES


def reload_few_shot_configs() -> Dict[str, Any]:
    # Force-reload few-shot training examples from disk.
    # Returns a small status dict for endpoint reporting.
    with _training_cache_lock:
        _training_cache.clear()
    _load_few_shot_configs()
    return {
        "modes": len(LLM_FEW_SHOT_EXAMPLES),
        "path": str(_few_shot_path) if _few_shot_path else "",
    }


def get_vision_few_shot_messages(task_name: str) -> list:
    # Get text-only few-shot example messages for a vision task.
    #
    # These are user/assistant text pairs that teach the model the desired
    # output style (no "illustration" prefixes, uncensored descriptions, etc.)
    # without requiring example images. Injected before the actual image message.
    #
    # Args:
    #     task_name: Display name like "Detailed Description" or key like "detailed_description"
    #
    # Returns:
    #     List of message dicts (user/assistant pairs), empty if no examples found
    if not task_name:
        return []

    # Normalize to key format: "Detailed Description" -> "detailed_description"
    task_key = task_name.lower().replace(" ", "_").replace("&", "&")

    config = get_llm_few_shot_examples().get(task_key)
    if not config:
        return []

    # All task entries use the list-of-pairs format: [["key", value], ...].
    # Convert to dict for .get() access.
    if isinstance(config, list):
        try:
            config = dict(config)
        except (TypeError, ValueError):
            return []

    examples = config.get("examples", [])
    if not examples:
        return []

    # Normalize content to list-of-dicts for VLM chat templates that iterate
    # content expecting {"type": "text", "text": ...} entries (Qwen3-VL,
    # Qwen3.5-VL, etc.). Mixing string content with list content in the same
    # message thread raises "string indices must be integers" inside the
    # jinja template. Real user messages always use list-of-dicts because
    # they need a {"type": "image"} entry alongside text.
    normalized = []
    for ex in examples:
        # Each example may itself be a list-of-pairs [["role", "user"], ["content", "..."]].
        # Convert to dict before accessing keys.
        if isinstance(ex, list):
            try:
                ex = dict(ex)
            except (TypeError, ValueError):
                continue
        if not isinstance(ex, dict):
            continue
        content = ex.get("content")
        if isinstance(content, str):
            normalized.append({**ex, "content": [{"type": "text", "text": content}]})
        else:
            normalized.append(ex)

    log.debug(
        _LOG_PREFIX,
        f"Found {len(normalized)} vision few-shot examples for '{task_key}'",
    )
    return normalized


# ============================================================================
# Auto-load configs on import
# ============================================================================
_load_few_shot_configs()
