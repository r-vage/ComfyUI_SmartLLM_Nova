# SmartLLM Model Registry
#
# Loads backend-split registry JSON files from registry/ and merges them
# into a single flat model list with backend suffixes. Provides lookup,
# defaults persistence, and helper functions for the new Smart Model Loader.
#
# Registry files:
#   registry/transformers_models.json  — no suffix (default backend)
#   registry/gguf_models.json          — "-GGUF" suffix
#   registry/ollama_models.json        — "-Ollama" suffix
#   registry/vllm_models.json          — "-vLLM" suffix
#   registry/vllm_native_models.json   — "-vLLM Native" suffix
#   registry/sglang_models.json        — "-SGLang" suffix
#   registry/llamacpp_models.json      — "-llama.cpp" suffix
#   registry/wd14_models.json          — no suffix (WD14- prefix is distinct)
#   registry/user_models.json          — per-backend sections, merged on top
#   registry/defaults.json             — global defaults (context_size, etc.)

import copy
import json
import os
import re
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from .json_store import JsonStoreError, read_json_object, update_json_object
from .logger import log
from .model_types import ModelFamily
from .text_encoders import (
    DISPLAY_PREFIX,
    MODEL_SEPARATOR_TEXT,
    MODEL_SEPARATOR_VISION,
    get_text_encoder_entries,
    get_text_encoder_entry,
)

_LOG_PREFIX = "Registry"

# ============================================================================
# Constants
# ============================================================================

_REGISTRY_DIR = (
    Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "registry"
)
_YOLO_DEFAULT_REGISTRY_PATH = (
    _REGISTRY_DIR.parent
    / ".defaults"
    / "registry"
    / "yolo_models.json.example"
)

# Backend file → display suffix mapping
# Transformers and WD14 get no suffix (Transformers is default, WD14 names are already distinct)
_BACKEND_FILES = {
    "transformers": ("transformers_models.json", ""),
    "gguf": ("gguf_models.json", "-GGUF"),
    "ollama": ("ollama_models.json", "-Ollama"),
    "vllm": ("vllm_models.json", "-vLLM"),
    "vllm_native": ("vllm_native_models.json", "-vLLM Native"),
    "sglang": ("sglang_models.json", "-SGLang"),
    "llamacpp": ("llamacpp_models.json", "-llama.cpp"),
    "wd14": ("wd14_models.json", ""),
}

SUPPORTED_REGISTRY_BACKENDS = (*_BACKEND_FILES, "yolo")

# Registry family string → ModelFamily enum
FAMILY_MAP: Dict[str, ModelFamily] = {
    "Qwen": ModelFamily.QWEN,
    "Mistral": ModelFamily.MISTRAL,
    "Florence": ModelFamily.FLORENCE,
    "LLaVA": ModelFamily.LLAVA,
    "LLM_TEXT": ModelFamily.LLM_TEXT,
    "VLM": ModelFamily.VLM,
    "WD14": ModelFamily.WD14,
    "YOLO": ModelFamily.YOLO,
}
SUPPORTED_REGISTRY_FAMILIES = tuple(FAMILY_MAP)

# Reverse: suffix → backend key
_SUFFIX_TO_BACKEND = {v[1]: k for k, v in _BACKEND_FILES.items() if v[1]}
# e.g. {"-GGUF": "gguf", "-Ollama": "ollama", "-vLLM": "vllm", "-SGLang": "sglang"}


# ============================================================================
# Module State
# ============================================================================

_lock = threading.Lock()
_merged_registry: Optional[Dict[str, Dict[str, Any]]] = None
# display_name → {"backend": str, "name": str, **entry_fields}

_defaults_cache: Optional[Dict[str, Any]] = None
_defaults_signature: tuple[int, int] | None = None
_defaults_lock = threading.RLock()


# ============================================================================
# Internal Loaders
# ============================================================================


def _load_json(path: Path) -> Any:
    # Load a JSON file, return empty dict on error.
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(_LOG_PREFIX, f"Failed to load {path.name}: {e}")
        return {}


def _merge_backend(
    registry: Dict[str, Dict],
    backend: str,
    models: Dict[str, Dict],
    suffix: str,
    origin: str,
):
    # Merge models from one backend into the flat registry dict.
    # Skips entries whose keys start with "_" (reserved for metadata).
    for name, entry in models.items():
        if name.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue

        display_name = f"{name}{suffix}"
        if display_name in registry:
            log.warning(
                _LOG_PREFIX,
                f"Duplicate model name '{display_name}' — skipping (already registered)",
            )
            continue
        sanitized_entry = {
            key: value for key, value in entry.items() if key != "trust_remote_code"
        }
        registry[display_name] = {
            "backend": backend,
            "name": name,
            "_registry_origin": origin,
            **sanitized_entry,
        }


def _build_registry() -> Dict[str, Dict[str, Any]]:
    # Load all registry files and build the merged flat dict.
    registry: Dict[str, Dict[str, Any]] = {}

    # 1. Load per-backend files
    for backend, (filename, suffix) in _BACKEND_FILES.items():
        path = _REGISTRY_DIR / filename
        models = _load_json(path)
        if models:
            _merge_backend(registry, backend, models, suffix, "curated")
            log.debug(_LOG_PREFIX, f"Loaded {len(models)} models from {filename}")

    # 2. Load user_models.json — sectioned by backend
    user_path = _REGISTRY_DIR / "user_models.json"
    user_data = _load_json(user_path)
    if user_data:
        for backend, (_, suffix) in _BACKEND_FILES.items():
            section = user_data.get(backend, {})
            if section:
                _merge_backend(registry, backend, section, suffix, "user")
                log.debug(
                    _LOG_PREFIX, f"Loaded {len(section)} user models for {backend}"
                )

    log.msg(
        _LOG_PREFIX,
        f"Registry loaded: {len(registry)} models across {len(_BACKEND_FILES)} backends",
    )
    return registry


# ============================================================================
# Public API
# ============================================================================


def load_all_registries(force: bool = False) -> Dict[str, Dict[str, Any]]:
    # Load (or reload) all registry files into the merged dict.
    # Thread-safe. Cached after first load unless force=True.
    global _merged_registry
    with _lock:
        if _merged_registry is None or force:
            _merged_registry = _build_registry()
        return _merged_registry


# Separator tokens for model dropdown grouping (matched in JS frontend)
MODEL_SEP_VISION = "__SEP__VISION_MODELS__"
MODEL_SEP_TEXT = "__SEP__TEXT_MODELS__"
MODEL_SEP_WD14 = "__SEP__WD14_MODELS__"
_MODEL_SEPARATORS = {
    MODEL_SEP_VISION, MODEL_SEP_TEXT, MODEL_SEP_WD14,
    MODEL_SEPARATOR_VISION, MODEL_SEPARATOR_TEXT,
}


def get_model_list() -> List[str]:
    # Get model display names grouped by type with separator tokens.
    # Registered vision/text/WD14, then native vision/text-only encoders.
    registry = load_all_registries()
    vision: List[str] = []
    text: List[str] = []
    wd14: List[str] = []
    for name, entry in registry.items():
        if entry.get("backend") == "wd14":
            wd14.append(name)
        elif entry.get("has_vision", False):
            vision.append(name)
        else:
            text.append(name)
    result: List[str] = []
    if vision:
        result.append(MODEL_SEP_VISION)
        result.extend(sorted(vision))
    if text:
        result.append(MODEL_SEP_TEXT)
        result.extend(sorted(text))
    if wd14:
        result.append(MODEL_SEP_WD14)
        result.extend(sorted(wd14))
    text_encoders = get_text_encoder_entries()
    for has_vision, separator in ((True, MODEL_SEPARATOR_VISION), (False, MODEL_SEPARATOR_TEXT)):
        names = sorted(name for name, entry in text_encoders.items()
                       if entry["has_vision"] == has_vision)
        if names:
            result.append(separator)
            result.extend(names)
    return result


def is_model_separator(name: str) -> bool:
    # Check if a name is a separator token (not a real model).
    return name in _MODEL_SEPARATORS


def get_model_entry(display_name: str) -> Optional[Dict[str, Any]]:
    # Look up a model by its display name (with suffix).
    # Returns the full entry dict with "backend" and "name" fields, or None.
    # Falls back to YOLO registry if not found in shared registry.
    if display_name.startswith(DISPLAY_PREFIX):
        return get_text_encoder_entry(display_name)
    registry = load_all_registries()
    entry = registry.get(display_name)
    if entry is not None:
        return entry
    # Fallback: check YOLO registry
    yolo_reg = _load_yolo_registry()
    return yolo_reg.get(display_name)


def resolve_model(display_name: str) -> Optional[Tuple[str, Dict[str, Any], str]]:
    # Resolve a display name to (backend, entry_data, clean_name).
    # Returns None if model not found.
    entry = get_model_entry(display_name)
    if entry is None:
        return None
    return (entry["backend"], entry, entry["name"])


def get_model_family(display_name: str) -> Optional[ModelFamily]:
    # Get the ModelFamily enum for a display name.
    entry = get_model_entry(display_name)
    if entry is None:
        return None
    family_str = entry.get("family", "")
    return FAMILY_MAP.get(family_str)


def get_backend_for_display_name(display_name: str) -> Optional[str]:
    # Extract backend key from a display name by checking suffix.
    # Falls back to registry lookup if no suffix matches.
    for suffix, backend in _SUFFIX_TO_BACKEND.items():
        if display_name.endswith(suffix):
            return backend
    # No suffix → could be Transformers or WD14
    entry = get_model_entry(display_name)
    if entry:
        return entry["backend"]
    return None


def is_wd14_model(display_name: str) -> bool:
    # Check if a display name is a WD14 tagger model.
    entry = get_model_entry(display_name)
    if entry:
        return entry["backend"] == "wd14"
    return display_name.startswith("WD14-")


def get_quantizations(display_name: str = "") -> List[str]:
    # Get the global GGUF quantization list from defaults.json.
    # Returns empty list for non-GGUF models (if display_name given).
    if display_name:
        entry = get_model_entry(display_name)
        if entry is None or entry.get("backend") not in ("gguf", "llamacpp"):
            return []
        entry_quantizations = entry.get("quantizations")
        if isinstance(entry_quantizations, list) and entry_quantizations:
            return list(entry_quantizations)
    defaults = load_defaults()
    return defaults.get("quantizations", ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])


def has_vision(display_name: str) -> bool:
    # Check if a model has vision capabilities.
    entry = get_model_entry(display_name)
    if entry is None:
        return False
    # WD14 models are always vision (no explicit field)
    if entry.get("backend") == "wd14":
        return True
    return entry.get("has_vision", False)


# ============================================================================
# Registry Manager Validation and Persistence
# ============================================================================


class RegistryValidationError(ValueError):
    # Raised when a registry-manager candidate is not safe to persist.
    pass


def _display_name_for(
    backend: str,
    name: str,
    detection_type: str | None = None,
) -> str:
    if backend == "yolo":
        return f"{name} [{detection_type or 'bbox'}]"
    suffix = _BACKEND_FILES[backend][1]
    return f"{name}{suffix}"


def _normalized_model_name(value: object) -> str:
    if not isinstance(value, str):
        raise RegistryValidationError("name must be a string")
    name = value.strip()
    if not name or len(name) > 200:
        raise RegistryValidationError("name must contain 1 to 200 characters")
    if name.startswith("_"):
        raise RegistryValidationError("names beginning with '_' are reserved")
    if name.startswith(DISPLAY_PREFIX):
        raise RegistryValidationError("The ComfyUI prefix is reserved for local text encoders")
    if any(character in name for character in ("/", "\\", "\x00", "\n", "\r")):
        raise RegistryValidationError("name contains an unsafe character")
    if name in _MODEL_SEPARATORS:
        raise RegistryValidationError("name is reserved for the model selector")
    return name


def _normalized_repo_id(value: object, backend: str, local_only: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise RegistryValidationError("repo_id must be a string")
    repo_id = value.strip()
    if local_only:
        return repo_id
    if not repo_id or len(repo_id) > 512 or "\x00" in repo_id:
        raise RegistryValidationError("repo_id is required")
    if backend == "ollama":
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*"
            r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?",
            repo_id,
        ):
            raise RegistryValidationError("repo_id is not a valid Ollama model ID")
        return repo_id
    if backend == "yolo" and repo_id.startswith(("http://", "https://")):
        match = re.fullmatch(
            r"https://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/.+",
            repo_id,
        )
        if not match:
            raise RegistryValidationError(
                "YOLO repository URL must be an HTTPS Hugging Face resolve URL"
            )
        repo_id = match.group(1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
        raise RegistryValidationError("repo_id must use canonical 'owner/model' form")
    return repo_id


def _normalized_repository_filename(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise RegistryValidationError(f"{field_name} must be a string")
    filename = value.strip().replace("\\", "/")
    if not filename or len(filename) > 512 or "\x00" in filename:
        raise RegistryValidationError(f"{field_name} is invalid")
    if any(part in {"", ".", ".."} for part in filename.split("/")):
        raise RegistryValidationError(f"{field_name} contains an unsafe path")
    path = PurePosixPath(filename)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RegistryValidationError(f"{field_name} contains an unsafe path")
    return path.as_posix()


def _normalized_expected_hashes(value: object) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise RegistryValidationError("expected_sha256 must be an object")
    result: dict[str, str] = {}
    for filename, digest in value.items():
        safe_filename = _normalized_repository_filename(
            filename, "expected_sha256 filename"
        )
        if not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", digest.strip()
        ):
            raise RegistryValidationError(
                f"expected_sha256 for '{safe_filename}' is not a SHA-256 digest"
            )
        result[safe_filename] = digest.strip().lower()
    return result


def _normalized_quantizations(value: object) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise RegistryValidationError("quantizations must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", item.strip()
        ):
            raise RegistryValidationError("quantizations contains an invalid value")
        normalized = item.strip()
        if normalized not in result:
            result.append(normalized)
    return result


def _normalized_server_quantization(value: object, backend: str) -> str:
    if value in (None, "", "auto", "none"):
        return ""
    if not isinstance(value, str):
        raise RegistryValidationError("quantization must be a string")
    normalized = value.strip().lower()
    supported = {
        "vllm": {"awq", "bitsandbytes", "fp8", "gptq"},
        "vllm_native": {"awq", "bitsandbytes", "fp8", "gptq", "squeezellm"},
        "sglang": {"awq", "fp8", "gptq"},
    }
    if normalized not in supported[backend]:
        raise RegistryValidationError(
            f"quantization is not supported by the {backend} backend"
        )
    return normalized


def _resolve_candidate_revision(entry: dict[str, Any]) -> None:
    if entry["backend"] == "ollama" or entry.get("local_only", False):
        entry.pop("revision", None)
        return
    source = entry.get("source", "huggingface")
    from .credentials import resolve_auth_token
    from .model_files import (
        resolve_huggingface_revision,
        resolve_modelscope_revision,
    )

    token = resolve_auth_token(source)
    if source == "modelscope":
        entry["revision"] = resolve_modelscope_revision(
            entry["repo_id"], entry.get("revision"), token
        )
    else:
        entry["revision"] = resolve_huggingface_revision(
            entry["repo_id"], entry.get("revision"), token
        )


def normalize_registry_candidate(
    candidate: object,
    *,
    resolve_revision: bool,
) -> dict[str, Any]:
    # Validate a guided registry-manager payload and return canonical fields.
    if not isinstance(candidate, dict):
        raise RegistryValidationError("entry must be a JSON object")

    backend_value = candidate.get("backend")
    if not isinstance(backend_value, str):
        raise RegistryValidationError("backend must be a string")
    backend = backend_value.strip().lower()
    if backend not in SUPPORTED_REGISTRY_BACKENDS:
        raise RegistryValidationError("backend is not supported by Smart LM")

    name = _normalized_model_name(candidate.get("name"))
    local_only = candidate.get("local_only", False)
    if not isinstance(local_only, bool):
        raise RegistryValidationError("local_only must be a boolean")
    if local_only and backend == "ollama":
        raise RegistryValidationError(
            "Ollama entries are managed by Ollama and cannot use local_only"
        )
    repo_id = _normalized_repo_id(candidate.get("repo_id", ""), backend, local_only)

    forced_family = {"wd14": "WD14", "yolo": "YOLO"}.get(backend)
    family_value = candidate.get("family", forced_family or "")
    if not isinstance(family_value, str) or family_value not in FAMILY_MAP:
        raise RegistryValidationError("family is not supported by Smart LM")
    family = family_value
    if forced_family:
        family = forced_family

    has_vision = candidate.get("has_vision", backend in {"wd14", "yolo"})
    if not isinstance(has_vision, bool):
        raise RegistryValidationError("has_vision must be a boolean")
    if backend in {"wd14", "yolo"}:
        has_vision = True

    normalized: dict[str, Any] = {
        "name": name,
        "backend": backend,
        "repo_id": repo_id,
        "family": family,
        "has_vision": has_vision,
    }

    description = candidate.get("description")
    if description not in (None, ""):
        if not isinstance(description, str) or len(description.strip()) > 2000:
            raise RegistryValidationError("description must be at most 2000 characters")
        normalized["description"] = description.strip()

    if backend == "yolo":
        filename = _normalized_repository_filename(
            candidate.get("filename"), "filename"
        )
        if "/" in filename or not filename.lower().endswith(".pt"):
            raise RegistryValidationError(
                "YOLO filename must be a .pt filename without directories"
            )
        detection_type = candidate.get("detection_type", "bbox")
        if detection_type not in {"bbox", "segm"}:
            raise RegistryValidationError("detection_type must be bbox or segm")
        normalized["filename"] = filename
        normalized["detection_type"] = detection_type

    if local_only and backend == "yolo":
        from .backend_yolo import resolve_yolo_model_path

        if resolve_yolo_model_path(normalized["filename"]) is None:
            raise RegistryValidationError(
                "Local-only YOLO file was not found in the configured "
                "ultralytics/bbox or ultralytics/segm folders"
            )
        normalized["local_only"] = True
        normalized["available"] = True
    elif local_only:
        from .model_acquisition import validate_local_registry_path

        normalized["local_only"] = True
        normalized["local_path"] = validate_local_registry_path(
            candidate.get("local_path"), require_exists=True
        )
    else:
        source = candidate.get("source", "huggingface")
        if not isinstance(source, str) or source.lower().strip() not in {
            "huggingface",
            "modelscope",
        }:
            raise RegistryValidationError("source must be huggingface or modelscope")
        source = source.lower().strip()
        if source == "modelscope" and backend not in {"gguf", "llamacpp"}:
            raise RegistryValidationError(
                "ModelScope is supported only for targeted GGUF entries"
            )
        if source != "huggingface":
            normalized["source"] = source

        revision = candidate.get("revision")
        if revision not in (None, ""):
            if not isinstance(revision, str) or len(revision.strip()) > 200:
                raise RegistryValidationError("revision is invalid")
            normalized["revision"] = revision.strip()

        hashes = _normalized_expected_hashes(candidate.get("expected_sha256"))
        if hashes:
            normalized["expected_sha256"] = hashes

        if resolve_revision:
            try:
                _resolve_candidate_revision(normalized)
            except Exception as error:
                raise RegistryValidationError(
                    f"Unable to resolve immutable repository revision: {type(error).__name__}"
                ) from error

    if backend in {"gguf", "llamacpp"}:
        file_pattern = candidate.get("file_pattern")
        if not local_only and (
            not isinstance(file_pattern, str) or "{quant}" not in file_pattern
        ):
            raise RegistryValidationError(
                "GGUF and llama.cpp entries require file_pattern with {quant}"
            )
        if isinstance(file_pattern, str) and file_pattern.strip():
            normalized["file_pattern"] = _normalized_repository_filename(
                file_pattern, "file_pattern"
            )
        quantizations = _normalized_quantizations(candidate.get("quantizations"))
        if quantizations:
            normalized["quantizations"] = quantizations
        mmproj = candidate.get("mmproj")
        if mmproj not in (None, ""):
            normalized["mmproj"] = _normalized_repository_filename(mmproj, "mmproj")

    if backend in {"vllm", "vllm_native", "sglang"}:
        server_quantization = _normalized_server_quantization(
            candidate.get("quantization"), backend
        )
        if server_quantization:
            normalized["quantization"] = server_quantization

    if backend in {"vllm", "sglang"}:
        tensor_parallel = candidate.get("tensor_parallel", 1)
        if not isinstance(tensor_parallel, int) or isinstance(tensor_parallel, bool):
            raise RegistryValidationError("tensor_parallel must be an integer")
        if not 1 <= tensor_parallel <= 64:
            raise RegistryValidationError("tensor_parallel must be between 1 and 64")
        normalized["tensor_parallel"] = tensor_parallel
    if backend == "sglang":
        data_parallel = candidate.get("data_parallel", 1)
        if not isinstance(data_parallel, int) or isinstance(data_parallel, bool):
            raise RegistryValidationError("data_parallel must be an integer")
        if not 1 <= data_parallel <= 64:
            raise RegistryValidationError("data_parallel must be between 1 and 64")
        normalized["data_parallel"] = data_parallel

    return normalized


def inspect_registry_candidate(
    candidate: object,
    *,
    original_display_name: str | None = None,
) -> dict[str, Any]:
    # Validate without writing. New/changed remote sources resolve immutably.
    original_entry = (
        get_model_entry(original_display_name) if original_display_name else None
    )
    should_resolve = original_entry is None
    if original_entry is not None and isinstance(candidate, dict):
        candidate_revision = candidate.get("revision") or None
        original_revision = original_entry.get("revision") or None
        should_resolve = (
            candidate.get("repo_id") != original_entry.get("repo_id")
            or candidate_revision != original_revision
            or candidate.get("source", "huggingface")
            != original_entry.get("source", "huggingface")
        )
    normalized = normalize_registry_candidate(
        candidate, resolve_revision=should_resolve
    )
    return {
        "display_name": _display_name_for(
            normalized["backend"],
            normalized["name"],
            normalized.get("detection_type"),
        ),
        **normalized,
    }


def _entry_storage(entry: dict[str, Any]) -> tuple[str, Path]:
    if entry.get("editable") is False:
        raise RegistryValidationError("ComfyUI text encoders are discovered from shared model folders and cannot be edited here")
    backend = entry["backend"]
    if backend == "yolo":
        return "curated", _REGISTRY_DIR / _YOLO_REGISTRY_FILE
    origin = entry.get("_registry_origin", "curated")
    if origin == "user":
        return "user", _REGISTRY_DIR / "user_models.json"
    return "curated", _REGISTRY_DIR / _BACKEND_FILES[backend][0]


def upsert_registry_entry(
    candidate: object,
    *,
    original_display_name: str | None = None,
) -> dict[str, Any]:
    # Add a user entry or atomically edit its existing runtime registry entry.
    original_entry = (
        get_model_entry(original_display_name) if original_display_name else None
    )
    if original_display_name and original_entry is None:
        raise RegistryValidationError("Original registry entry was not found")
    inspected = inspect_registry_candidate(
        candidate, original_display_name=original_display_name
    )
    backend = inspected["backend"]
    name = inspected["name"]
    new_display_name = inspected["display_name"]

    if original_entry is not None and original_entry["backend"] != backend:
        raise RegistryValidationError(
            "An existing registry entry cannot change backend; add a new entry instead"
        )

    existing = get_model_entry(new_display_name)
    if existing is not None and new_display_name != original_display_name:
        raise RegistryValidationError(
            f"A registry entry already uses display name '{new_display_name}'"
        )

    if backend == "yolo":
        origin = "curated"
        path = _REGISTRY_DIR / _YOLO_REGISTRY_FILE
        old_name = original_entry["name"] if original_entry is not None else None
    elif original_entry is None:
        origin = "user"
        path = _REGISTRY_DIR / "user_models.json"
        old_name = None
    else:
        origin, path = _entry_storage(original_entry)
        old_name = original_entry["name"]

    persisted = {
        key: copy.deepcopy(value)
        for key, value in inspected.items()
        if key not in {"display_name", "name", "backend"}
    }
    if backend == "yolo":
        persisted["_registry_origin"] = (
            original_entry.get("_registry_origin", "curated")
            if original_entry is not None
            else "user"
        )

    def apply_update(data: dict[str, Any]) -> None:
        target = data
        if origin == "user":
            section = data.setdefault(backend, {})
            if not isinstance(section, dict):
                raise JsonStoreError(f"user_models.json section '{backend}' is invalid")
            target = section
        if old_name is None and name in target:
            raise RegistryValidationError(
                f"A registry entry already uses display name '{new_display_name}'"
            )
        if old_name is not None and old_name not in target:
            raise RegistryValidationError(
                "Original registry entry no longer exists on disk"
            )
        if old_name is not None and old_name != name and name in target:
            raise RegistryValidationError(
                f"A registry entry already uses display name '{new_display_name}'"
            )
        if old_name and old_name != name:
            target.pop(old_name, None)
        target[name] = copy.deepcopy(persisted)

    update_json_object(path, apply_update, default={})
    if backend == "yolo":
        invalidate_yolo_cache()
        _load_yolo_registry(force=True)
    else:
        invalidate_cache()
        load_all_registries(force=True)
    result = get_model_entry_for_api(new_display_name)
    if result is None:
        raise RuntimeError("Registry entry was committed but could not be reloaded")
    return result


def remove_registry_entry(display_name: str) -> dict[str, Any]:
    # Remove one entry from its owning runtime registry file only.
    entry = get_model_entry(display_name)
    if entry is None:
        raise RegistryValidationError("Registry entry was not found")
    origin, path = _entry_storage(entry)
    backend = entry["backend"]
    name = entry["name"]
    removed = False

    def apply_remove(data: dict[str, Any]) -> None:
        nonlocal removed
        target = data
        if origin == "user":
            section = data.get(backend)
            if not isinstance(section, dict):
                raise JsonStoreError(f"user_models.json section '{backend}' is invalid")
            target = section
        if name in target:
            del target[name]
            removed = True

    update_json_object(path, apply_remove)
    if not removed:
        raise RegistryValidationError("Registry entry no longer exists on disk")
    if backend == "yolo":
        invalidate_yolo_cache()
        _load_yolo_registry(force=True)
    else:
        invalidate_cache()
        load_all_registries(force=True)
    return {
        "success": True,
        "display_name": display_name,
        "origin": origin,
    }


# ============================================================================
# Defaults Persistence
# ============================================================================


def _defaults_path() -> Path:
    return _REGISTRY_DIR / "defaults.json"


def _path_signature(path: Path) -> tuple[int, int]:
    file_stat = path.stat()
    return file_stat.st_mtime_ns, file_stat.st_size


def invalidate_defaults_cache() -> None:
    # Force defaults.json to be read again on next access.
    global _defaults_cache, _defaults_signature
    with _defaults_lock:
        _defaults_cache = None
        _defaults_signature = None


def load_defaults() -> Dict[str, Any]:
    # Load global defaults from registry/defaults.json.
    # Caches one detached generation and reloads after file replacement.
    global _defaults_cache, _defaults_signature
    path = _defaults_path()
    with _defaults_lock:
        try:
            signature = _path_signature(path)
        except OSError:
            _defaults_cache = None
            _defaults_signature = None
            return {}

        if _defaults_cache is not None and signature == _defaults_signature:
            return copy.deepcopy(_defaults_cache)

        try:
            data = read_json_object(path)
        except (JsonStoreError, OSError) as error:
            _defaults_cache = None
            _defaults_signature = None
            log.error(_LOG_PREFIX, f"Failed to load {path.name}: {error}")
            return {}

        try:
            final_signature = _path_signature(path)
        except OSError:
            return data

        # Cache only when the file generation stayed stable across the read.
        # A concurrent replacement remains a valid snapshot but must be reread.
        if final_signature == signature:
            _defaults_cache = copy.deepcopy(data)
            _defaults_signature = final_signature
        else:
            _defaults_cache = None
            _defaults_signature = None
        return copy.deepcopy(data)


def get_default(key: str, fallback: Any = None) -> Any:
    # Get a single default value.
    return load_defaults().get(key, fallback)


def save_defaults(updates: Dict[str, Any]) -> bool:
    # Persist updated default values to registry/defaults.json.
    # Only writes if at least one value actually changed.
    if not isinstance(updates, dict):
        log.error(_LOG_PREFIX, "Defaults update must be a JSON object")
        return False
    if not updates:
        return False

    path = _defaults_path()
    changed: list[str] = []
    try:
        def apply_updates(current: dict[str, Any]) -> None:
            for key, value in updates.items():
                if current.get(key) != value:
                    current[key] = copy.deepcopy(value)
                    changed.append(key)

        update_json_object(path, apply_updates, default={})
        # Do not publish the returned snapshot: a newer writer may commit after
        # the JSON lock is released but before this thread reaches the cache.
        invalidate_defaults_cache()
        if not changed:
            return False
        log.debug(_LOG_PREFIX, f"Saved defaults: {', '.join(changed)}")
        return True
    except (JsonStoreError, OSError) as error:
        invalidate_defaults_cache()
        log.error(_LOG_PREFIX, f"Failed to save defaults: {error}")
        return False


# ============================================================================
# Serialization for Endpoints
# ============================================================================


def get_model_list_for_api() -> List[Dict[str, Any]]:
    # Build the model list payload for the /smartllm-nova/model_list endpoint.
    # Returns a list of dicts with display_name, backend, family, has_vision,
    # and quantizations (if any).
    registry = {**load_all_registries(), **_load_yolo_registry(), **get_text_encoder_entries()}
    result = []
    for display_name in sorted(registry.keys()):
        entry = registry[display_name]
        if entry.get("backend") == "ollama":
            local_status = "managed"
        else:
            try:
                from .model_acquisition import resolve_registered_model_path

                quantizations = (
                    get_quantizations(display_name)
                    if entry.get("backend") in {"gguf", "llamacpp"}
                    else [None]
                )
                if not quantizations:
                    quantizations = [None]
                needs_download = True
                for quantization in quantizations:
                    _, candidate_needs_download = resolve_registered_model_path(
                        entry,
                        quantization,
                    )
                    if not candidate_needs_download:
                        needs_download = False
                        break
                local_status = "missing" if needs_download else "available"
            except (FileNotFoundError, OSError, TypeError, ValueError):
                local_status = "missing"
        item = {
            "display_name": display_name,
            "backend": entry["backend"],
            "name": entry["name"],
            "family": entry.get(
                "family", "WD14" if entry["backend"] == "wd14" else ""
            ),
            "has_vision": entry.get("has_vision", entry["backend"] == "wd14"),
            "origin": entry.get("_registry_origin", "curated"),
            "editable": entry.get("editable", True),
            "can_delete": entry.get("can_delete", True),
            "local_status": local_status,
            "can_download": not entry.get("local_only", False),
        }
        if "quantizations" in entry:
            item["quantizations"] = entry["quantizations"]
        result.append(item)
    return result


def get_model_entry_for_api(display_name: str) -> Optional[Dict[str, Any]]:
    # Build the model entry payload for the /smartllm-nova/model_entry endpoint.
    # Returns the full entry dict (safe for JSON serialization).
    # GGUF models get quantizations injected from defaults.json.
    entry = get_model_entry(display_name)
    if entry is None:
        return None
    result = {"display_name": display_name}
    result.update(
        {
            key: copy.deepcopy(value)
            for key, value in entry.items()
            if not key.startswith("_registry_")
        }
    )
    result["origin"] = entry.get("_registry_origin", "curated")
    result["editable"] = entry.get("editable", True)
    result["can_delete"] = entry.get("can_delete", True)
    result["can_download"] = not entry.get("local_only", False)
    if entry.get("backend") in ("gguf", "llamacpp"):
        result["quantizations"] = get_quantizations(display_name)
    return result


def invalidate_cache():
    # Force the registry to reload on next access.
    global _merged_registry
    with _lock:
        _merged_registry = None
    log.debug(_LOG_PREFIX, "Registry cache invalidated")


# ============================================================================
# YOLO Registry (separate from shared SmartLLM registry)
# ============================================================================

_YOLO_REGISTRY_FILE = "yolo_models.json"
_RETIRED_CURATED_YOLO_ARTIFACTS = {
    ("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt"),
    ("Bingsu/adetailer", "person_yolov8m-seg.pt"),
    ("Bingsu/adetailer", "person_yolov8n-seg.pt"),
    ("Bingsu/adetailer", "person_yolov8s-seg.pt"),
    ("Anzhc/Anzhcs_YOLOs", "Anzhc Eyes -seg-hd.pt"),
    ("Anzhc/Anzhcs_YOLOs", "Anzhc Face -seg.pt"),
    ("Anzhc/Anzhcs_YOLOs", "Anzhc HeadHair seg y8m.pt"),
    ("Anzhc/Anzhcs_YOLOs", "Anzhc HeadHair seg y8n.pt"),
    ("Anzhc/Anzhcs_YOLOs", "Anzhc Manga Panels -seg.pt"),
}

# Module-level YOLO registry cache
_yolo_registry: Optional[Dict[str, Dict[str, Any]]] = None
_yolo_lock = threading.Lock()

# Separator tokens for detection model dropdown
MODEL_SEP_DETECTION_VLM = "__SEP__DETECTION_VLM__"
MODEL_SEP_YOLO = "__SEP__YOLO__"


def _yolo_repository_id(repo_id: object) -> str | None:
    # Normalize supported YOLO registry repository forms for provenance lookup.
    if not isinstance(repo_id, str) or not repo_id:
        return None
    if repo_id.startswith(("http://", "https://")):
        match = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/", repo_id)
        return match.group(1) if match else None
    parts = repo_id.split("/", 2)
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None


def _is_retired_curated_yolo_entry(entry: dict[str, Any]) -> bool:
    # User-added entries remain under operator control; this gate retires only
    # the exact shipped artifacts confirmed unsafe by Hugging Face scanning.
    if entry.get("_registry_origin") == "user":
        return False
    identity = (_yolo_repository_id(entry.get("repo_id")), entry.get("filename"))
    return identity in _RETIRED_CURATED_YOLO_ARTIFACTS


def _load_yolo_registry(force: bool = False) -> Dict[str, Dict[str, Any]]:
    # Load the YOLO registry JSON and build a flat lookup dict.
    # Display names: "{name} [{detection_type}]" (e.g. "face_yolov8m [bbox]")
    # Thread-safe, cached.
    global _yolo_registry
    with _yolo_lock:
        if _yolo_registry is not None and not force:
            return _yolo_registry

        path = _REGISTRY_DIR / _YOLO_REGISTRY_FILE
        raw = _load_json(path)
        registry: Dict[str, Dict[str, Any]] = {}
        default_raw = _load_json(_YOLO_DEFAULT_REGISTRY_PATH)
        default_provenance = default_raw.get("_repositories", {})
        active_provenance = raw.get("_repositories", {})
        repository_provenance = (
            copy.deepcopy(default_provenance)
            if isinstance(default_provenance, dict)
            else {}
        )
        if isinstance(active_provenance, dict):
            repository_provenance.update(active_provenance)

        for name, entry in raw.items():
            if name.startswith("_") or not isinstance(entry, dict):
                continue
            if _is_retired_curated_yolo_entry(entry):
                continue
            resolved_entry = dict(entry)
            repository_id = _yolo_repository_id(entry.get("repo_id"))
            provenance = repository_provenance.get(repository_id, {})
            if isinstance(provenance, dict):
                revision = provenance.get("revision")
                expected_hashes = provenance.get("expected_sha256")
                filename = entry.get("filename")
                if revision and "revision" not in resolved_entry:
                    resolved_entry["revision"] = revision
                if (
                    isinstance(filename, str)
                    and isinstance(expected_hashes, dict)
                    and filename in expected_hashes
                    and "expected_sha256" not in resolved_entry
                ):
                    resolved_entry["expected_sha256"] = {
                        filename: expected_hashes[filename]
                    }
            det_type = entry.get("detection_type", "bbox")
            display_name = f"{name} [{det_type}]"
            registry[display_name] = {
                "backend": "yolo",
                "name": name,
                "_registry_origin": (
                    "local" if entry.get("local_only", False) else "curated"
                ),
                **resolved_entry,
            }

        _yolo_registry = registry
        log.debug(_LOG_PREFIX, f"YOLO registry loaded: {len(registry)} models")
        return registry


def sync_yolo_registry():
    # Sync registry/yolo_models.json with on-disk YOLO model files.
    # Called at startup. Adds newly discovered models, removes stale local_only entries,
    # and updates availability of curated entries.
    from .backend_yolo import get_yolo_model_files

    path = _REGISTRY_DIR / _YOLO_REGISTRY_FILE
    on_disk = get_yolo_model_files()
    default_raw = _load_json(_YOLO_DEFAULT_REGISTRY_PATH)
    curated_repositories = default_raw.get("_repositories", {})
    if not isinstance(curated_repositories, dict):
        curated_repositories = {}
    disk_files: Dict[str, str] = {}  # {filename: detection_type}
    for det_type, files in on_disk.items():
        for f in files:
            disk_files[f] = det_type

    changed = False
    debug_messages: list[str] = []
    try:
        def apply_disk_state(registry: dict[str, Any]) -> None:
            nonlocal changed

            if registry.get("_repositories") != curated_repositories:
                registry["_repositories"] = copy.deepcopy(curated_repositories)
                changed = True
                debug_messages.append("Updated curated YOLO provenance pins")

            # Retire only the exact former shipped entries. Preserve existing
            # checkpoint bytes by converting an installed artifact to a local-only
            # discovery; a missing artifact disappears from the runtime registry.
            for key, entry in list(registry.items()):
                if (
                    key.startswith("_")
                    or not isinstance(entry, dict)
                    or not _is_retired_curated_yolo_entry(entry)
                ):
                    continue
                filename = entry.get("filename", f"{key}.pt")
                if filename in disk_files:
                    registry[key] = {
                        "filename": filename,
                        "family": "YOLO",
                        "detection_type": disk_files[filename],
                        "description": f"Auto-discovered: {filename}",
                        "local_only": True,
                        "available": True,
                    }
                    debug_messages.append(
                        f"Converted retired curated YOLO model to local-only: {key}"
                    )
                else:
                    del registry[key]
                    debug_messages.append(f"Removed retired curated YOLO model: {key}")
                changed = True

            # Build filename → registry key lookup from the latest generation.
            filename_to_key: dict[str, str] = {}
            for key, entry in registry.items():
                if key.startswith("_") or not isinstance(entry, dict):
                    continue
                filename = entry.get("filename", f"{key}.pt")
                filename_to_key[filename] = key

            # Add newly discovered models as local-only entries.
            for filename, detection_type in disk_files.items():
                if filename in filename_to_key:
                    continue
                name = os.path.splitext(filename)[0]
                if name.startswith("_"):
                    continue
                registry[name] = {
                    "filename": filename,
                    "family": "YOLO",
                    "detection_type": detection_type,
                    "description": f"Auto-discovered: {filename}",
                    "local_only": True,
                    "available": True,
                }
                filename_to_key[filename] = name
                changed = True
                debug_messages.append(
                    f"Auto-discovered YOLO model: {filename} ({detection_type})"
                )

            keys_to_remove: list[str] = []
            for key, entry in list(registry.items()):
                if key.startswith("_") or not isinstance(entry, dict):
                    continue
                filename = entry.get("filename", f"{key}.pt")
                is_on_disk = filename in disk_files

                if entry.get("local_only", False):
                    if not is_on_disk:
                        keys_to_remove.append(key)
                        changed = True
                        debug_messages.append(
                            f"Removing stale local YOLO model: {key}"
                        )
                elif entry.get("available", False) != is_on_disk:
                    entry["available"] = is_on_disk
                    changed = True

            for key in keys_to_remove:
                del registry[key]

        snapshot = update_json_object(
            path,
            apply_disk_state,
            default={},
            indent=4,
        )
        if changed:
            for message in debug_messages:
                log.debug(_LOG_PREFIX, message)
            log.msg(
                _LOG_PREFIX,
                f"YOLO registry synced: {len([k for k in snapshot if not k.startswith('_')])} models",
            )
    except (JsonStoreError, OSError) as error:
        log.error(_LOG_PREFIX, f"Failed to sync YOLO registry: {error}")
    finally:
        # A failed sync must not leave a cached generation trusted indefinitely.
        invalidate_yolo_cache()


def get_detection_model_list() -> List[str]:
    # Get detection-capable models for the Detection node dropdown.
    # Combines registered VLMs, native vision encoders and YOLO models.
    # Returns list with separator tokens for frontend grouping.
    shared_registry = load_all_registries()
    yolo_registry = _load_yolo_registry()

    # Filter shared registry to detection-capable VLM families
    detection_families = {"Florence", "Qwen"}
    vlm_models: List[str] = []
    for name, entry in shared_registry.items():
        family = entry.get("family", "")
        if family in detection_families and entry.get("has_vision", False):
            vlm_models.append(name)

    # Curated YOLO entries stay selectable so choosing a missing model can enter
    # the verified acquisition path. Local-only discoveries remain selectable
    # only while their artifact is present.
    yolo_models: List[str] = []
    for name, entry in yolo_registry.items():
        if not entry.get("local_only", False) or entry.get("available", False):
            yolo_models.append(name)

    # Build grouped list with separators
    result: List[str] = []
    if vlm_models:
        result.append(MODEL_SEP_DETECTION_VLM)
        result.extend(sorted(vlm_models))
    native_vision = sorted(name for name, entry in get_text_encoder_entries().items()
                           if entry.get("has_vision", False))
    if native_vision:
        result.append(MODEL_SEPARATOR_VISION)
        result.extend(native_vision)
    if yolo_models:
        result.append(MODEL_SEP_YOLO)
        result.extend(sorted(yolo_models))

    return result


def is_yolo_model(display_name: str) -> bool:
    # Check if a display name refers to a YOLO model.
    yolo_registry = _load_yolo_registry()
    return display_name in yolo_registry


def invalidate_yolo_cache():
    # Force the YOLO registry to reload on next access.
    global _yolo_registry
    with _yolo_lock:
        _yolo_registry = None
    log.debug(_LOG_PREFIX, "YOLO registry cache invalidated")
