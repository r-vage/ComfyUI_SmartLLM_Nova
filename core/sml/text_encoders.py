"""Read-only discovery of ComfyUI language-model text encoder checkpoints."""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

BACKEND = "comfyui"
DISPLAY_PREFIX = "ComfyUI: "
MODEL_SEPARATOR = "__SEP__TEXT_ENCODERS__"

# ComfyUI's detector remains the architecture authority. These are the native
# generation implementations, not CLIP/T5 or truncated diffusion-only encoders.
_GENERATIVE_TYPES = {
    "GEMMA_2_2B", "GEMMA_3_4B", "GEMMA_3_4B_VISION", "GEMMA_3_12B",
    "GEMMA_4_E2B", "GEMMA_4_E4B", "GEMMA_4_12B", "GEMMA_4_31B",
    "QWEN25_7B", "QWEN3_06B", "QWEN3_4B", "QWEN3_8B",
    "QWEN35_08B", "QWEN35_2B", "QWEN35_4B", "QWEN35_9B", "QWEN35_27B",
    "QWEN3VL_4B", "QWEN3VL_8B",
}
_UNTIED_TYPES = {"QWEN3_8B", "QWEN3VL_8B", "QWEN35_9B", "QWEN35_27B"}


def file_signature(path: Path) -> tuple:
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def resolve_text_encoder_path(filename: str) -> Path:
    """Resolve only a listed checkpoint within ComfyUI's configured roots."""
    import folder_paths

    if (not isinstance(filename, str) or not filename or "\x00" in filename
            or any(part in {"", ".", ".."} for part in filename.replace("\\", "/").split("/"))
            or Path(filename).is_absolute()):
        raise ValueError("Choose a text encoder from ComfyUI's text_encoders list.")
    if Path(filename).suffix.lower() not in {".safetensors", ".sft"}:
        raise ValueError("Native text generation requires a supported safetensors text encoder.")
    if filename not in folder_paths.get_filename_list("text_encoders"):
        raise FileNotFoundError(f"Text encoder is no longer available: {filename}")
    full_path = folder_paths.get_full_path("text_encoders", filename)
    if full_path is None:
        raise FileNotFoundError(f"Text encoder is no longer available: {filename}")
    path = Path(full_path).resolve()
    roots = [Path(root).resolve() for root in folder_paths.get_folder_paths("text_encoders")]
    if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
        raise ValueError("Text encoder must remain inside a configured text_encoders folder.")
    return path


@lru_cache(maxsize=256)
def _inspect_checkpoint(signature: tuple) -> tuple[str, bool] | None:
    # Read tensor names/shapes only. Never materialize model weights for a menu.
    import comfy.sd
    from safetensors import safe_open

    if not callable(getattr(getattr(comfy.sd, "CLIP", None), "generate", None)):
        return None
    with safe_open(signature[0], framework="pt", device="cpu") as checkpoint:
        shapes = {key: SimpleNamespace(shape=checkpoint.get_slice(key).get_shape())
                  for key in checkpoint.keys()}  # noqa: SIM118 - safe_open is not iterable
    model_type = comfy.sd.detect_te_model(shapes)
    architecture = getattr(model_type, "name", "")
    if architecture not in _GENERATIVE_TYPES:
        return None
    # ACE music checkpoints share Qwen's detector markers but use a different
    # vocabulary and native loader. They are not language-generation encoders.
    embedding = shapes.get("model.embed_tokens.weight")
    if (architecture in {"QWEN3_06B", "QWEN3_4B", "QWEN3_8B"}
            and embedding is not None and embedding.shape[0] != 151936):
        return None
    # Native Qwen8B requires an independent output projection. Some diffusion
    # conversions omit it; do not run an uninitialized head on those files.
    requires_head = architecture in _UNTIED_TYPES
    if requires_head and not any(key in shapes for key in ("lm_head.weight", "model.lm_head.weight")):
        return None
    if architecture.startswith("GEMMA_"):
        tokenizer_key = "tokenizer_json" if architecture.startswith("GEMMA_4_") else "spiece_model"
        if tokenizer_key not in shapes:
            return None
    has_vision = any(key.startswith(("vision_model.", "visual.", "model.visual.")) for key in shapes)
    return architecture, has_vision


def _entry(filename: str) -> dict | None:
    path = resolve_text_encoder_path(filename)
    inspected = _inspect_checkpoint(file_signature(path))
    if inspected is None:
        return None
    architecture, has_vision = inspected
    return {
        "backend": BACKEND, "name": filename, "filename": filename,
        "family": "VLM" if has_vision else "LLM_TEXT", "has_vision": has_vision,
        "architecture": architecture, "local_only": True,
        "_registry_origin": "text_encoders", "editable": False,
        "can_delete": False, "can_download": False,
    }


def get_text_encoder_entry(display_name: str) -> dict | None:
    if not isinstance(display_name, str) or not display_name.startswith(DISPLAY_PREFIX):
        return None
    from safetensors import SafetensorError

    try:
        return _entry(display_name[len(DISPLAY_PREFIX):])
    except (OSError, ValueError, KeyError, IndexError, SafetensorError):
        return None


def get_text_encoder_entries() -> dict[str, dict]:
    try:
        import folder_paths
    except ImportError:
        return {}
    # Standalone tooling may provide only a minimal folder_paths shim.
    if not hasattr(folder_paths, "get_filename_list"):
        return {}
    from safetensors import SafetensorError

    entries = {}
    for filename in folder_paths.get_filename_list("text_encoders"):
        if Path(filename).suffix.lower() not in {".safetensors", ".sft"}:
            continue
        try:
            entry = _entry(filename)
        except (OSError, ValueError, KeyError, IndexError, SafetensorError):
            continue
        if entry is not None:
            entries[DISPLAY_PREFIX + filename] = entry
    return entries
