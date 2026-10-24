"""Local GGUF language encoders and explicitly matched vision companions."""

import re
from pathlib import Path
from types import SimpleNamespace

from ...extern.gguf.mappings import GEMMA3_SD_MAP, LLAMA_SD_MAP, sd_map_replace
from .gguf_metadata import GGUFHeader, read_gguf_header

# Qualified native generation layouts: (GGUF architecture, hidden size) ->
# (ComfyUI detector result, layer count, vocabulary). No diffusion-only T5/CLIP.
_LAYOUTS = {
    ("qwen2vl", 3584): ("QWEN25_7B", 28, 152064),
    ("qwen3", 1024): ("QWEN3_06B", 28, 151936),
    ("qwen3", 2560): ("QWEN3_4B", 36, 151936),
    ("qwen3", 4096): ("QWEN3_8B", 36, 151936),
    ("qwen3vl", 2560): ("QWEN3VL_4B", 36, 151936),
    ("qwen3vl", 4096): ("QWEN3VL_8B", 36, 151936),
    ("gemma3", 2560): ("GEMMA_3_4B", 34, 262208),
    ("gemma3", 3840): ("GEMMA_3_12B", 48, 262208),
}


def _model_stem(path: Path) -> str:
    stem = path.stem.casefold()
    stem = re.sub(r"[._-](?:ud-)?(?:i?q\d[\w-]*|bf16|f16|f32)$", "", stem)
    stem = re.sub(r"^mmproj[._-]|[._-]mmproj$", "", stem)
    return stem


def _compatible_projector(header: GGUFHeader, arch: str, hidden: int) -> bool:
    meta, shapes = header.metadata, header.shapes
    if (meta.get("general.architecture") != "clip"
            or meta.get("general.type") not in {"mmproj", "clip-vision"}
            or meta.get("clip.has_vision_encoder") is not True
            or meta.get("clip.vision.projection_dim") != hidden):
        return False
    if arch == "qwen2vl":
        kind, width, layers = "qwen2.5vl_merger", 1280, 32
    elif arch == "qwen3vl":
        kind, width, layers = "qwen3vl_merger", (1024 if hidden == 2560 else 1152), (24 if hidden == 2560 else 27)
    elif arch == "gemma3":
        kind, width, layers = "gemma3", 1152, 27
    else:
        return False
    if (meta.get("clip.projector_type") != kind
            or meta.get("clip.vision.embedding_length") != width
            or meta.get("clip.vision.block_count") != layers):
        return False
    if arch == "gemma3":
        required = {"mm.input_projection.weight": (width, hidden),
                    "mm.soft_emb_norm.weight": (width,),
                    "v.patch_embd.weight": (width, 3, 14, 14),
                    "v.position_embd.weight": (4096, width)}
    else:
        required = {"mm.2.weight": (hidden, width * 4),
                    "mm.0.weight": (width * 4, width * 4), "v.post_ln.weight": (width,)}
        patch_size = 16 if arch == "qwen3vl" else 14
        if arch == "qwen3vl":
            required["v.position_embd.weight"] = (2304, width)
        patch = shapes.get("v.patch_embd.weight")
        if patch != (width, 3, 2, patch_size, patch_size) and not (
                patch == (width, 3, patch_size, patch_size) and shapes.get("v.patch_embd.weight.1") == patch):
            return False
    if any(shapes.get(key) != shape for key, shape in required.items()):
        return False
    for index in range(layers):
        prefix = f"v.blk.{index}."
        if shapes.get(prefix + "attn_out.weight") != (width, width):
            return False
        if arch == "qwen3vl":
            if shapes.get(prefix + "attn_qkv.weight") != (3 * width, width):
                return False
        elif any(shapes.get(prefix + f"attn_{part}.weight") != (width, width) for part in "qkv"):
            return False
        if any(prefix + name not in shapes for name in ("ln1.weight", "ln2.weight", "ffn_up.weight", "ffn_down.weight")):
            return False
    return arch != "qwen3vl" or any("deepstack" in name or "deepstast" in name for name in shapes)


def _projector(path: Path, roots: list[Path], arch: str, hidden: int) -> tuple[Path | None, str]:
    from .text_encoders import file_signature

    if arch == "qwen3":
        return None, "Text-only architecture."
    candidates, incompatible = [], False
    for candidate in sorted(path.parent.iterdir()):
        if (candidate.suffix.lower() != ".gguf" or "mmproj" not in candidate.stem.casefold()
                or _model_stem(candidate) != _model_stem(path)):
            continue
        resolved = candidate.resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            incompatible = True
            continue
        try:
            header = read_gguf_header(file_signature(resolved))
            if _compatible_projector(header, arch, hidden):
                candidates.append(resolved)
            else:
                incompatible = True
        except (OSError, ValueError, KeyError):
            incompatible = True
    # Identical symlink targets are one companion, not an ambiguous choice.
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0], "Vision projector: " + candidates[0].name
    if len(candidates) > 1:
        return None, "Text only: multiple matching projectors. Keep one matching mmproj beside this encoder."
    if incompatible:
        return None, "Text only: the matching mmproj is incompatible or incomplete."
    return None, "Text only: place the model's matching mmproj beside this encoder, using the same base filename."


def inspect_gguf_encoder(path: Path) -> dict | None:
    import comfy.sd
    import folder_paths

    from .text_encoders import file_signature

    if not callable(getattr(getattr(comfy.sd, "CLIP", None), "generate", None)):
        return None
    header = read_gguf_header(file_signature(path))
    meta, shapes = header.metadata, header.shapes
    arch = meta.get("general.architecture")
    embedding = shapes.get("token_embd.weight", ())
    if len(embedding) != 2 or meta.get("general.type", "model") != "model":
        return None
    vocabulary, hidden = embedding
    layout = _LAYOUTS.get((arch, hidden))
    if layout is None:
        return None
    expected, layers, expected_vocabulary = layout
    if vocabulary != expected_vocabulary or meta.get(f"{arch}.block_count") != layers:
        return None
    if meta.get(f"{arch}.embedding_length") != hidden or meta.get("split.count", 1) != 1:
        return None
    blocks = {int(match[1]) for name in shapes if (match := re.match(r"blk\.(\d+)\.", name))}
    if blocks != set(range(layers)) or shapes.get("output_norm.weight") != (hidden,):
        return None
    names = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_gate", "ffn_down", "attn_norm", "ffn_norm")
    if any(f"blk.{index}.{name}.weight" not in shapes for index in range(layers) for name in names):
        return None
    if expected in {"QWEN25_7B", "QWEN3_8B", "QWEN3VL_8B"} and shapes.get("output.weight") != embedding:
        return None
    if arch == "gemma3" and any(header.arrays.get("tokenizer.ggml." + name) != vocabulary
                               for name in ("tokens", "scores", "token_type")):
        return None
    roots = [Path(root).resolve() for root in folder_paths.get_folder_paths("text_encoders")]
    projector, note = _projector(path, roots, arch, hidden)
    # Keep the installed ComfyUI detector as the final compatibility authority.
    mapped = sd_map_replace({key: SimpleNamespace(shape=value) for key, value in shapes.items()},
                            GEMMA3_SD_MAP if arch == "gemma3" else LLAMA_SD_MAP)
    if arch == "qwen3vl":
        mapped["model.visual.deepstack_merger_list.0.norm.weight"] = SimpleNamespace(shape=(hidden,))
        mapped["model.visual.merger.linear_fc2.weight"] = SimpleNamespace(shape=(hidden, hidden))
    elif arch == "gemma3" and projector:
        mapped["vision_model.embeddings.patch_embedding.weight"] = SimpleNamespace(shape=(1152, 3, 14, 14))
        if expected == "GEMMA_3_4B":
            expected = "GEMMA_3_4B_VISION"
    if getattr(comfy.sd.detect_te_model(mapped), "name", "") != expected:
        return None
    return {"architecture": expected, "has_vision": projector is not None,
            "format": "gguf", "projector_filename": projector.name if projector else None,
            "projector_note": note, "_projector_path": str(projector) if projector else None}


def load_gguf_clip(path: Path, projector: Path | None, *, clip_type: object,
                   embeddings: list[str], load_device: object, offload_device: object, architecture: str) -> object:
    import comfy.model_management as mm
    import comfy.sd
    import torch

    from ...extern.gguf.loader import gguf_clip_loader
    from ...extern.gguf.ops import GGMLOps
    from ...extern.gguf.patcher import GGUFModelPatcher
    from .comfy_qwen25 import generation_options

    state = gguf_clip_loader(str(path), projector_path=str(projector) if projector else None)
    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=[state], embedding_directory=embeddings, clip_type=clip_type,
        model_options=generation_options(architecture, {"custom_operations": GGMLOps, "initial_device": torch.device("cpu"),
                       "load_device": load_device, "offload_device": offload_device}),
    )
    try:
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
    except BaseException:
        mm.unload_model_and_clones(clip.patcher, all_devices=True)
        raise
    return clip
