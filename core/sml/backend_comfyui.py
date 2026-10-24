"""Text generation through ComfyUI's native CLIP/text encoder API."""

from dataclasses import dataclass
from pathlib import Path

import torch

from .comfy_qwen25 import generation_options
from .common import clean_model_output
from .config_templates import get_llm_few_shot_examples
from .device import resolve_requested_device
from .logger import log
from .tasks import get_system_prompt
from .text_encoders import (
    file_signature,
    resolve_text_encoder_path,
    text_encoder_signature,
)


@dataclass
class ComfyTextEncoder:
    clip: object
    architecture: str
    has_vision: bool
    cache_key: tuple
    is_comfyui: bool = True


_cached_encoder: ComfyTextEncoder | None = None


def release_text_encoder(instance: ComfyTextEncoder | None) -> None:
    """Unload only the patcher owned by this backend; never purge other nodes."""
    global _cached_encoder
    if instance is None:
        return
    if _cached_encoder is instance:
        _cached_encoder = None
    clip, instance.clip = instance.clip, None
    if clip is not None:
        import comfy.model_management as mm

        mm.unload_model_and_clones(clip.patcher, all_devices=True)


def clear_text_encoder_cache() -> None:
    release_text_encoder(_cached_encoder)


def load_text_encoder(entry: dict, *, device: str, keep_model_loaded: bool) -> ComfyTextEncoder:
    global _cached_encoder
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths

    path = resolve_text_encoder_path(entry["filename"])
    projector = None
    signature = file_signature(path)
    is_gguf = path.suffix.lower() == ".gguf"
    if is_gguf:
        from .gguf_text_encoders import inspect_gguf_encoder

        details = inspect_gguf_encoder(path)
        if details is None:
            raise ValueError("This GGUF is not a supported complete native text encoder.")
        entry = {**entry, **details}
        if details["_projector_path"]:
            projector = Path(details["_projector_path"])
        signature = (signature, file_signature(projector) if projector else None)
    load_device = mm.text_encoder_device() if device == "auto" else resolve_requested_device(device)
    # ComfyUI's dynamic patcher needs a concrete GPU index, including in cache identity.
    if load_device.type == "cuda" and load_device.index is None:
        load_device = torch.device("cuda", torch.cuda.current_device())
    offload_device = torch.device("cpu") if load_device.type == "cpu" else mm.text_encoder_offload_device()
    # Klein's native chat template is suitable for the same Qwen3 4B weights
    # that ComfyUI otherwise loads as a raw Z-Image conditioning encoder.
    clip_type = comfy.sd.CLIPType.FLUX2 if entry["architecture"] == "QWEN3_4B" else comfy.sd.CLIPType.STABLE_DIFFUSION
    embeddings = folder_paths.get_folder_paths("embeddings")
    key = (signature, str(load_device), str(offload_device), clip_type.name, tuple(embeddings))
    if keep_model_loaded and _cached_encoder is not None and _cached_encoder.cache_key == key:
        return _cached_encoder
    clear_text_encoder_cache()
    if is_gguf:
        from .gguf_text_encoders import load_gguf_clip

        clip = load_gguf_clip(path, projector, clip_type=clip_type, embeddings=embeddings,
                              load_device=load_device, offload_device=offload_device, architecture=entry["architecture"])
    else:
        clip = comfy.sd.load_clip(
            ckpt_paths=[str(path)], embedding_directory=embeddings, clip_type=clip_type,
            model_options=generation_options(entry["architecture"], {"load_device": load_device, "offload_device": offload_device}),
        )
    instance = ComfyTextEncoder(clip, entry["architecture"], entry["has_vision"], key)
    try:
        if device != "auto" and clip.patcher.load_device != load_device:
            raise RuntimeError("ComfyUI cannot run this text encoder on the requested device.")
        if not callable(getattr(clip.cond_stage_model, "generate", None)):
            raise TypeError("This text encoder does not support ComfyUI text generation.")
        if text_encoder_signature(path) != key[0]:
            raise RuntimeError("Text encoder or its projector changed while loading; run the node again.")
    except BaseException:
        try:
            release_text_encoder(instance)
        except Exception as cleanup_error:  # noqa: BLE001 - preserve the load failure
            log.warning("ComfyUI Text Encoder", f"Cleanup failed: {type(cleanup_error).__name__}")
        raise
    if keep_model_loaded:
        _cached_encoder = instance
    return instance


def _content_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content
                         if isinstance(item, dict) and item.get("type") == "text")
    return str(content or "")


def _prepare_prompt(prompt, task_name, llm_mode, system_prompt, use_few_shot):
    mode = llm_mode or task_name.lower().replace(" ", "_")
    config = get_llm_few_shot_examples().get(mode, {}) if use_few_shot or llm_mode else {}
    system = system_prompt if system_prompt is not None else get_system_prompt(task_name)
    template = config.get("instruction_template", "") if llm_mode != "direct_chat" else ""
    if isinstance(template, list):
        template = "\n".join(str(item) for item in template)
    if template and llm_mode:
        prompt = template.replace("{prompt}", prompt) if "{prompt}" in template else f"{template} {prompt}"
    # Keep native multimodal chat templates in charge of image placeholders.
    # Demonstrations are text in the system instructions because CLIP.tokenize
    # accepts one prompt, not a Transformers/OpenAI message list.
    if use_few_shot:
        examples = [f"{item.get('role', 'user').capitalize()}: {_content_text(item.get('content'))}"
                    for item in config.get("examples", []) if isinstance(item, dict)]
        if examples:
            system = (system or "") + "\n\nExample conversations:\n" + "\n\n".join(examples)
    return prompt, system or ""


def generate_comfyui(
    instance: ComfyTextEncoder, *, prompt: str, task_name: str, llm_mode: str | None,
    system_prompt: str | None, use_few_shot: bool, image, max_tokens: int,
    do_sample: bool, temperature: float, top_k: int, top_p: float, min_p: float,
    repetition_penalty: float, seed: int, stop_sequences=None,
) -> tuple[str, str]:
    if instance.clip is None:
        raise RuntimeError("The native text encoder has been unloaded; run the node again.")
    if image is not None and not instance.has_vision:
        raise ValueError("This text encoder does not contain vision weights.")
    prompt, system = _prepare_prompt(prompt, task_name, llm_mode, system_prompt, use_few_shot)
    if image is not None and image.ndim == 3:
        image = image.unsqueeze(0)
    options = {"image": image, "skip_template": False, "min_length": 1, "thinking": False, "system_prompt": system}
    architecture = instance.architecture
    if architecture in {"QWEN3_06B", "QWEN25_7B"}:
        # These older native tokenizers lack system_prompt support (Anima is
        # entirely raw). Use their model's chat tokens without a second wrapper.
        media = "<|vision_start|><|image_pad|><|vision_end|>" * (len(image) if image is not None else 0)
        prefix = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        prompt = f"{prefix}<|im_start|>user\n{media}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        if architecture == "QWEN3_06B":
            prompt += "<think>\n\n</think>\n\n"
        if architecture == "QWEN25_7B" and image is not None:
            options["images"] = [image[i:i + 1] for i in range(len(image))]
    elif architecture == "GEMMA_2_2B":
        content = f"{system}\n\n{prompt}" if system else prompt
        prompt = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"

    clip = instance.clip
    with torch.inference_mode():
        tokens = clip.tokenize(prompt, **options)
        generated = clip.generate(
            tokens, do_sample=bool(do_sample and temperature > 0), max_length=max_tokens,
            temperature=temperature if temperature > 0 else 1.0, top_k=top_k,
            top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed,
        )
        raw = clip.decode(generated)
    # ComfyUI does not accept arbitrary string stops. Apply Nova's output stop
    # contract after decoding; native EOS stopping still runs inside generate().
    positions = [raw.find(stop) for stop in stop_sequences or [] if stop and stop in raw]
    if positions:
        raw = raw[:min(positions)]
    return clean_model_output(raw)
