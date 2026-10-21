# RvLoader_SmartLMLoader - Smart LM Loader
#
# Registry-based model loader — no templates, no model_source, no manual path resolution.
# The model registry (core/model_registry.py) provides the unified model list with backend
# suffixes, and this node dispatches to the correct loader and generator based on registry data.
#
# Replaces: Smart Language Model Loader v3 (template-based workflow)
# Node ID: "Smart LM Loader [SmartLLM Nova]"

import gc
import os
import re
import time
import uuid

import torch  # type: ignore
from comfy_api.latest import io  # type: ignore

from ..core import CATEGORY
from ..core.image_helpers import unwrap_value as _unwrap_scalar
from ..core.logger import log
from ..core.sml.config_templates import TemplateContext, with_training_profile
from ..core.sml.lifecycle import (
    register_execution_cleanup,
    with_execution_cleanup,
)
from ..core.sml.loader_base import load_model_with_backend
from ..core.sml.model_acquisition import (
    BACKEND_TO_METHOD as _BACKEND_TO_METHOD,
)
from ..core.sml.model_acquisition import (
    FAMILY_TO_EXEC as _FAMILY_TO_EXEC,
)
from ..core.sml.model_acquisition import (
    GGUF_BACKENDS as _GGUF_BACKENDS,
)
from ..core.sml.model_acquisition import (
    acquire_registered_model as _ensure_downloaded,
)
from ..core.sml.model_acquisition import (
    download_registered_model,
)
from ..core.sml.model_acquisition import (
    resolve_registered_mmproj_path as _resolve_mmproj_path,
)
from ..core.sml.model_acquisition import (
    resolve_registered_model_path as _resolve_model_path,
)
from ..core.sml.model_registry import (
    get_model_entry,
    get_model_list,
    is_model_separator,
    load_defaults,
    save_defaults,
)
from ..core.sml.music3 import TASK_NAME as MUSIC3_TASK
from ..core.sml.music3 import resolve_source as resolve_music3_source
from ..core.sml.panel_configuration import SmartLLMPanel
from ..core.sml.song import active_source, generate_song
from ..core.sml.tasks import (
    TASK_BY_NAME,
    canonicalize_task_name,
    get_h3_mode,
    get_h3_story_system_prompt,
    get_system_prompt,
    get_task_names,
    is_h3_scene_task,
    push_system_prompt_override,
    reset_system_prompt_override,
    resolve_h3_input_mode,
)
from ..core.sml.yue2 import TASK_NAME as YUE2_TASK
from ..core.sml.yue2 import resolve_source as resolve_yue2_source

_LOG_PREFIX = "SmartLLM Loader"

# Transformers family → generate_transformers model_family param
_FAMILY_TF_MAP = {
    "Qwen": "QwenVL",
    "Mistral": "Mistral3",
    "LLM (Text-Only)": "LLM",
    "LLaVA": "LLaVA",
    "VLM": "LLaVA",
}

# Docker backends (need temp image files for vision)
_DOCKER_BACKENDS = {"vllm", "sglang", "ollama", "llamacpp"}

# Tasks that should never pass images (text processing only)
_TEXT_ONLY_TASKS = {
    name for name, task in TASK_BY_NAME.items() if task.category == "text"
}

# Tasks that use images when connected, but also work text-only
_FLEXIBLE_TASKS = {
    "Direct Chat",
    "Custom Instruction",
    "Question Answering",
    "Wan 2.2 Scene 5s",
    "Wan 2.2 Timeline 5s",
    "Wan 2.2 Timeline 5s 2s",
    "Wan 2.2 Timeline 5s 3s",
    "Wan 2.2 Scene 10s",
    "Wan 2.2 Timeline 10s",
    "Wan 2.2 Scene 20s",
    "Wan 2.2 Timeline 20s",
    "Wan 2.2 CN Atomic",
    "LTX 2.3 I2V",
}

_H3_REQUIRED_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_H3_FIELD_PATTERN = re.compile(
    r"(?im)^[ \t]*(integrated_multimodal_description|overall_soundscape|"
    r"non_diegetic_music):[ \t]*"
)


def _should_use_h3_story_mode(task_name, user_prompt, system_prompt):
    return (
        is_h3_scene_task(task_name)
        and not str(user_prompt or "").strip()
        and not str(system_prompt or "").strip()
    )


# ============================================================================
# Image Utilities
# ============================================================================


def _flatten_image_frames(input_image):
    # Preserve the encounter order of individual ComfyUI IMAGE frames.
    frames = []

    def _collect(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                _collect(child)
        elif isinstance(item, torch.Tensor):
            if item.dim() == 3:
                frames.append(item)
            elif item.dim() == 4:
                frames.extend(item[index] for index in range(item.shape[0]))
            else:
                raise ValueError(
                    "MiniMax H3 reference images must be 3D images or 4D batches."
                )
        elif item is not None:
            raise ValueError("MiniMax H3 reference images must be ComfyUI IMAGE tensors.")

    _collect(input_image)
    return frames


def _prepare_h3_reference_group(task_name, images):
    # Treat all H3 image inputs as one ordered keyframe group, not mapped runs.
    mode = get_h3_mode(task_name)
    if mode is None:
        return None, 0, None

    frames = _flatten_image_frames(images)
    image_count = len(frames)
    resolved_mode = resolve_h3_input_mode(task_name, image_count)
    if not frames:
        return None, image_count, resolved_mode

    try:
        grouped = torch.stack(frames, dim=0)
    except RuntimeError as error:
        raise ValueError(
            "MiniMax H3 reference images must have matching dimensions so their "
            "first-to-last order can be preserved."
        ) from error
    return grouped, image_count, resolved_mode


def _image_frame_count(input_image) -> int:
    if input_image is None:
        return 0
    if isinstance(input_image, (list, tuple)):
        return sum(_image_frame_count(item) for item in input_image)
    if isinstance(input_image, torch.Tensor):
        if input_image.dim() == 3:
            return 1
        if input_image.dim() == 4:
            return int(input_image.shape[0])
    return 1


def _generation_frame_count(task_name: str, requested: int, image_count: int) -> int:
    # frame_count is a video sampling control, never permission to drop an H3
    # endpoint. Non-H3 behavior remains unchanged.
    if get_h3_mode(task_name) is not None:
        return max(requested, image_count)
    return requested


def _align_execution_inputs(user_prompts, flat_images):
    num_runs = max(len(user_prompts), len(flat_images))
    prompts_aligned = [
        user_prompts[index if index < len(user_prompts) else -1]
        for index in range(num_runs)
    ]
    if not flat_images:
        images_aligned = [None] * num_runs
    elif len(flat_images) == 1:
        images_aligned = [flat_images[0]] * num_runs
    else:
        images_aligned = [
            flat_images[index if index < len(flat_images) else -1]
            for index in range(num_runs)
        ]
    return prompts_aligned, images_aligned


def _get_temp_image_path(suffix: str = ".jpg") -> str:
    # Temp file in ComfyUI/temp (cleaned on restart).
    import folder_paths  # type: ignore

    temp_dir = folder_paths.get_temp_directory()
    return os.path.join(temp_dir, f"smartllm_temp_{uuid.uuid4().hex}{suffix}")


def _tensor_to_temp_jpegs(input_image, max_pixels: int = 0, frame_count: int = 0):
    # Convert ComfyUI image tensor [B,H,W,C] or list of tensors to temp JPEG files.
    # When frame_count > 0 and the batch exceeds it, keep the LAST frame_count
    # frames (preserves the most recent context for chained / video workflows).
    # Returns (image_paths, original_size, resized_size).
    from ..core.sml.vlm_detection import smart_resize_for_vlm, tensor_to_pil

    image_paths = []
    original_size = None
    resized_size = None

    def _process(frame):
        nonlocal original_size, resized_size
        img = tensor_to_pil(frame)
        if img is None:
            raise ValueError("Failed to convert image tensor to PIL Image")
        if original_size is None:
            original_size = (img.width, img.height)
        if max_pixels > 0:
            img, _ = smart_resize_for_vlm(img, max_pixels=max_pixels)
        if resized_size is None:
            resized_size = (img.width, img.height)
        path = _get_temp_image_path()
        img.save(path, "JPEG", quality=95)
        image_paths.append(path)

    # Flatten any combination of list, tuple, or batched tensors to a list of individual 3D tensors
    flat_frames = []
    if isinstance(input_image, (list, tuple)):
        for item in input_image:
            if isinstance(item, torch.Tensor):
                if item.dim() == 3:
                    flat_frames.append(item)
                elif item.dim() == 4:
                    for j in range(item.shape[0]):
                        flat_frames.append(item[j])
    elif isinstance(input_image, torch.Tensor):
        if input_image.dim() == 3:
            flat_frames.append(input_image)
        elif input_image.dim() == 4:
            for j in range(input_image.shape[0]):
                flat_frames.append(input_image[j])

    if flat_frames:
        total = len(flat_frames)
        start = max(0, total - frame_count) if frame_count and frame_count > 0 else 0
        for i in range(start, total):
            _process(flat_frames[i])

    return (
        image_paths,
        original_size or (0, 0),
        resized_size or original_size or (0, 0),
    )


def _cleanup_temp_files(paths):
    if paths:
        for p in paths:
            try:
                os.remove(p)
            except Exception:
                pass


# ============================================================================
# Prompt Building
# ============================================================================


def _build_vlm_prompt(task_name, user_prompt, input_image, *, family="Qwen"):
    # Build VLM prompt as a (system, user, is_text_only) triple.
    # Backends now receive system + user separately — no more "\n\nAdditional context:"
    # marker hack and no parser ambiguity when the system text contains blank lines.
    has_text = bool(user_prompt and user_prompt.strip())
    h3_mode = get_h3_mode(task_name)
    image_count = _image_frame_count(input_image)
    if h3_mode is not None:
        resolve_h3_input_mode(task_name, image_count)
    has_image = image_count > 0
    is_flexible = task_name in _FLEXIBLE_TASKS or h3_mode is not None
    is_text_only = task_name in _TEXT_ONLY_TASKS or (
        is_flexible and has_text and not has_image
    )

    if is_text_only:
        # Backend supplies the system prompt via llm_mode (+ few-shot)
        return None, user_prompt, True

    if is_flexible and has_image and has_text:
        # Direct Chat / Custom / QA with image+text.
        # Prefer the task's system prompt (JSON entry, or wired override via ContextVar)
        # so user_prompt flows as the actual user message and few-shot training applies.
        # Only when neither override nor JSON entry exists fall back to the legacy
        # behavior: user_prompt drives the system slot (preserves prior workflows
        # for tasks like Direct Chat that have no JSON system prompt defined).
        base = get_system_prompt(task_name)
        if base:
            return base, user_prompt.strip(), False
        return user_prompt.strip(), "", False

    if has_text:
        base = get_system_prompt(task_name)
        if family in ("LLaVA", "VLM") or not base:
            return None, user_prompt, False
        return base, user_prompt, False

    # Image only, no user text
    base = get_system_prompt(task_name) or task_name or "Describe this image in detail."
    return base, "", False


def _h3_output_contract_issue(task_name, output, image_count, *, story_mode=False):
    # Return a content-safe diagnostic when an H3 response is incomplete.
    mode = get_h3_mode(task_name)
    if mode is None:
        return None

    text = str(output or "").strip()
    if not text:
        return "empty response"

    matches = list(_H3_FIELD_PATTERN.finditer(text))
    labels = tuple(match.group(1).lower() for match in matches)
    missing = [label for label in _H3_REQUIRED_FIELDS if label not in labels]
    if missing:
        return f"missing required field(s): {', '.join(missing)}"
    if labels != _H3_REQUIRED_FIELDS:
        return "required fields are duplicated or out of order"

    bodies = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        bodies.append(text[match.end() : end].strip())
    empty = [
        label
        for label, body in zip(_H3_REQUIRED_FIELDS, bodies, strict=True)
        if not body
    ]
    if empty:
        return f"empty required field(s): {', '.join(empty)}"
    if not bodies[0].startswith("[Shot 1]"):
        return "integrated_multimodal_description must begin with [Shot 1]"

    mode.resolve_input_mode(image_count)
    first_line = text.splitlines()[0].strip()
    if not first_line.lower().startswith("integrated_multimodal_description:"):
        return "H3 output must begin with integrated_multimodal_description"
    if story_mode:
        music = bodies[2].strip().casefold().rstrip(".! ")
        if music in {
            "n/a",
            "na",
            "none",
            "no music",
            "no background music",
            "silence",
        }:
            return "non_diegetic_music must describe background music in story mode"
    return None


def _h3_recovery_system_prompt(task_name, image_count):
    get_h3_mode(task_name).resolve_input_mode(image_count)
    opening = "Begin immediately with integrated_multimodal_description: [Shot 1]"

    base = get_system_prompt(task_name).rstrip()
    recovery = (
        "FORMAT RECOVERY (highest priority): Generate the complete response again "
        "from the beginning. "
        f"{opening}. Finish the full visible action or shot timeline before writing "
        "overall_soundscape, then finish non_diegetic_music. Return all three fields "
        "exactly once and in that order. Never begin with overall_soundscape or "
        "non_diegetic_music, and never return audio fields without the complete "
        "integrated_multimodal_description."
    )
    return f"{base}\n\n{recovery}" if base else recovery


def _h3_field_bodies(output):
    text = str(output or "").strip()
    matches = list(_H3_FIELD_PATTERN.finditer(text))
    bodies = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = match.group(1).lower()
        body = text[match.end() : end].strip()
        if body and label not in bodies:
            bodies[label] = body
    return bodies


def _normalize_h3_output(task_name, output, image_count):
    # Reference-image placement is transport metadata, not part of the motion
    # prompt. Drop any leading preamble from older/custom prompts or model output.
    mode = get_h3_mode(task_name)
    text = str(output or "").strip()
    if mode is None:
        return text
    mode.resolve_input_mode(image_count)
    first_field = _H3_FIELD_PATTERN.search(text)
    if (
        first_field is not None
        and first_field.group(1).lower() == "integrated_multimodal_description"
    ):
        return text[first_field.start() :].lstrip()
    return text


def _h3_fallback_output(
    task_name,
    user_prompt,
    image_count,
    *model_outputs,
    story_mode=False,
):
    # Build a valid, conservative prompt when the model omits required fields
    # twice. Prefer any usable generated field, but always retain the requested
    # action instead of failing the whole ComfyUI execution.
    mode = get_h3_mode(task_name)
    mode.resolve_input_mode(image_count)
    candidates = [_h3_field_bodies(output) for output in model_outputs]

    def _first_body(label):
        return next(
            (fields[label] for fields in candidates if fields.get(label)),
            None,
        )

    integrated = _first_body("integrated_multimodal_description")
    if integrated:
        if not integrated.startswith("[Shot 1]"):
            integrated = f"[Shot 1] {integrated}"
    else:
        action = re.sub(r"\s+", " ", str(user_prompt or "")).strip()
        if not action:
            if story_mode and image_count == 0:
                action = (
                    "An original traveler enters a rain-swept station, discovers a "
                    "forgotten letter beneath a bench, and looks toward the departing "
                    "train as its final carriage disappears into the night."
                )
            elif story_mode and image_count == 2:
                action = (
                    "The depicted subjects and environment move through a natural "
                    "story progression from the first composition into the final "
                    "composition."
                )
            elif image_count:
                action = (
                    "The visible subject maintains the depicted scene while breathing, "
                    "blinking, and making a subtle natural posture shift."
                )
            else:
                action = "The requested scene unfolds as one continuous visible action."
        integrated = f"[Shot 1] {action}"

    soundscape = _first_body("overall_soundscape")
    if story_mode and (
        not soundscape or soundscape.strip().casefold() in {"n/a", "na"}
    ):
        soundscape = "Scene-appropriate ambience and synchronized action sounds."
    music = _first_body("non_diegetic_music")
    if story_mode and (
        not music
        or music.strip().casefold().rstrip(".! ")
        in {"n/a", "na", "none", "no music", "no background music", "silence"}
    ):
        music = "An original cinematic score shaped to the scene's mood and pacing."

    fields = [
        f"integrated_multimodal_description: {integrated}",
        f"overall_soundscape: {soundscape or 'N/A'}",
        f"non_diegetic_music: {music or 'N/A'}",
    ]
    return "\n\n".join(fields)


# ============================================================================
# Backend Generation Dispatch
# ============================================================================


def _dispatch_generate(
    instance,
    *,
    prompt,
    input_image=None,
    is_text_only_task=False,
    max_tokens,
    temperature,
    top_p,
    top_k,
    seed,
    repetition_penalty=1.0,
    num_beams=1,
    do_sample=True,
    model_family="",
    task_name="",
    context_size=8192,
    frame_count=1,
    llm_mode=None,
    use_few_shot=True,
    min_p=0.0,
    mirostat=0,
    mirostat_eta=0.1,
    mirostat_tau=5.0,
    repeat_last_n=64,
    stop_sequences=None,
    system_prompt=None,
):
    # Route generation to the correct backend.
    # Returns (result, raw_output, data, original_size, resized_size).
    music_source = active_source.get()
    if task_name in (MUSIC3_TASK, YUE2_TASK) and music_source is not None:
        llm_mode = music_source.training_key
    is_vision = input_image is not None and not is_text_only_task
    vision_task = task_name if is_vision and not llm_mode else None
    image_paths = None
    original_size = resized_size = None
    raw_output = None
    data = {}

    try:
        if getattr(instance, "is_comfyui", False):
            from ..core.sml.backend_comfyui import generate_comfyui

            effective_image = input_image if is_vision else None
            if effective_image is not None and effective_image.ndim == 4 and frame_count > 0:
                effective_image = effective_image[-frame_count:]
            result, raw_output = generate_comfyui(
                instance, prompt=prompt, image=effective_image, task_name=task_name,
                llm_mode=llm_mode, system_prompt=system_prompt, use_few_shot=use_few_shot,
                max_tokens=max_tokens, do_sample=do_sample, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repetition_penalty=repetition_penalty, seed=seed,
                stop_sequences=stop_sequences,
            )
        # ── Docker / OpenAI-compatible backends ──
        elif hasattr(instance, "is_vllm") and instance.is_vllm:
            native = hasattr(instance, "is_vllm_native") and instance.is_vllm_native
            if native:
                from ..core.sml.backend_vllm_native import generate_vllm
            else:
                from ..core.sml.backend_vllm_docker import generate_vllm
            if is_vision:
                from ..core.sml.vlm_detection import get_max_pixels_for_model_type

                image_paths, original_size, resized_size = _tensor_to_temp_jpegs(
                    input_image,
                    max_pixels=get_max_pixels_for_model_type(
                        getattr(instance, "model_type", None)
                    ),
                    frame_count=frame_count,
                )
            kw = dict(
                smart_lm_instance=instance,
                prompt=prompt,
                image_paths=image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                use_few_shot=use_few_shot,
                min_p=min_p,
                stop_sequences=stop_sequences,
                system_prompt=system_prompt,
            )
            if vision_task:
                kw["vision_task"] = vision_task
            if llm_mode:
                kw["llm_mode"] = llm_mode
            gen = generate_vllm(**kw)
            result, raw_output = gen if isinstance(gen, tuple) else (gen, None)

        elif hasattr(instance, "is_sglang") and instance.is_sglang:
            from ..core.sml.backend_sglang_docker import generate_sglang

            if is_vision:
                from ..core.sml.vlm_detection import get_max_pixels_for_model_type

                image_paths, original_size, resized_size = _tensor_to_temp_jpegs(
                    input_image,
                    max_pixels=get_max_pixels_for_model_type(
                        getattr(instance, "model_type", None)
                    ),
                    frame_count=frame_count,
                )
            kw = dict(
                smart_lm_instance=instance,
                prompt=prompt,
                image_paths=image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                use_few_shot=use_few_shot,
                min_p=min_p,
                stop_sequences=stop_sequences,
                system_prompt=system_prompt,
            )
            if vision_task:
                kw["vision_task"] = vision_task
            if llm_mode:
                kw["llm_mode"] = llm_mode
            gen = generate_sglang(**kw)
            result, raw_output = gen if isinstance(gen, tuple) else (gen, None)

        elif hasattr(instance, "is_ollama") and instance.is_ollama:
            from ..core.sml.backend_ollama_docker import generate_ollama

            if is_vision:
                from ..core.sml.vlm_detection import get_max_pixels_for_model_type

                image_paths, original_size, resized_size = _tensor_to_temp_jpegs(
                    input_image,
                    max_pixels=get_max_pixels_for_model_type(
                        getattr(instance, "model_type", None)
                    ),
                    frame_count=frame_count,
                )
            kw = dict(
                smart_lm_instance=instance,
                prompt=prompt,
                image_paths=image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                use_few_shot=use_few_shot,
                min_p=min_p,
                mirostat=mirostat,
                mirostat_eta=mirostat_eta,
                mirostat_tau=mirostat_tau,
                repeat_last_n=repeat_last_n,
                stop_sequences=stop_sequences,
                system_prompt=system_prompt,
            )
            if vision_task:
                kw["vision_task"] = vision_task
            if llm_mode:
                kw["llm_mode"] = llm_mode
            result, raw_output = generate_ollama(**kw)

        elif hasattr(instance, "is_llamacpp_docker") and instance.is_llamacpp_docker:
            from ..core.sml.backend_llamacpp_docker import generate_llamacpp

            if is_vision:
                from ..core.sml.vlm_detection import get_max_pixels_for_model_type

                image_paths, original_size, resized_size = _tensor_to_temp_jpegs(
                    input_image,
                    max_pixels=get_max_pixels_for_model_type(
                        getattr(instance, "model_type", None)
                    ),
                    frame_count=frame_count,
                )
            kw = dict(
                smart_lm_instance=instance,
                prompt=prompt,
                image_paths=image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                use_few_shot=use_few_shot,
                min_p=min_p,
                mirostat=mirostat,
                mirostat_eta=mirostat_eta,
                mirostat_tau=mirostat_tau,
                repeat_last_n=repeat_last_n,
                stop_sequences=stop_sequences,
                system_prompt=system_prompt,
            )
            if vision_task:
                kw["vision_task"] = vision_task
            if llm_mode:
                kw["llm_mode"] = llm_mode
            result, raw_output = generate_llamacpp(**kw)

        # ── Local backends ──
        elif instance.is_gguf:
            from ..core.sml.backend_gguf import generate_gguf

            effective_image = input_image if is_vision else None
            is_text_family = model_family in ("LLM (Text-Only)",)
            kw = dict(
                smart_lm_instance=instance,
                model_type="text" if is_text_family else "vision",
                image=effective_image,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                use_few_shot=use_few_shot,
                min_p=min_p,
                mirostat=mirostat,
                mirostat_eta=mirostat_eta,
                mirostat_tau=mirostat_tau,
                repeat_last_n=repeat_last_n,
                stop_sequences=stop_sequences,
                system_prompt=system_prompt,
            )
            if not is_text_family:
                kw["frame_count"] = frame_count
            if vision_task:
                kw["vision_task"] = vision_task
            if llm_mode:
                kw["llm_mode"] = llm_mode
            result = generate_gguf(**kw)

        else:
            # Transformers
            from ..core.sml.backend_transformers import generate_transformers

            effective_image = input_image if is_vision else None
            tf_family = _FAMILY_TF_MAP.get(model_family, "LLM")
            kw = dict(
                smart_lm_instance=instance,
                model_family=tf_family,
                image=effective_image,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                context_size=context_size,
                use_few_shot=use_few_shot,
                system_prompt=system_prompt,
            )
            if is_vision:
                kw["num_beams"] = num_beams
                kw["do_sample"] = do_sample
                kw["frame_count"] = frame_count
            if vision_task:
                kw["vision_task"] = vision_task
            if llm_mode:
                kw["llm_mode"] = llm_mode
                kw["instruction_template"] = ""
            result, data = generate_transformers(**kw)
            raw_output = data.get("raw_output", result) if data else result
    finally:
        _cleanup_temp_files(image_paths)

    return result, raw_output, data, original_size, resized_size


# ============================================================================
# Per-Family Generation Router
# ============================================================================


def _generate_for_family(
    *,
    model_family,
    instance,
    task_name,
    user_prompt,
    input_image,
    max_tokens,
    temperature,
    top_p,
    top_k,
    num_beams,
    do_sample,
    seed,
    repetition_penalty,
    context_size,
    frame_count,
    use_few_shot=True,
    min_p=0.0,
    mirostat=0,
    mirostat_eta=0.1,
    mirostat_tau=5.0,
    repeat_last_n=64,
    stop_sequences=None,
):
    # Dispatch to correct generation path based on model family.
    # Returns (result, data).
    result = ""
    data = {}

    if model_family == "Florence":
        from ..core.sml.backend_transformers import generate_transformers

        # Florence uses florence_id as prompt token
        task_obj = TASK_BY_NAME.get(task_name)
        if not task_obj or not task_obj.florence_id:
            raise ValueError(
                f"Task '{task_name}' is not supported by Florence-2 (no florence_id mapping)"
            )
        florence_prompt = task_obj.florence_id

        # Florence text input: use user_prompt for detection-like tasks
        florence_text = user_prompt or ""

        if (
            input_image is not None
            and input_image.dim() == 4
            and input_image.shape[0] > 1
        ):
            log.warning(
                "Florence-2",
                f"Video not supported ({input_image.shape[0]} frames), using first frame only",
                notify=True,
            )

        result, data = generate_transformers(
            smart_lm_instance=instance,
            model_family="Florence2",
            image=input_image,
            prompt=florence_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            seed=seed,
            repetition_penalty=repetition_penalty,
            text_input=florence_text,
            context_size=context_size,
            convert_to_bboxes=False,
        )

    elif model_family in ("Qwen", "Mistral", "LLaVA", "VLM"):
        sys_prompt, user_msg, is_text_only = _build_vlm_prompt(
            task_name, user_prompt, input_image, family=model_family
        )

        # For text-only tasks, pass llm_mode for proper few-shot + system prompt handling
        result, raw, data, _, _ = _dispatch_generate(
            instance,
            prompt=user_msg,
            input_image=input_image,
            is_text_only_task=is_text_only,
            system_prompt=sys_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            do_sample=do_sample,
            model_family=model_family,
            task_name=task_name,
            context_size=context_size,
            frame_count=frame_count,
            llm_mode=task_name.lower().replace(" ", "_") if is_text_only else None,
            use_few_shot=use_few_shot,
            min_p=min_p,
            mirostat=mirostat,
            mirostat_eta=mirostat_eta,
            mirostat_tau=mirostat_tau,
            repeat_last_n=repeat_last_n,
            stop_sequences=stop_sequences,
        )

    elif model_family == "LLM (Text-Only)":
        text_content = user_prompt or ""
        # Allow empty user_prompt when a system prompt is available (wired override
        # via ContextVar OR JSON-defined task system prompt) — the model will
        # respond to the system instruction alone. Required for Direct Chat with
        # connected system_prompt and no user input.
        if not text_content.strip() and not get_system_prompt(task_name):
            raise ValueError(
                "LLM requires a prompt. Wire a string into 'user_prompt' or type one."
            )

        # Replace underscores for tag conversion tasks
        if task_name in ("Tags to Natural Language", "Natural Language to Tags"):
            text_content = text_content.replace("_", " ")

        llm_mode = task_name.lower().replace(" ", "_")
        # Don't prepend system — backend handles it via llm_mode
        prompt = text_content

        result, raw, data, _, _ = _dispatch_generate(
            instance,
            prompt=prompt,
            input_image=None,
            is_text_only_task=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            repetition_penalty=repetition_penalty,
            model_family="LLM (Text-Only)",
            do_sample=do_sample,
            task_name=task_name,
            context_size=context_size,
            llm_mode=llm_mode,
            use_few_shot=use_few_shot,
            min_p=min_p,
            mirostat=mirostat,
            mirostat_eta=mirostat_eta,
            mirostat_tau=mirostat_tau,
            repeat_last_n=repeat_last_n,
            stop_sequences=stop_sequences,
        )
        if raw is not None and raw != result:
            data = {"raw_output": raw}

    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return result, data


def _generate_with_task_contract(task_name, generate, user_prompt="", previous_task=None):
    """Apply each song contract to first and chained generations."""
    task_name = canonicalize_task_name(task_name)
    resolver = {MUSIC3_TASK: resolve_music3_source, YUE2_TASK: resolve_yue2_source}.get(task_name)
    if resolver is None:
        return generate()
    source = resolver(user_prompt or "", previous_task)
    return generate_song(task_name, source, generate, log)


def _generate_with_h3_recovery(**generation_kwargs):
    # H3 prompts have a machine-readable field contract. Retry one malformed
    # response with a stronger system instruction instead of silently returning
    # an audio-only or otherwise incomplete prompt to the downstream video model.
    story_mode = bool(generation_kwargs.pop("h3_story_mode", False))
    task_name = canonicalize_task_name(generation_kwargs["task_name"])
    generation_kwargs["task_name"] = task_name
    mode = get_h3_mode(task_name)
    if mode is None:
        return _generate_with_task_contract(
            task_name, lambda: _generate_for_family(**generation_kwargs),
            user_prompt=generation_kwargs.get("user_prompt"),
        )

    story_token = None
    if story_mode:
        story_token = push_system_prompt_override(get_h3_story_system_prompt(task_name))
    try:
        image_count = _image_frame_count(generation_kwargs.get("input_image"))
        initial_result, data = _generate_for_family(**generation_kwargs)
        initial_result = _normalize_h3_output(task_name, initial_result, image_count)
        issue = _h3_output_contract_issue(
            task_name, initial_result, image_count, story_mode=story_mode
        )
        if issue is None:
            return initial_result, data

        log.warning(
            _LOG_PREFIX,
            f"Incomplete MiniMax H3 output ({issue}); retrying once",
        )
        retry_kwargs = dict(generation_kwargs)

        recovery_prompt = _h3_recovery_system_prompt(task_name, image_count)
        recovery_token = push_system_prompt_override(recovery_prompt)
        try:
            result, data = _generate_for_family(**retry_kwargs)
            result = _normalize_h3_output(task_name, result, image_count)
        finally:
            reset_system_prompt_override(recovery_token)

        issue = _h3_output_contract_issue(
            task_name, result, image_count, story_mode=story_mode
        )
        if issue is None:
            return result, data

        log.warning(
            _LOG_PREFIX,
            f"MiniMax H3 corrective retry remained incomplete ({issue}); "
            "returning a structured fallback",
            notify=True,
        )
        result = _h3_fallback_output(
            task_name,
            generation_kwargs.get("user_prompt"),
            image_count,
            result,
            initial_result,
            story_mode=story_mode,
        )
        if isinstance(data, dict):
            data = dict(data)
            data["h3_format_fallback"] = True
        else:
            data = {"h3_format_fallback": True}
        return result, data
    finally:
        if story_token is not None:
            reset_system_prompt_override(story_token)


# ============================================================================
# Multi-Task Chaining
# ============================================================================


def _run_multi_task_chain(
    *,
    tasks_to_run,
    first_result,
    first_data,
    instance,
    model_family,
    max_tokens,
    temperature,
    top_p,
    top_k,
    num_beams,
    do_sample,
    seed,
    repetition_penalty,
    context_size,
    frame_count,
    use_few_shot=True,
    min_p=0.0,
    mirostat=0,
    mirostat_eta=0.1,
    mirostat_tau=5.0,
    repeat_last_n=64,
    stop_sequences=None,
):
    # Run tasks 2..N sequentially, chaining output → input.
    # Returns (final_result, final_data).

    # Clear GGUF state after first task
    if hasattr(instance, "is_gguf") and instance.is_gguf:
        from ..core.sml.backend_gguf import clear_gguf_state_between_tasks

        clear_gguf_state_between_tasks(instance)

    all_results = [
        {
            "step": 1,
            "task": tasks_to_run[0],
            "result": first_result,
            "data": first_data or None,
        }
    ]
    current_text = first_result

    for idx in range(1, len(tasks_to_run)):
        task_name = canonicalize_task_name(tasks_to_run[idx])
        log.info(
            _LOG_PREFIX, f"Multi-task step {idx + 1}/{len(tasks_to_run)}: {task_name}"
        )

        if hasattr(instance, "is_gguf") and instance.is_gguf:
            from ..core.sml.backend_gguf import clear_gguf_state_between_tasks

            clear_gguf_state_between_tasks(instance)

        # Song tasks must report an unusable Song Lyrics source, including an empty
        # one, through its parser instead of silently ending the requested chain.
        requires_song_source = (
            task_name in (MUSIC3_TASK, YUE2_TASK)
            and canonicalize_task_name(tasks_to_run[idx - 1]) == "Song Lyrics"
        )
        if (not current_text or not current_text.strip()) and not requires_song_source:
            log.warning(_LOG_PREFIX, f"Task {idx} returned empty, stopping chain", notify=True)
            break

        chained_llm_mode = task_name.lower().replace(" ", "_")

        # Don't prepend system — backend handles system + few-shot via llm_mode
        prompt = current_text

        dispatch_kwargs = {
            "prompt": prompt,
            "input_image": None,
            "is_text_only_task": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "model_family": model_family,
            "task_name": task_name,
            "context_size": context_size,
            "frame_count": frame_count,
            "llm_mode": chained_llm_mode,
            "use_few_shot": use_few_shot,
            "min_p": min_p,
            "mirostat": mirostat,
            "mirostat_eta": mirostat_eta,
            "mirostat_tau": mirostat_tau,
            "repeat_last_n": repeat_last_n,
            "stop_sequences": stop_sequences,
        }

        def generate_stage(dispatch_kwargs=dispatch_kwargs):
            result, _, data, _, _ = _dispatch_generate(instance, **dispatch_kwargs)
            return result, data

        task_result, task_data = _generate_with_task_contract(
            task_name, generate_stage, user_prompt=current_text,
            previous_task=canonicalize_task_name(tasks_to_run[idx - 1]),
        )

        all_results.append(
            {
                "step": idx + 1,
                "task": task_name,
                "result": task_result,
                "data": task_data or None,
            }
        )
        current_text = task_result

    data = {
        "multi_task": True,
        "task_count": len(all_results),
        "tasks": all_results,
        "final_result": current_text,
    }
    log.info(_LOG_PREFIX, f"Multi-task complete: {len(all_results)} tasks")
    return current_text, data


# ============================================================================
# WD14 Tagger Fast-Path
# ============================================================================


def _execute_wd14(
    *,
    repo_id,
    revision,
    expected_sha256,
    device,
    images,
    threshold,
    char_threshold,
    replace_underscore,
    keep_model_loaded,
    node_id=None,
    initial_title=None,
):
    # WD14 tagger — completely separate from LLM pipeline.
    import comfy.utils  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image

    from ..core.sml.backend_wd14 import load_wd14_model, tag_image, unload_wd14_model

    if images is None:
        raise ValueError("WD14 Tagger requires an image input")

    # Model name = last part of repo_id (e.g. "SmilingWolf/wd-swinv2-tagger-v3" → "wd-swinv2-tagger-v3")
    wd14_model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    wd14_entry = {
        "backend": "wd14",
        "name": wd14_model_name,
        "repo_id": repo_id,
        "family": "WD14",
        "has_vision": True,
    }
    if revision is not None:
        wd14_entry["revision"] = revision
    if expected_sha256 is not None:
        wd14_entry["expected_sha256"] = expected_sha256
    download_registered_model(wd14_entry, log_prefix=_LOG_PREFIX)

    if not keep_model_loaded:
        register_execution_cleanup(unload_wd14_model)

    session, tags_data = load_wd14_model(wd14_model_name, device=device)

    # Flatten/normalize images list for processing
    flat_list = []

    def _process(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                _process(sub)
        elif isinstance(item, torch.Tensor):
            if item.dim() == 4:
                for i in range(item.shape[0]):
                    flat_list.append(item[i : i + 1, ...])
            elif item.dim() == 3:
                flat_list.append(item.unsqueeze(0))
        elif item is not None:
            flat_list.append(item)

    _process(images)

    def update_title(sub_status: str):
        if node_id is not None:
            try:
                from server import PromptServer

                PromptServer.instance.send_sync(
                    "smartllm-nova/update_node_title",
                    {"node_id": node_id, "title": sub_status},
                )
            except Exception:
                pass

    def restore_title():
        if node_id is not None:
            try:
                from server import PromptServer

                PromptServer.instance.send_sync(
                    "smartllm-nova/update_node_title",
                    {"node_id": node_id, "title": initial_title},
                )
            except Exception:
                pass

    results = []
    pbar = comfy.utils.ProgressBar(len(flat_list))

    try:
        for batch_number, image in enumerate(flat_list):
            log.msg(
                _LOG_PREFIX,
                f"Tagging image {batch_number + 1}/{len(flat_list)}",
            )
            if node_id is not None:
                update_title(f"Tagging {batch_number + 1}/{len(flat_list)}")

            if hasattr(image, "shape") and image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            tags = tag_image(
                pil_img,
                session,
                tags_data,
                threshold=threshold,
                char_threshold=char_threshold,
                replace_underscore=replace_underscore,
                trailing_comma=False,
            )
            results.append(tags)
            pbar.update(1)
    finally:
        if initial_title and node_id is not None:
            restore_title()

    return io.NodeOutput(flat_list, results)


# ============================================================================
# Model Cleanup
# ============================================================================


def _cleanup_model(*, loading_method, keep_model_loaded, model_path, instance):
    # Handle Docker auto-stop and model VRAM cleanup.

    if loading_method == "ComfyUI":
        if not keep_model_loaded:
            from ..core.sml.backend_comfyui import release_text_encoder

            release_text_encoder(instance)
        return

    # Docker auto-stop: stop the backing container when Keep Loaded is OFF.
    if not keep_model_loaded:
        if loading_method == "vLLM (Docker)":
            from ..core.sml import backend_vllm_docker

            backend_vllm_docker.stop_vllm_container()
        elif loading_method == "SGLang (Docker)" or (
            hasattr(instance, "is_sglang") and instance.is_sglang
        ):
            from ..core.sml import backend_sglang_docker

            backend_sglang_docker.stop_sglang_container()
        elif loading_method == "Ollama (Docker)":
            from ..core.sml import backend_ollama_docker

            backend_ollama_docker.stop_ollama_container()
        elif loading_method == "llama.cpp (Docker)":
            from ..core.sml import backend_llamacpp_docker

            backend_llamacpp_docker.stop_llamacpp_container(
                getattr(instance, "llamacpp_container_name", None)
            )

    if keep_model_loaded:
        return

    is_gguf = loading_method == "GGUF (llama-cpp-python)"
    is_transformers = loading_method.lower() == "transformers"
    is_vllm_native = loading_method == "vLLM (Native)"

    if is_vllm_native:
        from ..core.sml import backend_vllm_native

        backend_vllm_native.unload_vllm(instance, model_path)

    if is_gguf:
        from ..core.sml.backend_gguf import cleanup_chat_handler_vision
        from ..core.sml.model_cache import clear_gguf_cache, is_gguf_cache_empty

        if not is_gguf_cache_empty():
            clear_gguf_cache()
        else:
            actual = instance.model if hasattr(instance, "model") else instance
            if actual is not None:
                for attr in (
                    "_smartllm_chat_handler",
                    "_sml_chat_handler",
                    "_eclipse_chat_handler",
                    "chat_handler",
                ):
                    handler = getattr(actual, attr, None)
                    if handler is not None:
                        cleanup_chat_handler_vision(handler)
                        setattr(actual, attr, None)
                if hasattr(actual, "close") and callable(actual.close):
                    actual.close()
        if hasattr(instance, "model"):
            instance.model = None
        if hasattr(instance, "chat_handler_ref"):
            instance.chat_handler_ref = None

    if is_transformers:
        from ..core.sml.model_cache import (
            clear_transformers_cache,
            is_transformers_cache_empty,
        )

        if not is_transformers_cache_empty():
            clear_transformers_cache()
        else:
            actual = instance.model if hasattr(instance, "model") else instance
            if actual is not None:
                if hasattr(actual, "eval"):
                    actual.eval()
                if hasattr(actual, "zero_grad"):
                    try:
                        actual.zero_grad(set_to_none=True)
                    except Exception:
                        pass
        if hasattr(instance, "model"):
            instance.model = None
        if hasattr(instance, "processor"):
            instance.processor = None

    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================================
# Node Class
# ============================================================================


class RvLoader_SmartModelLoader_LM(SmartLLMPanel, io.ComfyNode):
    PANEL_KIND = "loader"

    @classmethod
    def _engine_schema(cls):
        models = get_model_list()
        first_model = next(
            (m for m in models if not is_model_separator(m)),
            models[0] if models else "",
        )
        defaults = load_defaults()

        # Task list — full superset for schema validation (JS filters at runtime)
        task_names = get_task_names(has_vision=True, include_all_families=True)
        task_names_none = ["None"] + task_names

        # GGUF quantization list from defaults (user-editable, JS refreshes on model change)
        quant_placeholders = defaults.get(
            "quantizations", ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        )

        return io.Schema(
            node_id="Smart LM Loader [SmartLLM Nova]",
            display_name="Smart LM Loader",
            search_aliases=["Smart Model Loader", "SmartLML", "LLM", "VLM"],
            category=CATEGORY.MAIN.value + CATEGORY.LOADER.value,
            description="Registry and native ComfyUI text encoder loader with unified model dropdown, "
            "multi-task chaining, and WD14 tagger support.",
            inputs=[
                # ── Model selection ───────────────────────────────────
                io.Combo.Input(
                    "model",
                    options=models,
                    default=first_model,
                    tooltip="Choose the language, vision, or tagging model to load.\n"
                    "Suffixes indicate the backend engine:\n"
                    "• (no suffix) - Hugging Face Transformers (local, PyTorch)\n"
                    "• -GGUF - native llama-cpp-python engine\n"
                    "• -llama.cpp - llama.cpp Docker server using the selected GGUF file\n"
                    "• -vLLM / -SGLang / -Ollama - other Docker backends\n"
                    "• ComfyUI: - compatible local text encoder files, using ComfyUI's native generation\n"
                    "• -WD14 - WD14 tagger for anime-style image tagging.",
                ),
                io.Combo.Input(
                    "quantization",
                    options=quant_placeholders,
                    default="Q4_K_M",
                    tooltip="Choose GGUF quantization precision. Lower bits (e.g. Q4_K_M) use less VRAM but lose accuracy. "
                    "Higher bits (e.g. Q8_0) are more accurate but demand more VRAM. Applies to native GGUF and llama.cpp Docker models.",
                ),
                # ── Tasks ─────────────────────────────────────────────
                io.Combo.Input(
                    "task",
                    options=task_names,
                    default="Detailed Description",
                    tooltip="The primary task to run. Determines the prompt template, system instructions, and training examples. "
                    "Note: Vision tasks (VLM) require feeding an image into the 'images' input slot.",
                ),
                io.Combo.Input(
                    "task_2",
                    options=task_names_none,
                    default="None",
                    tooltip="Optional second task in a multi-task chain. Runs in sequence using the output of the first task as context.",
                ),
                io.Combo.Input(
                    "task_3",
                    options=task_names_none,
                    default="None",
                    tooltip="Optional third task in a multi-task chain. Runs in sequence using the output of the second task as context.",
                ),
                io.Combo.Input(
                    "task_4",
                    options=task_names_none,
                    default="None",
                    tooltip="Optional fourth task in a multi-task chain. Runs in sequence using the output of the third task as context.",
                ),
                # ── Prompt / context ──────────────────────────────────
                io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    tooltip="Your custom instructions, question, or prompt for the model. "
                    "Type text directly or wire an upstream string. Combined with the selected task prompt.",
                ),
                io.Int.Input(
                    "context_size",
                    default=int(defaults.get("context_size", 8192)),
                    min=512,
                    max=131072,
                    step=512,
                    tooltip="Maximum context window (tokens) allocated for the model. "
                    "Determines how much history, text, and image tokens the model can process at once. Larger sizes use more VRAM.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=int(defaults.get("image_max_tokens", 2048)),
                    min=1,
                    max=32768,
                    step=1,
                    tooltip="The maximum number of tokens the model is allowed to generate in its response. "
                    "Lower limits prevent long-winded answers and save generation time.",
                ),
                io.Combo.Input(
                    "attention_mode",
                    options=["auto", "flash_attention_2", "sdpa", "eager"],
                    default="auto",
                    tooltip="Attention mechanism for Hugging Face Transformers:\n"
                    "• auto: Automatically selects the best available engine\n"
                    "• flash_attention_2: Fastest, uses the least VRAM (requires Ampere/Ada GPU & flash-attn installed)\n"
                    "• sdpa: PyTorch Scaled Dot-Product Attention (good default)\n"
                    "• eager: Traditional PyTorch attention (fallback, uses most VRAM)",
                ),
                # ── Sampling and hardware ─────────────
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu", "mps"],
                    default=str(defaults.get("device", "cuda")),
                    tooltip="Hardware device to load the model onto. "
                    "Use 'auto' to prefer CUDA/ROCm and let Florence-2 fall back "
                    "to CPU when native GPU precision cannot safely fit; explicit "
                    "'cuda', 'mps', and 'cpu' selections remain strict.",
                ),
                io.Float.Input(
                    "temperature",
                    default=float(defaults.get("temperature", 0.7)),
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    tooltip="Controls randomness: higher values (e.g. 0.8+) make output more creative/diverse; "
                    "lower values (e.g. 0.2) make it more deterministic and focused.",
                ),
                io.Float.Input(
                    "top_p",
                    default=float(defaults.get("top_p", 0.9)),
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    tooltip="Nucleus sampling: limits generation to the cumulative probability tokens (e.g. 0.9 keeps top 90% likely words). "
                    "Filters out low-probability gibberish.",
                ),
                io.Int.Input(
                    "top_k",
                    default=int(defaults.get("top_k", 50)),
                    min=0,
                    max=1000,
                    step=1,
                    tooltip="Limits generation to the top K most likely next words. "
                    "Lower values (e.g. 40) make output more focused; 0 disables it.",
                ),
                io.Int.Input(
                    "num_beams",
                    default=int(defaults.get("num_beams", 1)),
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Number of parallel paths explored during beam search. "
                    "Values > 1 produce higher quality text but are significantly slower. Set to 1 for standard sampling.",
                ),
                io.Boolean.Input(
                    "do_sample",
                    default=bool(defaults.get("do_sample", True)),
                    tooltip="When enabled, uses probabilistic sampling (temperature, top_p, top_k). "
                    "When disabled, uses greedy decoding (always picking the most likely next word, ignoring temperature).",
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=float(defaults.get("repetition_penalty", 1.0)),
                    min=1.0,
                    max=2.0,
                    step=0.1,
                    tooltip="Penalizes repeating the same phrases or words. "
                    "Values > 1.0 (e.g. 1.1 or 1.2) help reduce loops and repetitive output.",
                ),
                io.Int.Input(
                    "frame_count",
                    default=int(defaults.get("frame_count", 8)),
                    min=1,
                    max=100,
                    step=1,
                    tooltip="QwenVL only — number of frames to extract and feed to QwenVL when processing video inputs. "
                    "More frames provide more temporal detail but require significantly more VRAM.",
                ),
                io.Boolean.Input(
                    "use_torch_compile",
                    default=bool(defaults.get("use_torch_compile", False)),
                    tooltip="JIT compiles the model using PyTorch 2.x compile. "
                    "Increases initial startup/load time (~1-3 minutes first run) but speeds up subsequent inference runs.",
                ),
                io.Float.Input(
                    "min_p",
                    default=float(defaults.get("min_p", 0.0)),
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="Minimum probability threshold relative to the top token. "
                    "For example, min_p=0.05 filters out any token with less than 5% probability of the most likely token. "
                    "Supported: GGUF, Ollama, vLLM, SGLang, llama.cpp.",
                ),
                io.Combo.Input(
                    "mirostat",
                    options=["0 (off)", "1 (Mirostat)", "2 (Mirostat 2.0)"],
                    default=str(defaults.get("mirostat", "0 (off)")),
                    tooltip="Mirostat sampling dynamically adjusts temperature to maintain target text quality/entropy. "
                    "Supported: GGUF, Ollama, llama.cpp.",
                ),
                io.Float.Input(
                    "mirostat_eta",
                    default=float(defaults.get("mirostat_eta", 0.1)),
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="Mirostat learning rate (eta). Controls how rapidly the temperature adjusts. Default is 0.1.",
                ),
                io.Float.Input(
                    "mirostat_tau",
                    default=float(defaults.get("mirostat_tau", 5.0)),
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Mirostat target entropy (tau). Controls target complexity/perplexity of generated text. Default is 5.0.",
                ),
                io.Int.Input(
                    "repeat_last_n",
                    default=int(defaults.get("repeat_last_n", 64)),
                    min=-1,
                    max=8192,
                    step=1,
                    tooltip="Lookback window size (number of recent tokens) to evaluate for the repetition penalty. "
                    "Set to -1 to use full context, 0 to disable. Supported: GGUF, Ollama, llama.cpp.",
                ),
                io.String.Input(
                    "stop_sequences",
                    default=str(defaults.get("stop_sequences", "")),
                    tooltip="Custom strings (separated by commas or newlines) that act as stopping triggers. "
                    "The model stops generating immediately when any of these strings are output.",
                ),
                # ── WD14 tagging ──────────
                io.Float.Input(
                    "threshold",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="WD14 Tagger only — minimum confidence threshold (0.0 to 1.0) for general tags (e.g. clothing, background) to be included in the output.",
                ),
                io.Float.Input(
                    "char_threshold",
                    default=0.85,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="WD14 Tagger only — minimum confidence threshold (0.0 to 1.0) for character/IP tags to be included.",
                ),

                io.Boolean.Input(
                    "replace_underscore",
                    default=True,
                    tooltip="WD14 Tagger only — replaces underscores in tags with spaces (e.g. 'blue_eyes' becomes 'blue eyes').",
                ),
                # ── Seed policy is resolved by the panel ──
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**64 - 1,
                    step=1,
                    tooltip="Controls generation reproducibility. Use specific values for deterministic output:\n"
                    "• -1: Randomize the seed on every execution\n"
                    "• -2: Increment the seed by 1 after each run\n"
                    "• -3: Decrement the seed by 1 after each run",
                ),
                # ── Mode options ──
                io.Boolean.Input(
                    "memory_cleanup", default=True, label_on="ON", label_off="OFF"
                ),
                io.Boolean.Input(
                    "keep_model_loaded", default=False, label_on="ON", label_off="OFF"
                ),
                io.Boolean.Input(
                    "multi_task_mode", default=False, label_on="ON", label_off="OFF"
                ),
                io.Boolean.Input(
                    "use_advanced",
                    default=True,
                    label_on="ON",
                    label_off="OFF",
                    tooltip="Apply advanced sampling parameters (temperature, top_p, top_k, etc.) to the model request. "
                    "When disabled, safe/conservative defaults are used regardless of widget settings.",
                ),
                io.Boolean.Input(
                    "use_few_shot_training",
                    default=True,
                    label_on="ON",
                    label_off="OFF",
                    tooltip="Append task-specific few-shot training examples (prompt/response pairs) to the context. "
                    "Helps the model follow formatting instructions but increases token count.",
                ),
                io.Boolean.Input(
                    "nsfw_training",
                    default=False,
                    optional=True,
                    tooltip="Select NSFW training examples when enabled; otherwise use Standard. "
                    "Applies only while Training is enabled. Stored in this node, without changing config.json.",
                ),
                # ── Connection slots ──────────────────────────────────
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Input image or video batch. Required for vision tasks (VLM) and WD14 tagging. Ignored for text-only LLMs.",
                ),
                io.String.Input(
                    "system_prompt",
                    optional=True,
                    force_input=True,
                    tooltip="Override system instructions. Connecting this slot automatically switches the task to 'Direct Chat' and bypasses default task prompt templates.",
                ),
            ],
            is_input_list=True,
            outputs=[
                io.Image.Output(
                    "image", is_output_list=True, tooltip="Passthrough of input images."
                ),
                io.String.Output(
                    "text", is_output_list=True, tooltip="Generated text or tags."
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )


    @classmethod
    @with_execution_cleanup(_LOG_PREFIX)
    @with_training_profile
    def _execute_engine(
        cls,
        model,
        quantization,
        task,
        task_2,
        task_3,
        task_4,
        user_prompt,
        max_tokens,
        context_size,
        attention_mode,
        seed,
        # Advanced
        device,
        temperature,
        top_p,
        top_k,
        num_beams,
        do_sample,
        repetition_penalty,
        frame_count,
        use_torch_compile,
        # WD14
        threshold,
        char_threshold,
        replace_underscore,
        # Mode options
        multi_task_mode,
        memory_cleanup,
        keep_model_loaded,
        use_advanced,
        use_few_shot_training,
        # Advanced sampling extras
        min_p,
        mirostat,
        mirostat_eta,
        mirostat_tau,
        repeat_last_n,
        stop_sequences,
        # Optional connections
        images=None,
        system_prompt=None,
    ):
        start_time = time.time()

        # Extract unique_id (node ID) from hidden inputs
        uid = getattr(cls.hidden, "unique_id", None)
        node_id = None
        if uid is not None:
            if isinstance(uid, list):
                if len(uid) > 0:
                    try:
                        node_id = int(uid[0])
                    except Exception:
                        node_id = uid[0]
            else:
                try:
                    node_id = int(uid)
                except Exception:
                    node_id = uid

        initial_title = cls.define_schema().display_name
        extra_pnginfo = getattr(cls.hidden, "extra_pnginfo", None)
        if node_id is not None and extra_pnginfo is not None:
            pnginfo = (
                extra_pnginfo[0] if isinstance(extra_pnginfo, list) else extra_pnginfo
            )
            if isinstance(pnginfo, dict) and "workflow" in pnginfo:
                nodes = pnginfo["workflow"].get("nodes", [])
                node_info = next(
                    (n for n in nodes if str(n.get("id")) == str(node_id)), None
                )
                if node_info:
                    initial_title = node_info.get("title") or initial_title

        # Safeguard: if the canvas title was left stuck as a progress state (e.g. from an interrupted or errored previous run),
        # discard it and fallback to the default clean name derived from the node schema.
        if initial_title:
            temp_statuses = ["%", "Scanning", "Probing", "Loading", "Resizing", "Tagging"]
            is_temp = any(status in initial_title for status in temp_statuses) or initial_title.endswith("...")
            if is_temp:
                initial_title = cls.define_schema().display_name

        def update_title(sub_status: str):
            if node_id is not None:
                try:
                    from server import PromptServer

                    PromptServer.instance.send_sync(
                        "smartllm-nova/update_node_title",
                        {"node_id": node_id, "title": sub_status},
                    )
                except Exception:
                    pass

        def restore_title():
            if node_id is not None:
                try:
                    from server import PromptServer

                    PromptServer.instance.send_sync(
                        "smartllm-nova/update_node_title",
                        {"node_id": node_id, "title": initial_title},
                    )
                except Exception:
                    pass


        model = _unwrap_scalar(model, "")
        quantization = _unwrap_scalar(quantization, "")
        task = canonicalize_task_name(_unwrap_scalar(task, ""))
        task_2 = canonicalize_task_name(_unwrap_scalar(task_2, "None"))
        task_3 = canonicalize_task_name(_unwrap_scalar(task_3, "None"))
        task_4 = canonicalize_task_name(_unwrap_scalar(task_4, "None"))
        max_tokens = _unwrap_scalar(max_tokens, 2048)
        context_size = _unwrap_scalar(context_size, 8192)
        attention_mode = _unwrap_scalar(attention_mode, "auto")
        seed = _unwrap_scalar(seed, 0)
        device = _unwrap_scalar(device, "cuda")
        temperature = _unwrap_scalar(temperature, 0.7)
        top_p = _unwrap_scalar(top_p, 0.9)
        top_k = _unwrap_scalar(top_k, 50)
        num_beams = _unwrap_scalar(num_beams, 1)
        do_sample = _unwrap_scalar(do_sample, True)
        repetition_penalty = _unwrap_scalar(repetition_penalty, 1.0)
        frame_count = _unwrap_scalar(frame_count, 8)
        use_torch_compile = _unwrap_scalar(use_torch_compile, False)
        threshold = _unwrap_scalar(threshold, 0.35)
        char_threshold = _unwrap_scalar(char_threshold, 0.85)
        replace_underscore = _unwrap_scalar(replace_underscore, True)
        multi_task_mode = _unwrap_scalar(multi_task_mode, False)
        memory_cleanup = _unwrap_scalar(memory_cleanup, True)
        keep_model_loaded = _unwrap_scalar(keep_model_loaded, False)
        use_advanced = _unwrap_scalar(use_advanced, True)
        use_few_shot_training = _unwrap_scalar(use_few_shot_training, True)
        min_p = _unwrap_scalar(min_p, 0.0)
        mirostat = _unwrap_scalar(mirostat, "0 (off)")
        mirostat_eta = _unwrap_scalar(mirostat_eta, 0.1)
        mirostat_tau = _unwrap_scalar(mirostat_tau, 5.0)
        repeat_last_n = _unwrap_scalar(repeat_last_n, 64)
        stop_sequences = _unwrap_scalar(stop_sequences, "")
        system_prompt = _unwrap_scalar(system_prompt, None)
        has_custom_system_prompt = bool(system_prompt and str(system_prompt).strip())

        # Normalize user_prompt to a list of strings
        if user_prompt is None:
            user_prompts = [""]
        elif isinstance(user_prompt, list):
            user_prompts = [str(p) if p is not None else "" for p in user_prompt]
        else:
            user_prompts = [str(user_prompt)]

        # Flatten/normalize input images to a list of tensors. H3 references are
        # one ordered group (first frame, then last frame), even when ComfyUI
        # supplies a batch/list that ordinary tasks would map across.
        h3_mode = get_h3_mode(task)
        h3_image_count = 0
        h3_resolved_mode = None
        if h3_mode is not None:
            grouped, h3_image_count, h3_resolved_mode = _prepare_h3_reference_group(
                task, images
            )
            flat_images = [grouped] if grouped is not None else []
        else:
            flat_images = []

            def _process_image(item):
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        _process_image(sub)
                elif isinstance(item, torch.Tensor):
                    # Video tasks keep a 4D batch as a single video input.
                    is_video_task = (
                        "video" in str(task).lower()
                        or "timeline" in str(task).lower()
                    )
                    if item.dim() == 4:
                        if is_video_task:
                            flat_images.append(item)
                        else:
                            for i in range(item.shape[0]):
                                flat_images.append(item[i : i + 1, ...])
                    elif item.dim() == 3:
                        flat_images.append(item.unsqueeze(0))
                elif item is not None:
                    flat_images.append(item)

            if images is not None:
                _process_image(images)

        # ── Use Advanced gate ──────────────────────────────────
        # When OFF, override sampling params with conservative defaults (matches
        # panel defaults & backend function defaults). Stored sampling values
        # are applied only while the Sampling option is enabled.
        # Not gated: seed, max_tokens, context_size, device, frame_count,
        # use_torch_compile, attention_mode, WD14 params.
        if not use_advanced:
            temperature = 0.7
            top_p = 0.9
            top_k = 50
            num_beams = 1
            do_sample = True
            repetition_penalty = 1.0
            min_p = 0.0
            mirostat = "0 (off)"
            mirostat_eta = 0.1
            mirostat_tau = 5.0
            repeat_last_n = 64
            stop_sequences = ""

        # Parse mirostat combo "N (label)" → int
        try:
            mirostat_int = int(str(mirostat).split(" ")[0]) if mirostat else 0
        except (ValueError, AttributeError):
            mirostat_int = 0

        # Parse stop_sequences multiline → list[str] (None when empty so backends omit the option)
        stop_list = [
            s.strip()
            for s in (stop_sequences or "").replace("\n", ",").split(",")
            if s.strip()
        ] or None

        # ── 1. Registry lookup ──────────────────────────────────
        entry = get_model_entry(model)
        if entry is None:
            raise ValueError(f"Model '{model}' not found in registry")

        backend = entry["backend"]
        repo_id = entry.get("repo_id", "")
        family_str = entry.get("family", "")
        model_has_vision = entry.get("has_vision", False)
        loading_method = _BACKEND_TO_METHOD.get(backend, "Transformers")
        model_family = _FAMILY_TO_EXEC.get(
            family_str, "VLM" if model_has_vision else "LLM (Text-Only)"
        )
        if h3_image_count and not model_has_vision:
            raise ValueError(
                f"{task} {h3_resolved_mode} mode requires a vision-language model "
                "to inspect its reference image(s)."
            )

        log.msg(
            _LOG_PREFIX, f"Model: {model} | backend={backend} | family={model_family}"
        )

        # ── 2. WD14 fast-path ───────────────────────────────────
        if backend == "wd14":
            return _execute_wd14(
                repo_id=repo_id,
                revision=entry.get("revision"),
                expected_sha256=entry.get("expected_sha256"),
                device=device,
                images=images,
                threshold=threshold,
                char_threshold=char_threshold,
                replace_underscore=replace_underscore,
                keep_model_loaded=keep_model_loaded,
                node_id=node_id,
                initial_title=initial_title,
            )

        # ── 3. Resolve model path ──────────────────────────────
        model_path, needs_download = _resolve_model_path(entry, quantization)

        enforce_registry_provenance = bool(
            backend
            in (
                "transformers",
                "gguf",
                "llamacpp",
                "vllm",
                "vllm_native",
                "sglang",
            )
            and (
                entry.get("revision") is not None
                or entry.get("expected_sha256") is not None
            )
        )
        if needs_download or enforce_registry_provenance:
            if needs_download:
                log.msg(
                    _LOG_PREFIX,
                    f"Model not found locally, downloading: {repo_id}",
                )
            else:
                log.debug(
                    _LOG_PREFIX,
                    "Validating local model against registry provenance",
                )
            model_path = _ensure_downloaded(
                entry,
                quantization,
                local_model_path=None if needs_download else model_path,
            )
            # Reset progress bar after download so generation progress starts fresh
            import comfy.utils  # type: ignore

            comfy.utils.ProgressBar(1).update_absolute(0, 1)

        log.debug(_LOG_PREFIX, f"Model path: {model_path}")

        # ── 4. Build TemplateContext (adapter for load_model_with_backend) ──
        backend_quantization = (
            quantization
            if backend in _GGUF_BACKENDS
            else entry.get("quantization", "auto")
        )
        ctx = TemplateContext.from_widgets(
            model_family=model_family,
            model_type="",
            loading_method=loading_method,
            quantization=backend_quantization,
            attention_mode=attention_mode,
            repo_id=repo_id,
            local_path=model_path,
            quantized=False,
            default_task="",
            has_vision=model_has_vision,
            max_tokens=max_tokens,
            context_size=context_size,
        )

        # GGUF: set mmproj from registry
        if backend in _GGUF_BACKENDS and entry.get("mmproj"):
            mmproj_path = _resolve_mmproj_path(entry, model_path)
            if mmproj_path:
                ctx.mmproj_path = mmproj_path

        # Ollama: set model name
        if backend == "ollama":
            ctx.update(model_source="Ollama", ollama_model=repo_id)

        if loading_method in (
            "GGUF (llama-cpp-python)",
            "vLLM (Docker)",
            "SGLang (Docker)",
            "Ollama (Docker)",
            "llama.cpp (Docker)",
        ):
            ctx.context_size = context_size

        # ── 5. Load model ──────────────────────────────────────
        # Read n_batch from defaults (GGUF only, no widget)
        n_batch = int(load_defaults().get("n_batch", 512)) if backend == "gguf" else 512

        cleanup_state = {"instance": None}
        if not keep_model_loaded:
            register_execution_cleanup(
                lambda: _cleanup_model(
                    loading_method=loading_method,
                    keep_model_loaded=keep_model_loaded,
                    model_path=model_path,
                    instance=cleanup_state["instance"],
                )
            )

        model_obj, processor, model_type = load_model_with_backend(
            loading_method=loading_method,
            model_family=model_family,
            model_path=model_path,
            ctx=ctx,
            quantization=backend_quantization,
            attention_mode=attention_mode,
            device=device,
            context_size=context_size,
            n_batch=n_batch,
            memory_cleanup=memory_cleanup,
            keep_model_loaded=keep_model_loaded,
            use_torch_compile=use_torch_compile,
            repo_id=repo_id,
            revision=entry.get("revision"),
            expected_sha256=entry.get("expected_sha256"),
            tensor_parallel_size=entry.get("tensor_parallel"),
            data_parallel_size=entry.get("data_parallel"),
            text_encoder_entry=entry if backend == "comfyui" else None,
        )
        cleanup_state["instance"] = model_obj

        log.debug(_LOG_PREFIX, f"Model loaded: type={model_type}")

        # Build wrapper instance (same pattern as v3)
        if getattr(model_obj, "is_comfyui", False):
            instance = model_obj
        elif hasattr(model_obj, "is_vllm") and model_obj.is_vllm:
            instance = model_obj
            instance.model_type = model_type
        elif hasattr(model_obj, "is_sglang") and model_obj.is_sglang:
            instance = model_obj
            instance.model_type = model_type
        elif hasattr(model_obj, "is_ollama") and model_obj.is_ollama:
            instance = model_obj
            instance.model_type = model_type
        elif hasattr(model_obj, "is_llamacpp_docker") and model_obj.is_llamacpp_docker:
            instance = model_obj
            instance.model_type = model_type
        else:

            class _Wrapper:
                def __init__(self, m, p, mt, is_gguf, ctx, keep):
                    self.model = m
                    self.processor = p
                    self.model_type = mt
                    self.is_gguf = is_gguf
                    self.is_vllm = False
                    effective_quantization = getattr(
                        m, "_smartllm_effective_quantization", ctx.quantization
                    )
                    self.is_quantized = effective_quantization not in (
                        None,
                        "auto",
                        "fp16",
                        "bf16",
                        "fp32",
                        "none",
                    )
                    self.effective_device = getattr(
                        m, "_smartllm_effective_device", None
                    )
                    self.effective_dtype = getattr(m, "_smartllm_effective_dtype", None)
                    self.keep_model_loaded = keep
                    self.tokenizer = getattr(p, "tokenizer", None) or p
                    self.chat_handler_ref = getattr(
                        m,
                        "_smartllm_chat_handler",
                        getattr(
                            m,
                            "_sml_chat_handler",
                            getattr(m, "_eclipse_chat_handler", None),
                        ),
                    )

            instance = _Wrapper(
                model_obj,
                processor,
                model_type,
                is_gguf=(backend == "gguf"),
                ctx=ctx,
                keep=keep_model_loaded,
            )
        cleanup_state["instance"] = instance

        # ── 6. Resolve execution family ────────────────────────
        # Normalize unknown families to generic VLM or LLM path
        _KNOWN_FAMILIES = {
            "Qwen",
            "Florence",
            "Mistral",
            "LLaVA",
            "VLM",
            "LLM (Text-Only)",
        }
        if model_family not in _KNOWN_FAMILIES:
            model_family = "VLM" if model_has_vision else "LLM (Text-Only)"
            log.warning(
                _LOG_PREFIX,
                f"Unknown family '{family_str}' → routing via {model_family}",
                notify=True,
            )

        # Align inputs for list execution. The H3 group occupies one list item,
        # so one story produces one generation with both endpoints attached.
        prompts_aligned, images_aligned = _align_execution_inputs(
            user_prompts, flat_images
        )
        num_runs = len(prompts_aligned)
        generation_frame_count = _generation_frame_count(
            task, frame_count, h3_image_count
        )

        results = []
        output_images = []

        import comfy.utils  # type: ignore
        pbar = comfy.utils.ProgressBar(num_runs)

        try:
            for run_idx in range(num_runs):
                percent = (run_idx * 100) // num_runs
                update_title(f"{initial_title} ({percent}%)")

                single_prompt = prompts_aligned[run_idx]
                single_img = images_aligned[run_idx]

                # Prepare input image for generation
                input_image = None
                if single_img is not None and model_has_vision:
                    input_image = single_img
                elif model_has_vision and single_img is None:
                    log.warning(_LOG_PREFIX, "No image provided for vision model", notify=True)

                # ── Generate (with system-prompt override if connected) ───
                _override_token = push_system_prompt_override(system_prompt)
                try:
                    run_res, run_data = _generate_with_h3_recovery(
                        model_family=model_family,
                        instance=instance,
                        task_name=task,
                        user_prompt=single_prompt,
                        input_image=input_image,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        seed=seed,
                        repetition_penalty=repetition_penalty,
                        context_size=context_size,
                        frame_count=generation_frame_count,
                        use_few_shot=use_few_shot_training,
                        min_p=min_p,
                        mirostat=mirostat_int,
                        mirostat_eta=mirostat_eta,
                        mirostat_tau=mirostat_tau,
                        repeat_last_n=repeat_last_n,
                        stop_sequences=stop_list,
                        h3_story_mode=_should_use_h3_story_mode(
                            task,
                            single_prompt,
                            system_prompt if has_custom_system_prompt else None,
                        ),
                    )
                finally:
                    reset_system_prompt_override(_override_token)

                # ── Multi-task chaining ─────────────────────────────
                if multi_task_mode and model_family != "Florence":
                    tasks_to_run = [task]
                    for t in [task_2, task_3, task_4]:
                        if t and t != "None":
                            tasks_to_run.append(t)
                        else:
                            break

                    if len(tasks_to_run) > 1:
                        run_res, run_data = _run_multi_task_chain(
                            tasks_to_run=tasks_to_run,
                            first_result=run_res,
                            first_data=run_data,
                            instance=instance,
                            model_family=model_family,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            num_beams=num_beams,
                            do_sample=do_sample,
                            seed=seed,
                            repetition_penalty=repetition_penalty,
                            context_size=context_size,
                            frame_count=frame_count,
                            use_few_shot=use_few_shot_training,
                            min_p=min_p,
                            mirostat=mirostat_int,
                            mirostat_eta=mirostat_eta,
                            mirostat_tau=mirostat_tau,
                            repeat_last_n=repeat_last_n,
                            stop_sequences=stop_list,
                        )

                results.append(run_res)
                # Keep output_image passthrough matching input image layout
                img_out = (
                    single_img if single_img is not None else torch.zeros((1, 64, 64, 3))
                )
                output_images.append(img_out)
                pbar.update(1)
        finally:
            restore_title()

        elapsed = time.time() - start_time
        log.msg(
            _LOG_PREFIX, f"Done ({elapsed:.1f}s) — {sum(len(r) for r in results)} chars"
        )

        # ── 11. Persist-on-execute: save changed defaults ──────
        # Only persist sampling params when Use Advanced is ON — otherwise the
        # gated defaults would overwrite the user's saved tuning.
        if use_advanced:
            _persist_defaults(
                context_size=context_size,
                device=device,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                frame_count=frame_count,
                use_torch_compile=use_torch_compile,
                min_p=min_p,
                mirostat=mirostat,
                mirostat_eta=mirostat_eta,
                mirostat_tau=mirostat_tau,
                repeat_last_n=repeat_last_n,
                stop_sequences=stop_sequences,
            )
        else:
            _persist_defaults(
                context_size=context_size,
                device=device,
                frame_count=frame_count,
                use_torch_compile=use_torch_compile,
            )

        return io.NodeOutput(output_images, results)


# ============================================================================
# Persist-on-Execute
# ============================================================================


def _persist_defaults(**kwargs):
    # Compare current values against stored defaults and save changes.
    defaults = load_defaults()
    updates = {}
    for key, value in kwargs.items():
        if defaults.get(key) != value:
            updates[key] = value
    if updates:
        save_defaults(updates)
