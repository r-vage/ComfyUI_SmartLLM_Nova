"""Native main controls shared by Nova settings helpers and processing panels."""

from comfy_api.latest import io

from .model_registry import (
    get_detection_model_list,
    get_model_list,
    is_model_separator,
    load_defaults,
)
from .settings_pipe import SYSTEM_PROMPT_SPEC
from .tasks import get_detection_task_names, get_task_names

SETTINGS_FIELDS = {
    "loader": ("model", "task", "quantization", "user_prompt", "context_size", "max_tokens", "system_prompt"),
    "detection": ("model_name", "task", "quantization", "user_input", "confidence",
                  "nms_iou_threshold", "detection_filter", "select_index"),
}
SETTINGS_TYPES = {"loader": "SMARTLLM_LM_SETTINGS", "detection": "SMARTLLM_DETECTION_SETTINGS"}


def settings_fields(kind, *, models=None, task_names=None, defaults=None):
    if kind not in SETTINGS_FIELDS:
        raise ValueError(f"Unknown settings kind: {kind}")
    if models is None:
        models = get_model_list() if kind == "loader" else get_detection_model_list()
    if task_names is None:
        task_names = (get_task_names(has_vision=True, include_all_families=True)
                      if kind == "loader" else get_detection_task_names())
    if defaults is None:
        defaults = load_defaults()
    first_model = next((name for name in models if not is_model_separator(name)), "")
    quant_placeholders = defaults.get("quantizations", ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    if kind == "loader":
        fields = [
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
                "task",
                options=task_names,
                default="Detailed Description",
                tooltip="The primary task to run. Determines the prompt template, system instructions, and training examples. "
                "Note: Vision tasks (VLM) require feeding an image into the 'images' input slot.",
            ),
            io.Combo.Input(
                "quantization",
                options=quant_placeholders,
                default="Q4_K_M",
                tooltip="Choose GGUF quantization precision. Lower bits (e.g. Q4_K_M) use less VRAM but lose accuracy. "
                "Higher bits (e.g. Q8_0) are more accurate but demand more VRAM. Applies to native GGUF and llama.cpp Docker models.",
            ),
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
            # Append so saved native helper widget values retain their positions.
            io.String.Input(
                "system_prompt", default="", multiline=True, optional=True,
                tooltip=SYSTEM_PROMPT_SPEC["tip"],
            ),
        ]
    else:
        fields = [
            io.Combo.Input(
                "model_name",
                options=models,
                default=first_model,
                tooltip="Choose the object detection or VLM grounding model to load.\n"
                "Suffixes indicate the backend engine (no suffix=Transformers, -GGUF=native llama-cpp-python, "
                "-llama.cpp=llama.cpp Docker).",
            ),
            io.Combo.Input(
                "task",
                options=task_names,
                default=(
                    task_names[0] if task_names else "Caption to Phrase Grounding"
                ),
                tooltip="Type of detection task to run:\n"
                "• Caption to Phrase Grounding: Locates objects matching user_input query.\n"
                "• Referring Expression Segmentation: Generates segmented mask regions for specified objects.\n"
                "• Region Caption: Detects objects and labels them with descriptions.\n"
                "• YOLO runs object detection automatically.",
            ),
            io.Combo.Input(
                "quantization",
                options=quant_placeholders,
                default="Q4_K_M",
                tooltip="Choose GGUF quantization precision. Lower bits (e.g. Q4_K_M) use less VRAM but lose accuracy. "
                "Higher bits (e.g. Q8_0) are more accurate but demand more VRAM. Applies to native GGUF and llama.cpp Docker models.",
            ),
            io.String.Input(
                "user_input",
                default="",
                multiline=True,
                tooltip="Input query for detection:\n"
                "• Phrase grounding: Separate targets with commas, semicolons or newlines (e.g. 'eye, breast'). "
                "Each target runs a separate query using the same loaded model; results are combined.\n"
                "• YOLO: Semicolon-separated target classes to filter (e.g. 'person;car;backpack'). Leave empty to output all classes.",
            ),
            io.Float.Input(
                "confidence",
                default=float(defaults.get("confidence", 0.5)),
                min=0.0,
                max=1.0,
                step=0.1,
                tooltip="Minimum confidence score (0.0 to 1.0) required for a detection to be kept. "
                "Higher values reduce false positives, lower values capture more candidate objects.",
            ),
            io.Float.Input(
                "nms_iou_threshold",
                default=float(defaults.get("nms_iou_threshold", 0.5)),
                min=0.0,
                max=1.0,
                step=0.1,
                tooltip="Non-Maximum Suppression (NMS) intersection-over-union threshold. "
                "Lower values (e.g. 0.3) aggressively merge overlapping boxes; higher values (e.g. 0.7) keep separate but close detections.",
            ),
            io.Float.Input(
                "detection_filter",
                default=float(defaults.get("detection_filter", 0.8)),
                min=0.0,
                max=1.0,
                step=0.1,
                tooltip="Max ratio of bounding box area to total image area. "
                "Detections covering more than this ratio (e.g. 0.8) are ignored to filter out useless full-image bounding boxes.",
            ),
            io.Int.Input(
                "select_index",
                default=int(defaults.get("select_index", -1)),
                min=-1,
                max=999,
                step=1,
                tooltip="Select a single detection: set to -1 to output all detections merged; "
                "set to 0, 1, 2... to output only the N-th detection (useful for isolating objects).",
            ),
        ]
    return {field.id: field for field in fields}
