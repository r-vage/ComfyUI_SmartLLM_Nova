"""Protected model operations and output contracts, independent of task titles."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FlorenceOperation:
    token: str
    text: bool = False
    spatial: bool = False
    promptgen: int = 0


FLORENCE_OPERATIONS = {
    "caption": FlorenceOperation("<CAPTION>"),
    "detailed_caption": FlorenceOperation("<DETAILED_CAPTION>"),
    "more_detailed_caption": FlorenceOperation("<MORE_DETAILED_CAPTION>"),
    "ocr": FlorenceOperation("<OCR>"),
    "od": FlorenceOperation("<OD>", spatial=True),
    "caption_to_phrase_grounding": FlorenceOperation("<CAPTION_TO_PHRASE_GROUNDING>", text=True, spatial=True),
    "dense_region_caption": FlorenceOperation("<DENSE_REGION_CAPTION>", spatial=True),
    "region_proposal": FlorenceOperation("<REGION_PROPOSAL>", spatial=True),
    "referring_expression_segmentation": FlorenceOperation("<REFERRING_EXPRESSION_SEGMENTATION>", text=True, spatial=True),
    "ocr_with_region": FlorenceOperation("<OCR_WITH_REGION>", spatial=True),
    "docvqa": FlorenceOperation("<DocVQA>", text=True),
    "prompt_gen_tags": FlorenceOperation("<GENERATE_TAGS>", promptgen=1),
    "prompt_gen_mixed_caption": FlorenceOperation("<MIXED_CAPTION>", promptgen=1),
    "prompt_gen_analyze": FlorenceOperation("<ANALYZE>", promptgen=2),
    "prompt_gen_mixed_caption_plus": FlorenceOperation("<MIXED_CAPTION_PLUS>", promptgen=2),
}

# Input/output protocols are code. Task files can select, not redefine them.
TASK_GROUPS = {
    "chat": "Chat & Text",
    "image": "Image Description",
    "video": "Video & Scenes",
    "tags": "Tags",
    "creative": "Stories & Music",
    "ocr": "OCR",
    "detection": "Detection",
}
LEGACY_TASK_GROUPS = {"text": "chat", "vision": "image"}


HANDLERS = {
    "chat": "text", "direct_chat": "text", "story": "text", "tags_to_text": "text", "text_to_tags": "text",
    "song_lyrics": "lyrics", "music3": "music3", "yue2": "yue2",
    "scene_motion": "scene", "scene_description": "scene",
    "h3_scene": "h3", "h3_t2va": "h3", "h3_i2va": "h3", "h3_fl2va": "h3", "h3_l2va": "h3",
    "grounding": "geometry", "region_caption": "geometry", "florence": "native",
}


def florence_operation(key):
    # The old region_caption key remains a protocol alias, never a display name.
    key = "od" if key == "region_caption" else key
    return FLORENCE_OPERATIONS.get(key)


def model_issue(task, entry, kind="loader"):
    if kind not in task.nodes:
        return "This task belongs to a different node."
    if not task.enabled or task.internal:
        return "This task is a draft or an internal operation."
    backend, family = entry.get("backend", ""), entry.get("family", "")
    if backend in {"wd14", "yolo"}:
        return "This model uses its native tagging/detection controls."
    if family == "Florence":
        operation = florence_operation(task.florence_id)
        if operation is None:
            return "This Florence model needs a supported native operation."
        name = str(entry.get("repo_id", entry.get("name", ""))).lower()
        supported = entry.get("task_operations")
        if supported is not None:
            if not isinstance(supported, list) or task.florence_id not in supported:
                return "This operation is not enabled for this checkpoint."
        elif operation.promptgen and ("promptgen" not in name or operation.promptgen == 2 and "v2" not in name):
            return "This operation requires a compatible PromptGen checkpoint."
        return ""
    if task.handler == "florence":
        return "This task requires Florence."
    if task.needs_image and entry.get("has_vision") is False:
        return "This task requires a model with vision weights."
    if kind == "detection" and task.handler not in {"grounding", "region_caption"}:
        return "This detection task requires Florence."
    return ""


def chain_issue(task, entry, previous=()):
    issue = model_issue(task, entry)
    if issue or not previous:
        return issue
    if entry.get("family") == "Florence":
        return "Florence does not support text task chaining."
    if task.needs_image:
        return "This task needs original images, not the previous text result."
    if task.definition['input']['text'] == 'none':
        return "This task cannot receive the previous text result."
    if previous[-1].output not in {"text", "lyrics"}:
        return "The previous task has a structured output contract; use a separate node."
    if previous[-1].output == "lyrics" and task.handler not in {"music3", "yue2", "chat", "direct_chat", "story"}:
        return "This task cannot consume a lyric sheet."
    return ""
