# SmartLLM Task Definitions
#
# Hardcoded task structure — developer data that changes only when tasks are
# added or removed, never by users. System prompts (user-editable) live in
# config/system_prompts.json and are loaded separately.
#
# Usage:
#     from ..core.tasks import (
#         ALL_TASKS, TASK_BY_NAME, FLORENCE_ID_TO_TASK,
#         get_task_names, get_detection_task_names,
#         get_florence_token, is_florence_task, get_system_prompt,
#     )

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Optional

from .logger import log

_LOG_PREFIX = "Tasks"


# ============================================================================
# Task Dataclass
# ============================================================================


@dataclass(frozen=True)
class Task:
    name: str  # Display name in dropdown
    category: str  # "custom" | "vision" | "text" | "detection"
    needs_image: bool  # True = requires image input
    florence_id: Optional[str] = None  # Florence machine key (e.g. "caption")
    florence_token: Optional[str] = None  # Florence prompt token (e.g. "<CAPTION>")
    family_filter: Optional[str] = (
        None  # None = all families, "Florence" = Florence-only
    )


@dataclass(frozen=True)
class H3Mode:
    task_name: str
    input_mode: str
    duration_seconds: int | None
    allowed_image_counts: tuple[int, ...]
    timeline: bool

    @property
    def needs_image(self) -> bool:
        return 0 not in self.allowed_image_counts

    def resolve_input_mode(self, image_count: int) -> str:
        if image_count not in self.allowed_image_counts:
            expected = (
                str(self.allowed_image_counts[0])
                if len(self.allowed_image_counts) == 1
                else "/".join(str(count) for count in self.allowed_image_counts)
            )
            noun = "image" if expected == "1" else "images"
            raise ValueError(
                f"{self.task_name} requires {expected} {noun}; received {image_count}."
            )
        if self.input_mode == "AUTO":
            return {0: "T2VA", 1: "I2VA", 2: "FL2VA"}[image_count]
        return self.input_mode


# ============================================================================
# Task Constants
# ============================================================================

# ── Custom tasks ──────────────────────────────────────────────────
TASK_DIRECT_CHAT = Task("Direct Chat", "custom", False)
TASK_QUESTION_ANSWER = Task("Question Answering", "custom", False)
TASK_CUSTOM_INSTRUCT = Task("Custom Instruction", "custom", False)
TASK_WAN_SCENE_5S = Task("Wan 2.2 Scene 5s", "custom", False)
TASK_WAN_TIMELINE_5S = Task("Wan 2.2 Timeline 5s", "custom", False)
TASK_WAN_TIMELINE_5S_2 = Task("Wan 2.2 Timeline 5s 2s", "custom", False)
TASK_WAN_TIMELINE_5S_3 = Task("Wan 2.2 Timeline 5s 3s", "custom", False)
TASK_WAN_SCENE_10S = Task("Wan 2.2 Scene 10s", "custom", False)
TASK_WAN_TIMELINE_10S = Task("Wan 2.2 Timeline 10s", "custom", False)
TASK_WAN_SCENE_20S = Task("Wan 2.2 Scene 20s", "custom", False)
TASK_WAN_TIMELINE_20S = Task("Wan 2.2 Timeline 20s", "custom", False)
TASK_WAN_CN_ATOMIC = Task("Wan 2.2 CN Atomic", "custom", False)
TASK_LTX_23_I2V = Task("LTX 2.3 I2V", "custom", False)

MINIMAX_H3_SCENE_TASK = "MiniMax H3 Scene"
LEGACY_TASK_ALIASES: dict[str, str] = {
    "YuE2": "YuE2 Music",
    "MiniMax H3 Scene 5s": MINIMAX_H3_SCENE_TASK,
    "MiniMax H3 T2VA Timeline 15s": "MiniMax H3 T2VA Timeline",
    "MiniMax H3 I2VA Timeline 15s": "MiniMax H3 I2VA Timeline",
    "MiniMax H3 FL2VA Timeline 15s": "MiniMax H3 FL2VA Timeline",
    "MiniMax H3 L2VA Timeline 15s": "MiniMax H3 L2VA Timeline",
}


def canonicalize_task_name(task_name: str) -> str:
    # Keep retired serialized values executable without advertising them.
    return LEGACY_TASK_ALIASES.get(task_name, task_name)


def is_h3_scene_task(task_name: str) -> bool:
    return canonicalize_task_name(task_name) == MINIMAX_H3_SCENE_TASK


H3_MODE_METADATA: dict[str, H3Mode] = {
    mode.task_name: mode
    for mode in (
        H3Mode(MINIMAX_H3_SCENE_TASK, "AUTO", None, (0, 1, 2), False),
        H3Mode("MiniMax H3 T2VA Timeline", "T2VA", None, (0,), True),
        H3Mode("MiniMax H3 I2VA Timeline", "I2VA", None, (1,), True),
        H3Mode("MiniMax H3 FL2VA Timeline", "FL2VA", None, (1, 2), True),
        H3Mode("MiniMax H3 L2VA Timeline", "L2VA", None, (1,), True),
    )
}

_H3_MUSIC_INTENT_POLICY = (
    "MUSIC INTENT (mandatory): Treat the user's music request as authoritative. "
    "If the user requests no background music, no non-diegetic music, no score, "
    "no soundtrack, or equivalent, return exactly non_diegetic_music: N/A. Never "
    "invent music or move it into overall_soundscape. Keep requested or "
    "scene-grounded ambience, dialogue, and action sounds in overall_soundscape "
    "unless the user also requests silence."
)

_H3_EMPTY_SCENE_STORY_POLICY = (
    "EMPTY-PROMPT STORY MODE (highest priority): The user prompt is blank and no "
    "custom system prompt is connected, so create an original, coherent cinematic "
    "mini-story as one continuous [Shot 1] with no timestamps or cuts. With zero "
    "images, invent the setting, characters, visible action, and story progression. "
    "With one image, ground the story in Picture 1 and preserve its visible subjects, "
    "appearance, objects, composition, environment, style, and lighting. With two "
    "ordered images, preserve both endpoints and create a natural visible progression "
    "from Picture 1 to Picture 2. This story-mode instruction overrides ordinary "
    "prohibitions on inventing goals, events, and dialogue only for this blank-input "
    "case; attached images remain authoritative. Add plausible dialogue only when the "
    "scene naturally supports speech, using speaker IDs such as (S1) and "
    "<d>[Language] dialogue</d>; do not force dialogue into silent scenes. Always "
    "plan pacing within MiniMax H3's maximum 15-second range when a duration bound "
    "is useful, without adding timestamps or extra shots. Always "
    "provide synchronized ambience and action sounds in overall_soundscape and a "
    "scene-appropriate background score in non_diegetic_music. "
    "non_diegetic_music must never be N/A in this mode. Return only the three required "
    "fields in their required order."
)


def _h3_task(task_name: str) -> Task:
    mode = H3_MODE_METADATA[task_name]
    return Task(mode.task_name, "custom", mode.needs_image)


TASK_MINIMAX_H3_SCENE = _h3_task(MINIMAX_H3_SCENE_TASK)
TASK_MINIMAX_H3_T2VA_TIMELINE = _h3_task("MiniMax H3 T2VA Timeline")
TASK_MINIMAX_H3_I2VA_TIMELINE = _h3_task("MiniMax H3 I2VA Timeline")
TASK_MINIMAX_H3_FL2VA_TIMELINE = _h3_task("MiniMax H3 FL2VA Timeline")
TASK_MINIMAX_H3_L2VA_TIMELINE = _h3_task("MiniMax H3 L2VA Timeline")

# ── Vision tasks (all families) ───────────────────────────────────
TASK_SIMPLE_DESC = Task("Simple Description", "vision", True, "caption", "<CAPTION>")
TASK_DETAILED_DESC = Task(
    "Detailed Description", "vision", True, "detailed_caption", "<DETAILED_CAPTION>"
)
TASK_ULTRA_DESC = Task(
    "Ultra Detailed Description",
    "vision",
    True,
    "more_detailed_caption",
    "<MORE_DETAILED_CAPTION>",
)
TASK_CINEMATIC_DESC = Task("Cinematic Description", "vision", True)
TASK_IMAGE_ANALYSIS = Task("Image Analysis", "vision", True)
TASK_DETAILED_ANALYSIS = Task("Detailed Analysis", "vision", True)
TASK_TAGS = Task("Tags", "vision", True, "prompt_gen_tags", "<GENERATE_TAGS>")
TASK_VIDEO_SUMMARY = Task("Video Summary", "vision", True)
TASK_OCR = Task("OCR", "vision", True, "ocr", "<OCR>")

# ── Vision tasks (Florence-only) ──────────────────────────────────
TASK_PG_ANALYSE = Task(
    "PromptGen Analyse", "vision", True, "prompt_gen_analyze", "<ANALYZE>", "Florence"
)
TASK_PG_MIXED = Task(
    "PromptGen Mixed Caption",
    "vision",
    True,
    "prompt_gen_mixed_caption",
    "<MIXED_CAPTION>",
    "Florence",
)
TASK_PG_MIXED_PLUS = Task(
    "PromptGen Mixed Caption Plus",
    "vision",
    True,
    "prompt_gen_mixed_caption_plus",
    "<MIXED_CAPTION_PLUS>",
    "Florence",
)

# ── Text tasks ────────────────────────────────────────────────────
TASK_EXPAND = Task("Expand Text", "text", False)
TASK_REFINE_EXPAND = Task("Refine & Expand Prompt", "text", False)
TASK_REWRITE_STYLE = Task("Rewrite Style", "text", False)
TASK_TAGS_TO_NL = Task("Tags to Natural Language", "text", False)
TASK_NL_TO_TAGS = Task("Natural Language to Tags", "text", False)
TASK_TRANSLATE = Task("Translate to English", "text", False)
TASK_SHORT_STORY = Task("Short Story", "text", False)
TASK_SONG_LYRICS = Task("Song Lyrics", "text", False)
TASK_MINIMAX_MUSIC3 = Task("MiniMax Music 3", "text", False)
TASK_YUE2 = Task("YuE2 Music", "text", False)
TASK_SUMMARIZE = Task("Summarize", "text", False)
TASK_PROMPT_VARIATIONS = Task("Prompt Variations", "text", False)

# ── Detection tasks (Florence-only, used by detection node) ───────
# Single source of truth for all tasks. Main node excludes category="detection".
TASK_DET_PHRASE_GROUND = Task(
    "Caption to Phrase Grounding",
    "detection",
    True,
    "caption_to_phrase_grounding",
    "<CAPTION_TO_PHRASE_GROUNDING>",
)
TASK_DET_REGION_CAP = Task("Region Caption", "detection", True, "od", "<OD>")
TASK_DET_DENSE_CAP = Task(
    "Dense Region Caption",
    "detection",
    True,
    "dense_region_caption",
    "<DENSE_REGION_CAPTION>",
    "Florence",
)
TASK_DET_REGION_PROP = Task(
    "Region Proposal",
    "detection",
    True,
    "region_proposal",
    "<REGION_PROPOSAL>",
    "Florence",
)
TASK_DET_REF_EXPR_SEG = Task(
    "Referring Expression Segmentation",
    "detection",
    True,
    "referring_expression_segmentation",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "Florence",
)
TASK_DET_OCR_REGION = Task(
    "OCR With Region",
    "detection",
    True,
    "ocr_with_region",
    "<OCR_WITH_REGION>",
    "Florence",
)
TASK_DET_DOCVQA = Task("DocVQA", "detection", True, "docvqa", "<DocVQA>", "Florence")


# ── Master list ───────────────────────────────────────────────────

ALL_TASKS: tuple[Task, ...] = (
    TASK_DIRECT_CHAT,
    TASK_QUESTION_ANSWER,
    TASK_CUSTOM_INSTRUCT,
    TASK_WAN_SCENE_5S,
    TASK_WAN_TIMELINE_5S,
    TASK_WAN_TIMELINE_5S_2,
    TASK_WAN_TIMELINE_5S_3,
    TASK_WAN_SCENE_10S,
    TASK_WAN_TIMELINE_10S,
    TASK_WAN_SCENE_20S,
    TASK_WAN_TIMELINE_20S,
    TASK_WAN_CN_ATOMIC,
    TASK_LTX_23_I2V,
    TASK_MINIMAX_H3_SCENE,
    TASK_MINIMAX_H3_T2VA_TIMELINE,
    TASK_MINIMAX_H3_I2VA_TIMELINE,
    TASK_MINIMAX_H3_FL2VA_TIMELINE,
    TASK_MINIMAX_H3_L2VA_TIMELINE,
    TASK_SIMPLE_DESC,
    TASK_DETAILED_DESC,
    TASK_ULTRA_DESC,
    TASK_CINEMATIC_DESC,
    TASK_IMAGE_ANALYSIS,
    TASK_DETAILED_ANALYSIS,
    TASK_TAGS,
    TASK_VIDEO_SUMMARY,
    TASK_OCR,
    TASK_PG_ANALYSE,
    TASK_PG_MIXED,
    TASK_PG_MIXED_PLUS,
    TASK_EXPAND,
    TASK_REFINE_EXPAND,
    TASK_REWRITE_STYLE,
    TASK_TAGS_TO_NL,
    TASK_NL_TO_TAGS,
    TASK_TRANSLATE,
    TASK_SHORT_STORY,
    TASK_SONG_LYRICS,
    TASK_MINIMAX_MUSIC3,
    TASK_YUE2,
    TASK_SUMMARIZE,
    TASK_PROMPT_VARIATIONS,
    TASK_DET_PHRASE_GROUND,
    TASK_DET_REGION_CAP,
    TASK_DET_DENSE_CAP,
    TASK_DET_REGION_PROP,
    TASK_DET_REF_EXPR_SEG,
    TASK_DET_OCR_REGION,
    TASK_DET_DOCVQA,
)


# ============================================================================
# Lookup Dicts
# ============================================================================

TASK_BY_NAME: Dict[str, Task] = {t.name: t for t in ALL_TASKS}
FLORENCE_ID_TO_TASK: Dict[str, Task] = {
    t.florence_id: t for t in ALL_TASKS if t.florence_id
}


# ============================================================================
# Task Filtering
# ============================================================================


def get_task_names(
    has_vision: bool = True,
    family: str = "",
    with_separators: bool = False,
    include_all_families: bool = False,
) -> list[str]:
    # Return filtered task names for the main node dropdown.
    # Excludes detection tasks — those are for the detection node only.
    # When with_separators=True, insert category separator tokens between groups.
    # When include_all_families=True, skip family_filter checks (for schema validation).
    _SEP = {
        "vision": "__SEP__VISION__",
        "text": "__SEP__TEXT__",
    }
    buckets: dict[str, list[str]] = {"custom": [], "vision": [], "text": []}
    for t in ALL_TASKS:
        if t.category == "detection":
            continue
        if t.needs_image and not has_vision:
            continue
        if not include_all_families:
            if t.family_filter and t.family_filter != family:
                continue
            # Florence requires a florence_id for every task — skip tasks without one
            if family == "Florence" and not t.florence_id:
                continue
        buckets.setdefault(t.category, []).append(t.name)
    if not with_separators:
        result: list[str] = []
        for cat in ("custom", "vision", "text"):
            result.extend(buckets.get(cat, []))
        return result
    result = []
    for cat in ("custom", "vision", "text"):
        items = buckets.get(cat, [])
        if items and cat in _SEP:
            result.append(_SEP[cat])
        result.extend(items)
    return result


def get_detection_task_names() -> list[str]:
    # Return detection task names for the detection node dropdown.
    return [t.name for t in ALL_TASKS if t.category == "detection"]


def get_florence_token(task_name: str) -> Optional[str]:
    # Map display name → Florence prompt token (e.g. "<CAPTION>").
    task = TASK_BY_NAME.get(task_name)
    return task.florence_token if task else None


def is_florence_task(task_name: str) -> bool:
    # Check if a task has a Florence protocol mapping.
    task = TASK_BY_NAME.get(task_name)
    return task is not None and task.florence_id is not None


def get_h3_mode(task_name: str) -> H3Mode | None:
    # Return centralized MiniMax H3 input/timing metadata, if applicable.
    return H3_MODE_METADATA.get(canonicalize_task_name(task_name))


def resolve_h3_input_mode(task_name: str, image_count: int) -> str | None:
    # Validate an H3 task's reference count and resolve Scene's inferred mode.
    mode = get_h3_mode(task_name)
    return mode.resolve_input_mode(image_count) if mode else None


# ============================================================================
# System Prompts (user-editable, loaded from config/system_prompts.json)
# ============================================================================

_EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SYSTEM_PROMPTS_PATH = os.path.join(_EXTENSION_ROOT, "config", "system_prompts.json")

_system_prompts_cache: Optional[Dict[str, str]] = None
_system_prompts_mtime: float = 0.0


def _load_system_prompts() -> Dict[str, str]:
    # Load system prompts from config/system_prompts.json.
    # Returns cached result if file hasn't changed.
    global _system_prompts_cache, _system_prompts_mtime
    try:
        mtime = os.path.getmtime(_SYSTEM_PROMPTS_PATH)
    except OSError:
        log.warning(_LOG_PREFIX, "system_prompts.json not found — using empty prompts")
        return {}

    if _system_prompts_cache is not None and mtime == _system_prompts_mtime:
        return _system_prompts_cache

    try:
        with open(_SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _system_prompts_cache = data
        _system_prompts_mtime = mtime
        log.debug(_LOG_PREFIX, f"Loaded {len(data)} system prompts")
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.error(_LOG_PREFIX, f"Failed to load system_prompts.json: {e}")
        return {}


def get_system_prompt(task_name: str) -> str:
    # Get the system prompt for a task.
    # If a per-execution override is active (set via push_system_prompt_override),
    # return the override instead. Falls back to empty string if not found.
    task_name = canonicalize_task_name(task_name)
    prompt = _system_prompt_override.get()
    if not prompt:
        prompts = _load_system_prompts()
        prompt = prompts.get(task_name, "")
        if not prompt:
            prompt = next(
                (
                    prompts[legacy_name]
                    for legacy_name, canonical_name in LEGACY_TASK_ALIASES.items()
                    if canonical_name == task_name and prompts.get(legacy_name)
                ),
                "",
            )
    if task_name in H3_MODE_METADATA and _H3_MUSIC_INTENT_POLICY not in prompt:
        prompt = (
            f"{prompt.rstrip()}\n\n{_H3_MUSIC_INTENT_POLICY}"
            if prompt
            else _H3_MUSIC_INTENT_POLICY
        )
    return prompt


def get_h3_story_system_prompt(task_name: str) -> str:
    # Add the creative blank-input branch without changing ordinary H3 prompting.
    if not is_h3_scene_task(task_name):
        return get_system_prompt(task_name)
    base = get_system_prompt(task_name).rstrip()
    return (
        f"{base}\n\n{_H3_EMPTY_SCENE_STORY_POLICY}"
        if base
        else _H3_EMPTY_SCENE_STORY_POLICY
    )


def get_all_system_prompts() -> Dict[str, str]:
    # Get all system prompts (for endpoint serialization).
    return dict(_load_system_prompts())


# ============================================================================
# System Prompt Override (ContextVar)
# ============================================================================
#
# Allows the Smart LM Loader node to inject a custom system prompt for the
# duration of a single generation call without modifying any backend code.
# All 7 backends call get_system_prompt() — when an override is active, that
# function returns the override transparently.
#
# Caller contract (try/finally is mandatory):
#     token = push_system_prompt_override(value)
#     try:
#         ...do work that calls get_system_prompt()...
#     finally:
#         reset_system_prompt_override(token)

_system_prompt_override: ContextVar[Optional[str]] = ContextVar(
    "_eclipse_system_prompt_override", default=None
)


def push_system_prompt_override(value: Optional[str]):
    # Push a system-prompt override. Empty/whitespace strings are normalized to None
    # (no-op). Returns a token that MUST be passed to reset_system_prompt_override().
    normalized = value if (value and value.strip()) else None
    return _system_prompt_override.set(normalized)


def reset_system_prompt_override(token) -> None:
    # Reset the system-prompt override to its previous value. Always call from
    # a `finally` block paired with push_system_prompt_override().
    _system_prompt_override.reset(token)
