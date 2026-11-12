"""Task catalog compatibility helpers and protected H3 runtime contracts."""
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace

from .task_library import get_task, handler, load_catalog, prompt_config
from .task_operations import FLORENCE_OPERATIONS


class _TaskMap(Mapping):
    def __getitem__(self, key):
        task = get_task(key)
        if task is None:
            raise KeyError(key)
        return task

    def __iter__(self):
        return iter(task.name for task in load_catalog().tasks.values() if not task.internal)

    def __len__(self):
        return sum(not task.internal for task in load_catalog().tasks.values())


class _Tasks(Sequence):
    def __getitem__(self, index):
        return tuple(task for task in load_catalog().tasks.values() if not task.internal)[index]

    def __len__(self):
        return len(TASK_BY_NAME)


TASK_BY_NAME = _TaskMap()
ALL_TASKS = _Tasks()
# Model protocol support survives user deletion/renaming of a library task.
FLORENCE_ID_TO_TASK = {key: SimpleNamespace(florence_token=value.token, florence_id=key)
                       for key, value in FLORENCE_OPERATIONS.items()}


def get_task_names(has_vision=True, family='', with_separators=False, include_all_families=False):
    result = []
    for group in ('vision', 'text'):
        tasks = [task for task in ALL_TASKS if 'loader' in task.nodes and task.enabled
                 and task.category == group and (has_vision or not task.needs_image)
                 and (include_all_families or ((not task.family_filter or task.family_filter == family)
                      and (family != 'Florence' or task.florence_id)))]
        if tasks and with_separators:
            result.append('__SEP__' + group.upper() + '__')
        result.extend(task.name for task in tasks)
    return result


def get_detection_task_names():
    return [task.name for task in ALL_TASKS if 'detection' in task.nodes and task.enabled]


def get_florence_token(task_name):
    task = get_task(task_name)
    return task.florence_token if task else None


def is_florence_task(task_name):
    task = get_task(task_name)
    return bool(task and task.florence_id)


def is_h3_scene_task(task_name):
    return handler(task_name) == 'h3_scene'


def get_h3_mode(task_name):
    spec = {'h3_scene': ('AUTO', (0, 1, 2), False), 'h3_t2va': ('T2VA', (0,), True),
            'h3_i2va': ('I2VA', (1,), True), 'h3_fl2va': ('FL2VA', (1, 2), True),
            'h3_l2va': ('L2VA', (1,), True)}.get(handler(task_name))
    if spec is None:
        return None
    task = get_task(task_name)
    return H3Mode(task.name, spec[0], None, spec[1], spec[2])


def get_system_prompt(task_name):
    prompt = _system_prompt_override.get()
    if prompt is None:
        prompt = prompt_config(task_name)['system_prompt']
    if get_h3_mode(task_name) is not None and _H3_MUSIC_INTENT_POLICY not in prompt:
        prompt = (prompt.rstrip() + '\n\n' + _H3_MUSIC_INTENT_POLICY).strip()
    return prompt


def get_all_system_prompts():
    return {task.name: get_system_prompt(task.id) for task in ALL_TASKS}


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

    def resolve_input_mode(self, image_count: int, *, frame_sequence: bool = False) -> str:
        # Together supplies scene observations, which can outnumber the target
        # video's reference images. T2VA remains a text-only input contract.
        if frame_sequence and image_count > 0 and self.input_mode != "T2VA":
            image_count = min(image_count, max(self.allowed_image_counts))
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
    "leave duration to the downstream generation settings; do not add seconds, "
    "timestamps, a total duration or extra shots. Always "
    "provide synchronized ambience and action sounds in overall_soundscape and a "
    "scene-appropriate background score in non_diegetic_music. "
    "non_diegetic_music must never be N/A in this mode. Return only the three required "
    "fields in their required order."
)


def resolve_h3_input_mode(
    task_name: str, image_count: int, *, frame_sequence: bool = False,
) -> str | None:
    # Validate an H3 task's reference count and resolve Scene's inferred mode.
    mode = get_h3_mode(task_name)
    return mode.resolve_input_mode(image_count, frame_sequence=frame_sequence) if mode else None


def get_h3_sequence_system_prompt(task_name: str, image_count: int, *, synthesis: bool = False) -> str:
    """Keep the selected H3 format while grounding it in an observed scene."""
    resolve_h3_input_mode(task_name, image_count, frame_sequence=True)
    base = get_system_prompt(task_name).rstrip()
    source = (
        'Use only the supplied window observations; no images are attached to this request. '
        'Combine their observed action, including intermediate frames, without inventing visual evidence. '
        if synthesis else
        'Inspect every supplied sample, including intermediate frames. Samples may be separate images '
        'or labelled tiles in a contact sheet; follow the supplied chronological order. '
    )
    context = (
        f"TOGETHER SCENE INPUT: The input represents {image_count} chronological samples. {source}"
        "SCENE INPUT is active: ignore reference-only image counts, endpoint interpolation, "
        "default shot counts and duration rules. "
        "Apply the user's guidance for focus, explicitly requested changes, camera behavior, "
        "dialogue and sound. User guidance may be in source_context or follow the input metadata. "
        "The first sample shows the starting state and the "
        "last sample shows the ending state. Describe the visible action, poses "
        "and camera changes as one connected scene in the selected H3 output "
        "format, retaining all three required fields. A single sample establishes "
        "appearance and pose only. Without an explicitly requested change, preserve the observed "
        "action and camera behavior; do not invent movement between uncertain samples, a new "
        "story or events before/after the observed sequence. Blank prompts describe the observed "
        "scene instead of enabling story mode. The downstream generation settings control duration; "
        "describe action order without adding seconds, timestamps, cut times or a total duration. "
        "Preserve cuts explicitly identified in the supplied evidence without inventing motion across them. "
        "For I2VA the starting reference is the first sample; for L2VA the ending "
        "reference is the last sample; for FL2VA the references are the first and "
        "last samples, with every intermediate sample providing motion evidence. "
        "Do not mistake the second sample for the final reference in a longer "
        "sequence. A blank user prompt means describe the observed scene. Images "
        "do not establish audio: use N/A for soundscape and music unless the user "
        "provides or requests them. Preserve the user's explicit instructions."
    )
    return f"{base}\n\n{context}" if base else context


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


_system_prompt_override: ContextVar[str | None] = ContextVar(
    "_eclipse_system_prompt_override", default=None
)


def push_system_prompt_override(value: str | None):
    # Push a system-prompt override. Empty/whitespace strings are normalized to None
    # (no-op). Returns a token that MUST be passed to reset_system_prompt_override().
    normalized = value if (value and value.strip()) else None
    return _system_prompt_override.set(normalized)


def reset_system_prompt_override(token) -> None:
    # Reset the system-prompt override to its previous value. Always call from
    # a `finally` block paired with push_system_prompt_override().
    _system_prompt_override.reset(token)
