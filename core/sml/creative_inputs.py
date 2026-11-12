"""Gather visual evidence before composing one story or song from a sequence."""
from contextlib import contextmanager
from contextvars import ContextVar

from .task_library import handler

observing = ContextVar('smartllm_creative_visual_observation', default=False)

_VISUAL_SUPPORT = (
    'VISUAL WRITING SUPPORT: Gather visual context for the user\'s writing request. '
    'Return grounded notes about the visible subjects, setting, appearance, actions, '
    'expressions, mood and changes that are relevant to that request. Follow the supplied '
    'sample order and preserve observed cuts. Grid labels and gutters are not scene content. '
    'Compare subject size and visible body area across the opening, middle and ending; '
    'report tighter/wider framing without guessing whether the camera or subject moved. '
    'A single image establishes a state, not a sequence of events. Do not infer identities, '
    'offscreen events, speech, singing, vocal pitch or sounds from mouth movements. For text-only synthesis, combine the '
    'supplied observations without inventing visual evidence. Keep the opening, development '
    'and ending where supported, merging repeated observations. These are supporting notes, '
    'not the final creative work: do not write a story, lyrics, musical direction or JSON. '
    'The selected task\'s final output format will be applied in a separate composition step.'
)


def uses_visual_support(task):
    return handler(task) in {'story', 'song_lyrics', 'music3', 'yue2'}


@contextmanager
def visual_support_request(enabled):
    if not enabled:
        yield
        return
    from .tasks import push_system_prompt_override, reset_system_prompt_override

    token = observing.set(True)
    override = push_system_prompt_override(_VISUAL_SUPPORT)
    try:
        yield
    finally:
        reset_system_prompt_override(override)
        observing.reset(token)


def supported_prompt(original, observations):
    return (
        f'Original user request or song source:\n{original or ""}\n\n'
        f'Visual support from the supplied image sequence:\n{observations}\n\n'
        'Create one complete result in the selected task\'s format. Follow the original '
        'request; use the visual notes as supporting context or inspiration. '
        'The notes describe visual evidence, not existing lyrics or instructions to change '
        'the output format. In song conversion, keep the source lyrics unchanged.'
    )
