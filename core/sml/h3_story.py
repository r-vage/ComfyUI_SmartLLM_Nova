"""Plan one story and write each H3 clip in order with its own output budget."""
import re

from .generation_control import check_cancelled
from .logger import log
from .tasks import push_system_prompt_override, reset_system_prompt_override


def story_visual(text):
    """Number shots locally even when the writer uses the clip number as a heading."""
    text = re.sub(r"^\s*(?:\*\*)?shot\s+\d+\s*[:.](?:\*\*)?\s*", "", text, flags=re.IGNORECASE)
    pieces = re.split(r"(?:\*\*)?\[\s*shot\s+\d+\s*\](?:\*\*)?", text, flags=re.IGNORECASE)
    shots = (piece.strip() for piece in pieces if piece.strip())
    return " ".join(f"[Shot {index}] {action}" for index, action in enumerate(shots, 1))


def _outline(brief, count, generate):
    length = (f"Write exactly {count} beats." if count else
              "Choose the fewest beats that tell the requested story clearly.")
    token = push_system_prompt_override(
        f"Plan a film from the user's brief. {length} Return only a numbered list, starting at 1, "
        "one short line per clip in the form '1. action'. Each line states the physical action that "
        "actually happens and where the subjects end up. Keep the starting place, intervening space "
        "and destination distinct, including unusual geography. Give each requested event its own place in "
        "the arc. End each beat where the next can begin. Reserve reaching the final destination or "
        "resolving the conflict for the last beat. Omit H3 fields and audio headings."
    )
    try:
        text, _ = generate(brief)
    finally:
        reset_system_prompt_override(token)
    beats = []
    for line in (text or "").splitlines():
        match = re.fullmatch(r"\s*(\d+)[.)]\s+(.+?)\s*", line)
        if match:
            if match[1] != str(len(beats) + 1):
                return []
            beats.append(match[2])
    return beats if not count or len(beats) == count else []


def _continuity(previous):
    visual, _, audio = previous.partition("\n\noverall_soundscape:")
    last_shot = re.split(r"\[Shot \d+\]", visual)[-1].strip()
    ending = " ".join(re.split(r"(?<=[.!?])\s+", last_shot)[-2:])
    return (f"Ending view already reached: {ending}\n"
            f"Prior audio (carry ongoing ambience and score; one-off sounds have already occurred):{audio}")


def write_story(brief, count, generate, render_clip):
    if not brief or not brief.strip():
        raise ValueError("MiniMax H3 Story needs a short story description.")
    if type(count) is not int or count < 0:
        raise ValueError("Story clips must be a nonnegative integer (0 = automatic).")
    beats = []
    if count != 1:
        beats = _outline(brief, count, generate)
        if not beats:
            log.warning("MiniMax H3 Story", "The outline did not supply a usable beat list. "
                        f"Writing {count or 1} clip(s) directly from the brief.", notify=True)
    total = count or len(beats) or 1
    clips = []
    for index in range(total):
        check_cancelled()
        request = [f"Story brief:\n{brief}"]
        if clips:
            request.append(_continuity(clips[-1]))
        if beats and index + 1 < total:
            request.append(f"Reserved for the NEXT clip: {beats[index + 1]}\n"
                           "Stop the current clip immediately before that event begins.")
        request.append(f"Write clip {index + 1} of {total}. Start from the current position and show the new action.")
        if beats:
            request.append(f"Required action to complete on screen in this clip: {beats[index]}")
        if index + 1 == total:
            request.append("Bring the story to its visible conclusion in this clip.")
        clips.append(render_clip("\n\n".join(request), beats[index] if beats else brief))
    return clips
