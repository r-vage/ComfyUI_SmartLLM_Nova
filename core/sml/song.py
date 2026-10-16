"""Provider-neutral lyric sheets, protected JSON and execution-local song state."""

import json
import re
from contextvars import ContextVar

from .tasks import (
    get_system_prompt,
    push_system_prompt_override,
    reset_system_prompt_override,
)

active_source: ContextVar = ContextVar("smartllm_song_source", default=None)

_SECTIONS = (
    "Final Chorus", "Pre-Chorus", "Guitar Solo", "Instrumental", "Breakdown",
    "Interlude", "Refrain", "Chorus", "Verse", "Bridge", "Intro", "Outro", "Hook", "Solo",
)
_SECTION = "(?:" + "|".join(name.replace("-", "[- ]") for name in _SECTIONS) + ")"
_LABEL = re.compile(rf"^({_SECTION})(\s*\d+)?(.*)$", re.IGNORECASE)
_COUNT = r"(once|twice|three times|[1-9]\d?\s*(?:times|x)?)"
_REPEAT = re.compile(rf"^(?:repeat\s+{_COUNT}|x\s*([1-9]\d?))$", re.IGNORECASE)
_PRODUCTION = re.compile(
    r"\b(?:BPM|tuning|tempo|meter|instrumental|no vocals|no singing)\b|\b\d+/\d+\b|"
    r"^(?:Title|Style|Genre|Key|Production|Vocal Details|Arrangement):", re.IGNORECASE,
)


def _source_error(detail: str, task_name: str) -> ValueError:
    return ValueError(f"{task_name} source parsing error: {detail}")


def _label(line: str, task_name: str):
    """Return a canonical section, performance hint, and explicit repeat count."""
    text = line.strip()
    if text.startswith(("**", "__")) and text.endswith(text[:2]):
        text = text[2:-2].strip()
    wrapped = text.startswith(("[", "("))
    if wrapped:
        if not text.endswith("]" if text[0] == "[" else ")"):
            raise _source_error("unclosed section label or standalone direction.", task_name)
        text = text[1:-1].strip()
    else:
        text = text.removesuffix(":")
    leading_repeat = text.lower().startswith("repeat ") and (
        wrapped or re.match(rf"repeat\s+(?:the\s+)?{_SECTION}\b", text, re.IGNORECASE)
    )
    if leading_repeat:
        instruction = re.fullmatch(rf"repeat\s+({_SECTION}(?:\s*\d+)?)\s+{_COUNT}", text, re.IGNORECASE)
        if instruction:
            text = f"{instruction[1]} repeat {instruction[2]}"
            leading_repeat = False
        else:
            text = text[7:]
    match = _LABEL.fullmatch(text)
    if not match:
        if wrapped or leading_repeat:
            raise _source_error("unrecognized section or ambiguous standalone direction.", task_name)
        return None
    name, number, suffix = match.groups()
    name = next(n for n in _SECTIONS if n.lower().replace("-", " ") == name.lower().replace("-", " "))
    name += f" {number.strip()}" if number else ""
    suffix = suffix.strip()
    if leading_repeat:
        suffix = "repeat " + suffix
    repeat = _REPEAT.fullmatch(suffix)
    if repeat:
        count = repeat[1] or repeat[2]
        count = {"once": 1, "twice": 2, "three times": 3}.get(count.lower(), count)
        if isinstance(count, str):
            count = int(re.match(r"\d+", count)[0])
        return name, "", count
    if not suffix:
        return name, "", None
    if wrapped and suffix.startswith(("-", "—", "–", ":")):
        hint = suffix[1:].strip()
        if not hint or re.search(r"\brepeat\b|\bx\s*\d", hint, re.IGNORECASE):
            raise _source_error("ambiguous repeat instruction in section hint.", task_name)
        return name, hint, None
    if wrapped or leading_repeat:
        raise _source_error("ambiguous section label or repeat instruction.", task_name)
    return None


def parse_sheet(source: str, task_name: str) -> str:
    blocks = []
    references = {}
    header = []
    current = None
    body = []
    ended = False

    def finish():
        nonlocal current, body
        if current is None:
            return
        # Normalize section spacing only; never strip or rewrite sung lines.
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body and not re.match(r"(?:Intro|Outro|Instrumental|.*Solo|Interlude)\b", current):
            raise _source_error("section has no lyrics; supply words or an explicit counted repeat.", task_name)
        block = f"[{current}]" + ("\n" + "\n".join(body) if body else "")
        blocks.append(block)
        references.setdefault(current, set()).add(block)
        current, body = None, []

    for line in source.splitlines():
        stripped = line.strip()
        if ended:
            if stripped:
                raise _source_error("Structure summary must be the final line.", task_name)
            continue
        if stripped.lower().startswith("structure:"):
            finish()
            ended = True
            continue
        label = _label(line, task_name) if stripped else None
        if label:
            finish()
            name, _hint, count = label
            if count is not None:
                candidates = references.get(name, set())
                if len(candidates) != 1:
                    raise _source_error("repeat must name one previously defined section with identical words.", task_name)
                blocks.extend([next(iter(candidates))] * count)
            else:
                current = name
            continue
        if re.match(
            rf"(?:repeat\s+(?:the\s+)?{_SECTION}\b|repeat\s*$|again\s*$|x\s*\d|\.{{3}}$|…$)",
            stripped, re.IGNORECASE,
        ):
            raise _source_error("ambiguous repeat instruction; supply a section and explicit count.", task_name)
        if current is not None:
            body.append(line)
        elif blocks and stripped:
            raise _source_error("text after a repeat has no section.", task_name)
        elif stripped:
            header.append(line)
    finish()
    if not blocks:
        raise _source_error("expected usable lyric sections or caption/lyrics JSON after Song Lyrics.", task_name)
    # One title and production metadata may precede sections. Never discard prose.
    if sum(not bool(_PRODUCTION.search(line)) for line in header) > 1:
        raise _source_error("unrecognized text before lyric sections; use a title and production header.", task_name)
    lyrics = "\n\n".join(blocks)
    if not re.sub(r"(?m)^\[[^\]\n]+\]$", "", lyrics).strip():
        if any(re.search(r"\binstrumental\b|\bno (?:vocals|singing)\b", line, re.IGNORECASE) for line in header) or all(
            block == "[Instrumental]" for block in blocks
        ):
            return ""
        raise _source_error("empty sheet must explicitly identify instrumental music.", task_name)
    return lyrics


def resolve_sheet(source: str, previous_task: str | None, task_name: str) -> str | None:
    """Return retained lyrics, or None for a concept that needs composition."""
    recognizable = re.search(
        rf"(?im)^\s*(?:\*\*|__)?[\[(]\s*(?:Repeat\s+)?{_SECTION}\b|"
        rf"^\s*(?:\*\*|__)?{_SECTION}(?:\s*\d+)?\s*(?:[:—–-]|(?:\*\*|__)?\s*$|repeat\b|x\s*\d)|"
        rf"^\s*Repeat\s+{_SECTION}\b|^\s*Structure:",
        source,
    )
    if previous_task == "Song Lyrics" or recognizable:
        return parse_sheet(source, task_name)
    return None


def unfence_song_json(text: str) -> str:
    """Accept only a single enclosing JSON/plain code fence."""
    candidate = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, re.DOTALL | re.IGNORECASE)
    return match[1].strip() if match else candidate


def protected_song_json(text: str) -> str | None:
    """Protect structured song values from prose cleanup, even if fields are invalid."""
    if not isinstance(text, str):
        return None
    # Remove outer reasoning/chat wrappers only, never tags inside JSON strings.
    candidate = text.strip()
    candidate = re.sub(r"^(?:<\|[^|]+\|>\s*)+", "", candidate)
    candidate = re.sub(
        r"^(?:(?:<(think|thinking|reasoning|summary)>.*?</\1>|"
        r"\[(THINK|THINKING|REASONING|SUMMARY)\].*?\[/\2\])\s*)+",
        "", candidate, flags=re.DOTALL | re.IGNORECASE,
    )
    candidate = re.sub(r"(?:\s*<\|[^|]+\|>)+$", "", candidate)
    candidate = unfence_song_json(candidate)

    # Models may apologize or introduce JSON even after a corrective retry.
    # Decode one complete object rather than matching braces inside lyric strings.
    # Refuse arrays, nested wrappers, multiple objects and malformed leading JSON;
    # never skip ahead to a later object that happens to satisfy the contract.
    opening = re.search(r"[{}\[\]]", candidate)
    if opening is None or opening[0] != "{":
        return None
    start = opening.start()
    try:
        value, end = json.JSONDecoder().raw_decode(candidate, start)
    except (ValueError, TypeError):
        return None
    if re.search(r"[{}\[\]]", candidate[end:]):
        return None
    if isinstance(value, dict) and ("caption" in value or "style" in value or "lyrics" in value):
        # Keep the original bytes, including duplicate fields: the strict parser
        # must still reject them, and decoded caption/lyrics must stay unchanged.
        return candidate[start:end]
    return None


def generate_song(task_name, source, generate, logger):
    """Retain one resolved mode through a retry and always restore caller state."""
    logger.info("Smart LM Loader", f"{task_name} mode: {source.mode}")  # noqa: PLE1205 - SmartLLM prefix/message API
    base = get_system_prompt(source.prompt_key)
    instruction = f"{base}\n\n{source.instruction()}"
    source_token = active_source.set(source)
    token = push_system_prompt_override(instruction)
    try:
        result, data = generate()
        try:
            return json.dumps(source.assemble(result), ensure_ascii=False), data
        except ValueError as error:
            issue = str(error)
        correction = push_system_prompt_override(
            f"{instruction}\n\nCorrect the output format: {issue} {source.instruction()}"
        )
        try:
            result, data = generate()
            try:
                return json.dumps(source.assemble(result), ensure_ascii=False), data
            except ValueError as error:
                raise ValueError(f"{task_name} output invalid after corrective retry: {error}") from None
        finally:
            reset_system_prompt_override(correction)
    finally:
        reset_system_prompt_override(token)
        active_source.reset(source_token)
