"""YuE2 text preparation; ABC planning belongs to the native YuE2 nodes."""

import json
import re
from dataclasses import dataclass

from .song import protected_song_json, resolve_sheet, unfence_song_json

TASK_NAME = "YuE2 Music"
_INSTRUMENTAL = re.compile(r"\binstrumental\b|\bno (?:vocals?|singing)\b", re.IGNORECASE)


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("YuE2 JSON must not contain duplicate fields.")
        result[key] = value
    return result


def _parse_fields(text, fields):
    if not isinstance(text, str):
        raise ValueError("YuE2 requires a JSON string.")  # noqa: TRY004 - validation API
    try:
        value = json.loads(unfence_song_json(text), object_pairs_hook=_unique_fields)
    except json.JSONDecodeError:
        raise ValueError("YuE2 requires valid song JSON.") from None
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"YuE2 requires exactly {' and '.join(fields)} fields.")
    if not all(isinstance(item, str) for item in value.values()):
        raise ValueError("YuE2 song fields must be strings.")
    if not value[fields[0]].strip():
        raise ValueError("YuE2 musical description must not be empty.")
    return value


def parse_song_json(text: str) -> dict[str, str]:
    """Strict splitter contract; preserve the decoded strings verbatim."""
    return _parse_fields(text, ("style", "lyrics"))


@dataclass(frozen=True)
class YuE2Source:
    mode: str
    lyrics: str | None = None
    instrumental: bool = False

    @property
    def prompt_key(self) -> str:
        # Keep user-customized prompt keys independent of the visible task name.
        return "YuE2 Conversion" if self.mode == "convert" else "YuE2"

    @property
    def training_key(self) -> str:
        return "yue2_convert" if self.mode == "convert" else "yue2"

    def instruction(self) -> str:
        if self.mode == "convert":
            return (
                'CONVERSION MODE: Return only JSON with exactly one string field, "style". '
                "Use the entire source for genre, instruments, language, tempo, vocals, "
                "arrangement and section performance hints. Lyrics are retained by the "
                "application; do not write, translate or replace them. "
                + ("Describe instrumental music with no vocals. " if self.lyrics == "" else
                   "Describe vocals matching the retained lyrics. ")
                + "Use free-form musical direction without mandatory headings. No ABC or code fences."
            )
        return (
            "COMPOSITION MODE: Write a complete original song from the concept or story. "
            'Return only JSON with exactly two string fields, "style" and "lyrics". '
            "Use Song Lyrics craft: structure, rhyme, repeated hooks and requested language. "
            + ('This is an explicit instrumental request: use empty lyrics and an instrumental, no vocals style. '
               if self.instrumental else "Write complete nonempty sung lyrics. ")
            + "Style is free-form musical direction. No mandatory headings, ABC or code fences."
        )

    def assemble(self, output: str) -> dict[str, str]:
        candidate = protected_song_json(output) or output
        if self.mode == "convert":
            style = _parse_fields(candidate, ("style",))["style"]
            song = {"style": style, "lyrics": self.lyrics}
        else:
            song = parse_song_json(candidate)
        words = re.sub(r"(?m)^\s*\[[^\]\n]+\]\s*$", "", song["lyrics"]).strip()
        if not words and not (self.instrumental and _INSTRUMENTAL.search(song["style"])):
            raise ValueError("YuE2 vocal song requires complete sung lyrics; empty lyrics require explicit instrumental direction.")
        return song


def resolve_source(source: str, previous_task: str | None = None) -> YuE2Source:
    candidate = unfence_song_json(source)
    if (candidate.startswith("{") or candidate.lower().startswith("```json")
            or re.search(r'"(?:lyrics|caption|style)"\s*:', candidate)):
        # MiniMax JSON is a conversion source, not a YuE2 splitter input.
        # Validate its interchange fields without requiring provider headings.
        try:
            song = parse_song_json(candidate)
        except ValueError:
            song = _parse_fields(candidate, ("caption", "lyrics"))
        direction = song.get("style", song.get("caption", ""))
        instrumental = bool(_INSTRUMENTAL.search(direction))
        if not song["lyrics"].strip() and not instrumental:
            raise ValueError("YuE2 empty source lyrics require explicit instrumental direction.")
        return YuE2Source("convert", song["lyrics"], instrumental)
    lyrics = resolve_sheet(source, previous_task, TASK_NAME)
    if lyrics is not None:
        return YuE2Source("convert", lyrics, lyrics == "")
    return YuE2Source("compose", instrumental=bool(_INSTRUMENTAL.search(source)))
