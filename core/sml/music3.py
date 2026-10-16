"""MiniMax Music 3's text interchange contract (no model dependencies)."""

import json
import re
from dataclasses import dataclass

# Compatibility exports for existing Music 3 callers.
from .song import active_source, protected_song_json, resolve_sheet, unfence_song_json

__all__ = [
    "CONVERT_PROMPT", "CONVERT_TRAINING", "TASK_NAME", "Music3Source",
    "active_source", "parse_generated_song", "parse_song_json",
    "protected_song_json", "resolve_source", "unfence_song_json",
]

TASK_NAME = "MiniMax Music 3"
_HEADINGS = ("Global Metadata", "Vocal Details", "Arrangement")
_HEADING = re.compile(r"^(Global Metadata|Vocal Details|Arrangement):", re.MULTILINE)

CONVERT_PROMPT = "MiniMax Music 3 Conversion"
CONVERT_TRAINING = "minimax_music_3_convert"


@dataclass(frozen=True)
class Music3Source:
    mode: str
    lyrics: str | None = None

    @property
    def prompt_key(self) -> str:
        return CONVERT_PROMPT if self.mode == "convert" else TASK_NAME

    @property
    def training_key(self) -> str:
        return CONVERT_TRAINING if self.mode == "convert" else "minimax_music_3"

    def instruction(self) -> str:
        if self.mode == "convert":
            return (
                'CONVERSION MODE: Return only JSON with exactly one string field, "caption". '
                "Use the entire source sheet for tempo, genre, voices and section performance hints. "
                "Lyrics are retained by the application; do not write, translate or replace them. "
                "Begin caption with Global Metadata:, then Vocal Details: and Arrangement: "
                "on separate paragraphs in that order. "
                + ("The source is instrumental: begin Vocal Details with 'Instrumental; no vocals.' "
                   if self.lyrics == "" else "Describe vocals matching the retained lyrics. ")
                + "No commentary or code fences."
            )
        return (
            "COMPOSITION MODE: Write a complete original song from the concept, story or request. "
            'Return only JSON with exactly two string fields, "caption" and "lyrics". '
            "Only an explicit instrumental request permits empty lyrics."
        )

    def assemble(self, output: str) -> dict[str, str]:
        if self.mode == "compose":
            return parse_generated_song(output)
        try:
            value = json.loads(protected_song_json(output) or unfence_song_json(output), object_pairs_hook=_unique_fields)
        except json.JSONDecodeError:
            raise ValueError("MiniMax Music 3 conversion requires caption JSON.") from None
        if not isinstance(value, dict) or set(value) != {"caption"} or not isinstance(value["caption"], str):
            raise ValueError("MiniMax Music 3 conversion requires exactly one caption string; omit lyrics.")
        return parse_generated_song(json.dumps({"caption": value["caption"], "lyrics": self.lyrics}, ensure_ascii=False))


def resolve_source(source: str, previous_task: str | None = None) -> Music3Source:
    """Resolve once, before generation. Recognizable invalid sources fail closed."""
    candidate = unfence_song_json(source)
    if candidate.startswith("{") or candidate.lower().startswith("```json") or re.search(r'"(?:lyrics|caption)"\s*:', candidate):
        song = parse_song_json(candidate)
        return Music3Source("convert", song["lyrics"])
    lyrics = resolve_sheet(source, previous_task, TASK_NAME)
    return Music3Source("convert", lyrics) if lyrics is not None else Music3Source("compose")


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("MiniMax Music 3 JSON must not contain duplicate fields.")
        result[key] = value
    return result


def parse_song_json(text: str) -> dict[str, str]:
    """Validate without including user content in error messages."""
    if not isinstance(text, str):
        raise ValueError("MiniMax Music 3 requires a JSON string.")  # noqa: TRY004 - one validation error API
    try:
        song = json.loads(unfence_song_json(text), object_pairs_hook=_unique_fields)
    except json.JSONDecodeError:
        raise ValueError("MiniMax Music 3 requires valid JSON with caption and lyrics strings.") from None
    if not isinstance(song, dict) or set(song) != {"caption", "lyrics"}:
        raise ValueError("MiniMax Music 3 requires exactly caption and lyrics fields.")
    if not all(isinstance(value, str) for value in song.values()):
        raise ValueError("MiniMax Music 3 caption and lyrics must both be strings.")
    caption = song["caption"]
    if not caption.strip():
        raise ValueError("MiniMax Music 3 caption must not be empty.")
    matches = list(_HEADING.finditer(caption))
    if tuple(match[1] for match in matches) != _HEADINGS or matches[0].start() != 0:
        raise ValueError(
            "MiniMax Music 3 caption requires ordered headings: "
            "Global Metadata:, Vocal Details:, Arrangement:."
        )
    return song


def parse_generated_song(text: str) -> dict[str, str]:
    """Reject missing sung words unless the vocal direction is instrumental."""
    song = parse_song_json(protected_song_json(text) or text)
    words = re.sub(r"(?m)^\s*\[[^\]\n]+\]\s*$", "", song["lyrics"]).strip()
    if not words:
        headings = list(_HEADING.finditer(song["caption"]))
        vocals = song["caption"][headings[1].end():headings[2].start()].strip()
        if not re.match(r"(?:instrumental|no vocals?|no singing)\b", vocals, re.IGNORECASE):
            raise ValueError(
                "MiniMax Music 3 vocal song is missing lyrics. Write complete sung "
                "lyrics from the user's style and subject, with verses and a chorus. "
                "Empty lyrics are reserved for explicitly instrumental output."
            )
    return song
