"""Bounded GGUF header inspection, without mapping weights or decoding vocabularies."""

import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
            7: "?", 10: "Q", 11: "q", 12: "d"}
_HEADER_LIMIT = 128 * 1024 * 1024


@dataclass(frozen=True)
class GGUFHeader:
    metadata: dict
    shapes: dict[str, tuple[int, ...]]
    arrays: dict[str, int]


class _HeaderReader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream = stream
        self.limit = min(size, _HEADER_LIMIT)
        self.endian = "<"

    def read(self, count: int) -> bytes:
        if count < 0 or self.stream.tell() + count > self.limit:
            raise ValueError("GGUF header exceeds the file or inspection limit.")
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("Incomplete GGUF header.")
        return data

    def number(self, fmt: str) -> int | float | bool:
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def skip(self, count: int) -> None:
        if count < 0 or self.stream.tell() + count > self.limit:
            raise ValueError("Invalid GGUF metadata length.")
        self.stream.seek(count, 1)

    def string(self, *, keep: bool = True) -> str | None:
        length = self.number("Q")
        if length > 16 * 1024 * 1024:
            raise ValueError("GGUF string exceeds the inspection limit.")
        if keep:
            return self.read(length).decode("utf-8")
        self.skip(length)
        return None

    def value(self, kind: int, *, keep: bool = True) -> object:
        if kind in _SCALARS:
            return self.number(_SCALARS[kind])
        if kind == 8:
            return self.string(keep=keep)
        raise ValueError(f"Unsupported GGUF metadata type: {kind}")


@lru_cache(maxsize=256)
def read_gguf_header(signature: tuple) -> GGUFHeader:
    """Stat-keyed snapshot. Callers must not mutate the cached dictionaries."""
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    path = Path(signature[0])
    size = path.stat().st_size
    metadata, arrays, shapes, spans = {}, {}, {}, []
    with path.open("rb") as stream:
        reader = _HeaderReader(stream, size)
        if reader.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file.")
        version = reader.read(4)
        if struct.unpack("<I", version)[0] not in (2, 3):
            if struct.unpack(">I", version)[0] != 3:
                raise ValueError("Unsupported GGUF version.")
            reader.endian = ">"
        tensors, fields = reader.number("Q"), reader.number("Q")
        if tensors > 100_000 or fields > 100_000:
            raise ValueError("GGUF inventory exceeds the inspection limit.")
        for _ in range(fields):
            key, kind = reader.string(), reader.number("I")
            if key in metadata or key in arrays:
                raise ValueError("Duplicate GGUF metadata key.")
            if kind != 9:
                metadata[key] = reader.value(kind)
                continue
            subtype, count = reader.number("I"), reader.number("Q")
            if count > 2_000_000:
                raise ValueError("GGUF array exceeds the inspection limit.")
            arrays[key] = count
            if key.startswith("comfy.gguf.orig_shape."):
                if subtype != 5 or not 1 <= count <= 4:
                    raise ValueError("Invalid original GGUF tensor shape.")
                metadata[key] = tuple(reader.number("i") for _ in range(count))
            elif subtype in _SCALARS:
                reader.skip(count * struct.calcsize(_SCALARS[subtype]))
            elif subtype == 8:
                for _ in range(count):
                    reader.string(keep=False)
            else:
                raise ValueError("Unsupported GGUF array type.")
        for _ in range(tensors):
            name, dimensions = reader.string(), reader.number("I")
            if name in shapes or not 1 <= dimensions <= 4:
                raise ValueError("Invalid GGUF tensor inventory.")
            shape = tuple(reversed([reader.number("Q") for _ in range(dimensions)]))
            qtype, offset = reader.number("I"), reader.number("Q")
            block, byte_size = GGML_QUANT_SIZES[GGMLQuantizationType(qtype)]
            count = math.prod(shape)
            if count <= 0 or shape[-1] % block:
                raise ValueError("Invalid GGUF tensor dimensions.")
            original = metadata.get("comfy.gguf.orig_shape." + name, shape)
            if any(n <= 0 for n in original) or math.prod(original) != count:
                raise ValueError("Invalid original GGUF tensor dimensions.")
            shapes[name] = original
            spans.append((offset, count // block * byte_size))
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF alignment.")
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        previous_end = 0
        for offset, length in sorted(spans):
            if offset % alignment or offset < previous_end or data_start + offset + length > size:
                raise ValueError("Incomplete or overlapping GGUF tensor data.")
            previous_end = offset + length
    return GGUFHeader(metadata, shapes, arrays)
