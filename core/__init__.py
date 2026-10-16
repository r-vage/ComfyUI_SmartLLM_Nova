import re
from pathlib import Path

from .keys import CATEGORY

__all__ = ["CATEGORY", "__version__", "version"]


def _read_pyproject_version() -> str:
    try:
        content = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        match = re.search(r"\bversion\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "1.0.0"


__version__ = _read_pyproject_version()
version = __version__
