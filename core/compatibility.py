from pathlib import Path
from typing import Any

_SMART_LM_ROUTE_PREFIX = "/smartllm-nova/"


def _registered_smart_lm_routes() -> list[str]:
    try:
        from server import PromptServer  # type: ignore

        route_table: Any = PromptServer.instance.routes
        items = getattr(route_table, "_items", ())
        paths = []
        for item in items:
            path = getattr(item, "path", "")
            if isinstance(path, str) and path.startswith(_SMART_LM_ROUTE_PREFIX):
                method = getattr(item, "method", "*")
                paths.append(f"{method} {path}")
        return sorted(set(paths))
    except (AttributeError, ImportError, RuntimeError):
        return []


def find_conflicts(repo_root: Path) -> list[str]:
    # Nova can coexist with the original pack; only its own routes conflict.
    conflicts: list[str] = []
    registered = _registered_smart_lm_routes()
    if registered:
        conflicts.append(
            "Smart LM routes are already registered: " + ", ".join(registered)
        )
    return conflicts
