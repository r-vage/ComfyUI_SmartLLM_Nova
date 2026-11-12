"""File-backed task catalog, immutable execution snapshots and explicit edits."""
import copy
import hashlib
import json
import re
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from ..json_store import (
    locked_path,
    update_json_object,
    write_json_object,
)
from .task_operations import (
    HANDLERS,
    LEGACY_TASK_GROUPS,
    TASK_GROUPS,
    chain_issue,
    florence_operation,
    model_issue,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 256 * 1024
MAX_TASKS = 512
MAX_STEPS = 32
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_snapshot = ContextVar("smartllm_task_snapshot", default=None)
_cache_lock = threading.RLock()
_cache = {}


class TaskConflict(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("Task IDs use lowercase letters, digits, dots, underscores or hyphens (up to 96 characters).")
    return value


def _safe(path, root, *, exists=True):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Task folders must not be symbolic links.")
    if not path.absolute().is_relative_to(root.absolute()):
        raise ValueError("Task files must stay inside the task library.")
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError("Task files and folders must not be symbolic links.")
    if exists and not path.is_file():
        raise ValueError(f"Task file is missing: {path.name}")
    return path


def _read(path, root):
    _safe(path, root)
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"Task file exceeds 256 KiB: {path.name}")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")  # noqa: TRY004 -- public validation contract
    return value


def validate_task(value, *, legacy=False):
    keys = {"version", "id", "name", "description", "group", "nodes", "aliases", "input", "handler",
            "florence_operation", "instructions", "template", "internal", "enabled"}
    version = 1 if legacy else 2
    if legacy:
        keys |= {"profiles", "examples"}
    if not isinstance(value, dict) or set(value) != keys or type(value["version"]) is not int or value["version"] != version:
        raise ValueError(f"Expected a version {version} task definition with the documented fields.")
    _identifier(value["id"])
    for key in ("name", "description", "instructions", "template"):
        if not isinstance(value[key], str) or "\0" in value[key]:
            raise ValueError(f"Task {key} must be text without NUL characters.")
    if not value["name"].strip() or len(value["name"]) > 160:
        raise ValueError("Task name must contain 1–160 characters.")
    if (not isinstance(value['group'], str) or not isinstance(value['handler'], str)
            or value["group"] not in {*TASK_GROUPS, *LEGACY_TASK_GROUPS} or value["handler"] not in HANDLERS):
        raise ValueError("Unknown task group or operation.")
    for key in ("enabled", "internal"):
        if type(value[key]) is not bool:
            raise ValueError(f"Task {key} must be a Boolean.")
    if not isinstance(value["nodes"], list) or not value["nodes"] or any(not isinstance(v, str) or v not in {"loader", "detection"} for v in value["nodes"]):
        raise ValueError("Task nodes must contain loader and/or detection.")
    if not isinstance(value["aliases"], list) or len(value["aliases"]) > 32 or any(not isinstance(v, str) or not v or len(v) > 160 for v in value["aliases"]):
        raise ValueError("Invalid task aliases.")
    input_spec = value["input"]
    if (not isinstance(input_spec, dict) or set(input_spec) != {"text", "images", "sequence", "query"}
            or any(not isinstance(input_spec[key], str) for key in ('text', 'images', 'query'))
            or input_spec["text"] not in {"required", "optional", "none"}
            or input_spec["images"] not in {"required", "optional", "none"}
            or type(input_spec["sequence"]) is not bool or input_spec["query"] not in {"single", "terms"}):
        raise ValueError("Invalid task input policy.")
    operation = value["florence_operation"]
    if operation is not None and (not isinstance(operation, str) or florence_operation(operation) is None):
        raise ValueError("Unknown Florence operation.")
    if value["handler"] == "florence" and operation is None:
        raise ValueError("Florence tasks require a native operation.")
    if "detection" in value["nodes"] and value["handler"] not in {"grounding", "region_caption", "florence"}:
        raise ValueError("Detection tasks require a supported spatial or Florence operation.")
    if value["handler"] in {"grounding", "region_caption", "florence", "scene_motion", "scene_description"} and input_spec["images"] != "required":
        raise ValueError("This operation requires image input.")
    if value["handler"] == "grounding" and (input_spec["query"] != "terms" or input_spec["text"] != "required"):
        raise ValueError("Phrase grounding requires query text and the terms policy.")
    if input_spec["query"] == "terms" and value["handler"] != "grounding":
        raise ValueError("Only phrase grounding splits comma-separated targets.")
    if input_spec['sequence'] and input_spec['images'] == 'none':
        raise ValueError('Image sequences require an image-capable task.')
    if legacy:
        _validate_legacy_profiles(value)
    # Match json_store's on-disk formatting, including its trailing newline.
    if len((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()) > MAX_FILE_BYTES:
        raise ValueError("Task definition exceeds 256 KiB.")
    return copy.deepcopy(value)


def _validate_legacy_profiles(value):
    """Validate archived v1 data before migration, never used during generation."""
    if not isinstance(value["profiles"], dict) or set(value["profiles"]) - {"standard", "nsfw"}:
        raise ValueError("Unknown instruction profile.")
    for override in value["profiles"].values():
        if not isinstance(override, dict) or set(override) - {"instructions", "append", "template"} or any(not isinstance(v, str) for v in override.values()):
            raise ValueError("Invalid profile instruction override.")
    if not isinstance(value["examples"], dict) or set(value["examples"]) != {"standard", "nsfw"}:
        raise ValueError("Examples require Standard and NSFW lists.")
    for examples in value["examples"].values():
        if not isinstance(examples, list) or len(examples) > 64:
            raise ValueError("A profile supports up to 64 example messages.")
        for example in examples:
            if (not isinstance(example, dict) or set(example) != {"role", "content"}
                    or not isinstance(example['role'], str) or example["role"] not in {"user", "assistant"}):
                raise ValueError("Examples must be user/assistant messages.")
            content = example["content"]
            if not isinstance(content, str) and (not isinstance(content, list) or any(
                    not isinstance(v, dict) or set(v) != {"type", "text"} or v["type"] != "text" or not isinstance(v["text"], str) for v in content)):
                raise ValueError("Example content must be text.")


@dataclass(frozen=True)
class Task:
    definition: dict
    filename: str
    revision: str

    @property
    def id(self): return self.definition["id"]
    @property
    def name(self): return self.definition["name"]
    @property
    def category(self):
        if "detection" in self.nodes:
            return "detection"
        group = self.definition["group"]
        return group if group in LEGACY_TASK_GROUPS else "vision" if self.needs_image else "text"
    @property
    def nodes(self): return tuple(self.definition["nodes"])
    @property
    def needs_image(self): return self.definition["input"]["images"] == "required"
    @property
    def handler(self): return self.definition["handler"]
    @property
    def florence_id(self): return self.definition["florence_operation"]
    @property
    def florence_token(self):
        operation = florence_operation(self.florence_id)
        return operation.token if operation else None
    @property
    def family_filter(self): return "Florence" if self.handler == "florence" else None
    @property
    def enabled(self): return self.definition["enabled"]
    @property
    def internal(self): return self.definition["internal"]
    @property
    def output(self): return HANDLERS[self.handler]


@dataclass(frozen=True)
class Catalog:
    tasks: dict
    aliases: dict
    errors: tuple
    deleted: dict

    def get(self, reference):
        key = reference if reference in self.tasks else self.aliases.get(reference)
        return self.tasks.get(key)

    def require(self, reference):
        task = self.get(reference)
        if task is None:
            raise ValueError(f"Task is missing or invalid: {reference}. Restore it in Tasks or select another task.")
        return task


def invalidate_catalog():
    with _cache_lock:
        _cache.clear()


def load_catalog(root=None):
    if root is None and _snapshot.get() is not None:
        return _snapshot.get()
    root = Path(root or REPO_ROOT)
    directory = root / "tasks"
    if not directory.exists():
        directory = root / ".defaults/tasks"
    if directory.is_symlink():
        raise ValueError("Task folder must not be a symbolic link.")
    files = sorted(directory.glob("*.json" if directory.name == "tasks" and directory.parent == root else "*.json.example"))
    files = [path for path in files if not path.name.startswith(".")]
    if len(files) > MAX_TASKS:
        raise ValueError("Task library exceeds 512 files.")
    state_path = root / "tasks/.state.json"
    extra = [state_path] if state_path.exists() else []
    signature = tuple((str(path), path.lstat().st_mtime_ns, path.lstat().st_size, path.lstat().st_ino) for path in [*files, *extra])
    with _cache_lock:
        cached = _cache.get(str(root))
        if cached and cached[0] == signature:
            return cached[1]
    state = _read(state_path, root / "tasks") if state_path.exists() else {}
    deleted = state.get("deleted", {})
    if not isinstance(deleted, dict) or any(not isinstance(value, dict) for value in deleted.values()):
        raise ValueError("Invalid task deletion state.")
    tasks, aliases, errors, duplicates = {}, {}, [], set()
    for path in files:
        try:
            data = validate_task(_read(path, directory))
            if data["id"] in deleted:
                continue
            if data["id"] in tasks:
                duplicates.add(data["id"])
                raise ValueError(f"Duplicate task ID: {data['id']}")
            tasks[data["id"]] = Task(data, path.name, digest(data))
        except (ValueError, OSError, UnicodeError) as error:
            errors.append(f"{path.name}: {error}")
    for key in duplicates:
        tasks.pop(key, None)
    ambiguous = set()
    missing = state.get('missing', {})
    if not isinstance(missing, dict) or any(
            not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs) for refs in missing.values()):
        raise ValueError('Invalid task migration alias state.')
    for key, task in tasks.items():
        # A consolidated task's old-ID aliases are fallbacks. Retained custom
        # variants keep their own names/aliases; explicit deletions stay deleted.
        shadowed = set()
        for reference in task.definition['aliases']:
            if reference in tasks and reference != key:
                prior = tasks[reference]
                shadowed.update((prior.id, prior.name, *prior.definition['aliases']))
            elif reference in deleted:
                name = deleted[reference].get('name', '')
                shadowed.add(reference)
                if isinstance(name, str):
                    shadowed.add(name)
                    shadowed.add(name.lower().replace(' ', '_'))
            elif reference in missing:
                shadowed.update(missing[reference])
        for alias in (task.name, *task.definition["aliases"]):
            if alias in shadowed:
                continue
            if alias in tasks and alias != key or alias in aliases and aliases[alias] != key:
                ambiguous.add(alias)
            else:
                aliases[alias] = key
    for alias in ambiguous:
        aliases.pop(alias, None)
        errors.append(f"Ambiguous task alias: {alias}")
    catalog = Catalog(tasks, aliases, tuple(errors), deleted)
    with _cache_lock:
        _cache[str(root)] = (signature, catalog)
    return catalog


def get_task(reference):
    return load_catalog().get(reference)


def handler(reference):
    task = get_task(reference)
    return task.handler if task else ""


@contextmanager
def task_scope(*, catalog=None):
    token = _snapshot.set(catalog or load_catalog())
    try:
        yield _snapshot.get()
    finally:
        _snapshot.reset(token)


def prompt_config(reference):
    catalog = load_catalog()
    task = catalog.require(reference)
    return {"display_name": task.id, "system_prompt": task.definition["instructions"],
            "instruction_template": task.definition["template"]}


def task_configs():
    catalog = load_catalog()
    resolved = {**catalog.aliases, **{key: key for key in catalog.tasks}}
    references = {key.lower().replace(" ", "_"): value for key, value in resolved.items()}
    return _TaskConfigs(references)


class _TaskConfigs(Mapping):
    def __init__(self, references):
        self.references = references

    def __iter__(self):
        return iter(self.references)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, key):
        from .prompt_flow import prepared
        config = prompt_config(self.references[key])
        if prepared.get():
            config['instruction_template'] = ''
        return config


def task_references(parameters):
    raw = parameters.get("task_sequence", "")
    while isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    if raw:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid task selection JSON.") from error
        if not isinstance(data, dict) or set(data) != {"version", "tasks"} or type(data["version"]) is not int or data["version"] != 1 or not isinstance(data["tasks"], list):
            raise ValueError("Expected version 1 task selection.")
        refs = data["tasks"]
    else:
        def scalar(key, default=None):
            value = parameters.get(key, default)
            while isinstance(value, (list, tuple)):
                value = value[0] if value else default
            return value
        refs = [scalar("task", "")]
        if scalar("multi_task_mode", False):
            for key in ("task_2", "task_3", "task_4"):
                value = scalar(key, "None")
                if not value or value == "None":
                    break
                refs.append(value)
    if not 1 <= len(refs) <= MAX_STEPS or any(not isinstance(ref, str) or not ref for ref in refs):
        raise ValueError("Select between 1 and 32 tasks.")
    return refs


def validate_plan(references, entry, kind="loader", *, scene_json=False):
    if kind == "detection" and len(references) != 1:
        raise ValueError("Detection uses one task with comma-separated grounding targets.")
    if entry.get("backend") in {"wd14", "yolo"}:
        return []
    selected = []
    for reference in references:
        task = load_catalog().require(reference)
        issue = chain_issue(task, entry, selected) if kind == "loader" else model_issue(task, entry, kind)
        if issue:
            raise ValueError(f"{task.name}: {issue}")
        if task.id in {previous.id for previous in selected}:
            raise ValueError("A task may only be selected once in a chain.")
        selected.append(task)
    if scene_json and selected and selected[-1].output not in {"text", "h3", "scene"}:
        raise ValueError("Scene JSON requires visual prose, H3 or a scene observation task.")
    return selected


def catalog_metadata(entry, kind):
    catalog = load_catalog()
    values = []
    for task in catalog.tasks.values():
        if task.internal or kind not in task.nodes:
            continue
        values.append({"id": task.id, "name": task.name, "aliases": task.definition["aliases"],
                       "group": LEGACY_TASK_GROUPS.get(task.definition["group"], task.definition["group"]), "description": task.definition["description"],
                       "handler": task.handler, "output": task.output, "input": task.definition["input"],
                       "florence_operation": task.florence_id, "issue": model_issue(task, entry, kind),
                       "revision": task.revision, "enabled": task.enabled})
    return copy.deepcopy({"tasks": values, "groups": TASK_GROUPS, "group_aliases": LEGACY_TASK_GROUPS,
                          "errors": list(catalog.errors), "deleted": [
        {"id": key, "name": value.get("name", key)} for key, value in catalog.deleted.items()]})


def read_task(reference):
    task = load_catalog().require(reference)
    return {"task": copy.deepcopy(task.definition), "revision": task.revision, "file": f"tasks/{task.filename}"}


def new_task(kind='loader'):
    if kind not in {'loader', 'detection'}:
        raise ValueError('Unknown task node.')
    detection = kind == 'detection'
    return {"version": 2, "id": 'draft', "name": 'New task', "description": '',
                "group": 'detection' if detection else 'chat', "nodes": [kind], "aliases": [],
                "input": {"text": 'required' if detection else 'optional', "images": 'required' if detection else 'optional',
                           "sequence": not detection, "query": 'terms' if detection else 'single'},
                "handler": 'grounding' if detection else 'chat', "florence_operation": None,
                "instructions": 'Locate the requested targets and return bounding boxes as JSON.' if detection else 'Follow the user’s request.',
                "template": '', "internal": False, "enabled": True}


def save_task(reference, revision, changes, *, copy_as_new=False):
    root = REPO_ROOT / "tasks"
    _safe(root / ".catalog", root, exists=False)
    if not isinstance(changes, dict):
        raise ValueError("Task changes must be an object.")  # noqa: TRY004 -- public validation contract
    with locked_path(root / ".catalog"):
        invalidate_catalog()
        catalog = load_catalog()
        current = catalog.require(reference) if reference else None
        if current is not None and current.revision != revision:
            raise TaskConflict("This task changed since you opened it. Your draft is retained; reload before saving.")
        value = copy.deepcopy(current.definition) if current else new_task()
        if set(changes) - (set(value) - {"version", "id", "aliases", "internal"}):
            raise ValueError("Unknown or protected task fields.")
        value.update(changes)
        copy_as_new = copy_as_new or current is None
        if copy_as_new:
            _check_add_capacity(root)
            value.update(id=f"user.{uuid.uuid4().hex}", aliases=[], internal=False)
            filename = value["id"] + ".json"
        else:
            filename = current.filename
            if value["name"] != current.name and current.name not in value["aliases"]:
                value["aliases"].append(current.name)
        value = validate_task(value)
        existing = catalog.get(value["name"])
        if existing is not None and existing.id != value["id"]:
            raise ValueError("Another task already uses this name.")
        path = _safe(root / filename, root, exists=not copy_as_new)
        def replace(data):
            if copy_as_new and data or not copy_as_new and digest(data) != revision:
                raise TaskConflict("The task changed while saving. Your draft is retained.")
            data.clear()
            data.update(value)
        update_json_object(path, replace)
    invalidate_catalog()
    return read_task(value["id"])


def _check_add_capacity(root):
    # Count files, including invalid definitions, just as load_catalog does.
    # Call while holding the catalog lock, before creating/restoring a file.
    if sum(not path.name.startswith('.') for path in root.glob('*.json')) >= MAX_TASKS:
        raise ValueError(f"Task library has reached its limit of {MAX_TASKS} files. Remove a task first.")


def remove_task(reference, revision):
    root = REPO_ROOT / "tasks"
    _safe(root / '.state.json', root, exists=False)
    with locked_path(root / ".catalog"):
        invalidate_catalog()
        task = load_catalog().require(reference)
        if task.revision != revision:
            raise TaskConflict("This task changed. Reload before removing it.")
        path = _safe(root / task.filename, root)
        trash = root / ".trash"
        _safe(trash / "placeholder", root, exists=False)
        trash.mkdir(exist_ok=True)
        archive = f"{uuid.uuid4().hex}.json"
        write_json_object(trash / archive, task.definition, private=True)
        def mark(state):
            state.setdefault("deleted", {})[task.id] = {"name": task.name, "file": task.filename, "archive": archive}
        update_json_object(root / ".state.json", mark, private=True)
        path.unlink()
    invalidate_catalog()


def restore_task(reference):
    _identifier(reference)
    root = REPO_ROOT / "tasks"
    _safe(root / '.state.json', root)
    with locked_path(root / ".catalog"):
        state = _read(root / ".state.json", root)
        removed = state.get("deleted", {}).get(reference)
        if not isinstance(removed, dict):
            raise ValueError("No removed task with this ID.")  # noqa: TRY004 -- public validation contract
        filename, archive = removed.get("file", ""), removed.get("archive", "")
        if Path(filename).name != filename or not filename.endswith(".json") or not re.fullmatch(r"[a-f0-9]{32}\.json", archive):
            raise ValueError("Invalid removed-task record.")
        value = validate_task(_read(root / ".trash" / archive, root))
        if value["id"] != reference:
            raise ValueError("Removed task ID does not match its record.")
        path = _safe(root / filename, root, exists=False)
        if path.exists():
            raise TaskConflict("A file already occupies this task's original path.")
        _check_add_capacity(root)
        write_json_object(path, value)
        def unmark(data):
            data['deleted'].pop(reference)
        update_json_object(root / ".state.json", unmark, private=True)
    invalidate_catalog()
    return read_task(reference)
