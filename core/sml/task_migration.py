"""Install bundled tasks and preserve edits made to the task library itself."""
import hashlib
import shutil
from pathlib import Path

from ..json_store import (
    locked_path,
    read_json_object,
    write_json_object,
)
from .logger import log
from .task_library import _read, _safe, digest, invalidate_catalog, validate_task

# Retired config files are ignored, including by general default extraction.
LEGACY_FILES = ('system_prompts.json', 'system_prompts_nsfw.json',
                'llm_few_shot_training.json', 'llm_few_shot_training_nsfw.json')
_PRIOR_NSFW_PROFILE = ('\n\nNSFW: when the input contains explicit content, describe it directly and anatomically. '
                      'Do not censor, soften, or euphemize.')
_DIRECT_DESCRIPTION = ('Describe supported anatomy and sexual activity directly, using terms such as breasts, '
                       'nipples, vulva or penis where applicable. Do not infer nudity beneath clothing. '
                       "Keep the task's requested format and detail.")
_DIRECT_TEXT = ('Preserve explicit meaning and terminology from the source without censorship or euphemisms. '
                "Keep the task's requested format and detail; do not add unsupported content.")
_OLD_ADDITIONS = (_PRIOR_NSFW_PROFILE.strip(),
                  'NSFW: describe explicit content directly and anatomically without censorship.',
                  _DIRECT_DESCRIPTION)

# Only untouched duration presets are retired; customized tasks keep their IDs.
_RETIRED_TASKS = {'builtin.wan-2-2-scene-10s': {'name': 'Wan 2.2 Scene 10s',
                               'aliases': ['wan_2.2_scene_10s'],
                               'hash': '0295131c9d1f6a0a10afae1ca485bfb3217f2b9d9836e433792f56eee2de6999',
                               'instructions': '3cea4d9f3a5485d5864354bec2ea99f695244abea9a769317fc64411d76758b9',
                               'template': 'ba967b5998db2c9b05e9f5920d70b54491a6705d6ab6441f3dc6c89276eeaea3'},
 'builtin.wan-2-2-scene-20s': {'name': 'Wan 2.2 Scene 20s',
                               'aliases': ['wan_2.2_scene_20s'],
                               'hash': '2b461cc488bff56cdc467acd29cc818b45a1d29b879bf871b16275da9cc35e6c',
                               'instructions': 'a0758b7e067d7796fcb2576930f7123c9f8d74e3b5ad07b692f11c6a8ff7604d',
                               'template': 'e8b6b8217dc78266123cb758962e42d7ac0e72e8b9b574b5cd1229d8a119f163'},
 'builtin.wan-2-2-timeline-10s': {'name': 'Wan 2.2 Timeline 10s',
                                  'aliases': ['wan_2.2_timeline_10s'],
                                  'hash': 'f6a8130fc958fd523e8801d9a184ecec6157afc6a72e00a08e5abbc046ab3953',
                                  'instructions': '14680012beec769c5f9bf8490e34ffdd8f1ff3adb83de180519bfd7e3c93a4c5',
                                  'template': '3da6cf1cc031ab49b94b0fe459b545bec58b20e070b2ed8832caad05a819c403'},
 'builtin.wan-2-2-timeline-20s': {'name': 'Wan 2.2 Timeline 20s',
                                  'aliases': ['wan_2.2_timeline_20s'],
                                  'hash': 'b5b1fd3aa36463614c67f274f8e7a6e8981f8aee11d3fdb06c39d663dd1ff3bc',
                                  'instructions': 'edecc6a4f9f76558fa30a51a18ddb5c7125a23a2c9b894be04a0c6c72e1017be',
                                  'template': '7fa5be29e6c06e594ddfbae68123fe33dca7eb8585ec2b7b08e6487bdf2d5b9b'},
 'builtin.wan-2-2-timeline-5s-2s': {'name': 'Wan 2.2 Timeline 5s 2s',
                                    'aliases': ['wan_2.2_timeline_5s_2s'],
                                    'hash': 'f89576025c42d8921501f5ed75fe2a2805b5cb488d4470f060fa86174eedb228',
                                    'instructions': '77cbdebe5b7bdaf11cc4533655497eeadd71ba70cf0f9ac4f3064f14647d4267',
                                    'template': '2d1393b5e8bd2b300224f060d0d7a16f95566a87ecd694e9b835af67a539cb0c'},
 'builtin.wan-2-2-timeline-5s-3s': {'name': 'Wan 2.2 Timeline 5s 3s',
                                    'aliases': ['wan_2.2_timeline_5s_3s'],
                                    'hash': '4ce43e6a406b0d416cdd88c114efc97f12e0a41c54c54b5e0948b05f79195609',
                                    'instructions': '688d4c88d6ad7b36424e769ec4b157d1789a36a6ff96a1075f2f37cfa4f94f3b',
                                    'template': '40a7ee48fe39450a1784803f9289617867b69454ac3bf96d148faf62f5b2de26'},
 'builtin.wan-2-2-timeline-5s': {'name': 'Wan 2.2 Timeline 5s',
                                 'aliases': ['wan_2.2_timeline_5s'],
                                 'hash': '3dbfc4acfe809e14a20c2f8aa69ebf10d27bf6fcf6f35496ed1c964576f45223',
                                 'instructions': '252ad3655e709be6d9ed1c3779986add7903e4fb4986bda9cafde69b7e165186',
                                 'template': 'c2372e45566616bd37ecbe036eeb8671281e3c11d2dca46cccc88d2d8a73474f'}}


def _guidance(task):
    """Migration/default-authoring policy; generation reads only the saved task."""
    if task['handler'] == 'florence' or task['internal']:
        return ''
    if task['handler'] in {'grounding', 'region_caption'}:
        return ('Use precise names, including anatomical terms for applicable targets. '
                'Report only visible matches in the required JSON structure.')
    if task['id'] == 'builtin.ocr':
        return 'Transcribe the visible text faithfully, without censoring or paraphrasing.'
    if task['input']['images'] == 'required' or task['handler'].startswith('h3_') or task['id'].startswith(('builtin.wan-', 'builtin.ltx-')):
        return _DIRECT_DESCRIPTION
    if task['handler'] in {'direct_chat', 'story', 'music3', 'yue2', 'song_lyrics'}:
        return ''
    return _DIRECT_TEXT


def _strip_stock_addition(text):
    text = text.rstrip()
    for addition in _OLD_ADDITIONS:
        if text.endswith(addition):
            text = text[:-len(addition)].rstrip()
    return text


def unify_task(task):
    """Upgrade a task's own fields without importing retired shared profiles."""
    if task.get('version') == 2:
        return validate_task(task)
    value = validate_task(task, legacy=True)
    if value['description'] == f"{value['name']}. Instructions and examples can be edited.":
        value['description'] = f"{value['name']}. Task instructions can be edited."
    standard = value['profiles'].get('standard', {})
    base = standard.get('instructions', value['instructions'])
    value['template'] = standard.get('template', value['template'])
    # A different alternate prompt/template may contain an incompatible output
    # contract. The primary version wins; the complete original is backed up.
    additions = []
    for override in (standard, value['profiles'].get('nsfw', {})):
        alternate = override.get('instructions', '')
        if alternate.startswith(base) and alternate != base:
            additions.append(alternate[len(base):].strip())
    base = _strip_stock_addition(base)
    for addition in additions:
        if addition and addition not in _OLD_ADDITIONS and addition not in base:
            base = f'{base}\n\n{addition}'.strip()
    guidance = _guidance(value)
    if guidance and guidance not in base:
        base = f'{base}\n\n{guidance}'.strip()
    value.update(version=2, instructions=base)
    del value['profiles'], value['examples']
    return validate_task(value)


def _backup(source, directory, root):
    target = directory / source.name
    _safe(target, root, exists=False)
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        target = directory / (source.stem + '-' + hashlib.sha256(source.read_bytes()).hexdigest()[:16] + source.suffix)
        _safe(target, root, exists=False)
    if not target.exists():
        shutil.copy2(source, target)


def materialize_tasks(repo_root):
    root = Path(repo_root)
    bundled = root / '.defaults/tasks'
    if not bundled.is_dir():
        return
    directory = root / 'tasks'
    _safe(directory / '.state.json', directory, exists=False)
    directory.mkdir(exist_ok=True)
    with locked_path(directory / '.catalog'):
        state_path = directory / '.state.json'
        state = read_json_object(state_path)
        if state and (type(state.get('version')) is not int or state.get('version') != 1
                      or not isinstance(state.get('shipped'), dict) or not isinstance(state.get('deleted'), dict)
                      or not isinstance(state.get('retired', {}), dict) or not isinstance(state.get('missing', {}), dict)):
            raise ValueError('Invalid task migration state. Restore tasks/.state.json from backup.')
        original_state_hash = digest(state)
        existing, invalid = {}, set()
        for path in sorted([*directory.glob('*.json'), *(directory / '.trash').glob('*.json')]):
            if not path.name.startswith('.'):
                try:
                    existing[path] = _read(path, directory)
                except (ValueError, OSError, UnicodeError) as error:
                    invalid.add(path)
                    log.warning('Tasks', f'Leaving invalid task unchanged: {path.name}: {error}')
        upgrading = [path for path, value in existing.items() if value.get('version') == 1]
        updates = {}
        for path in upgrading:
            try:
                updates[path] = unify_task(existing[path])
            except (ValueError, OSError, UnicodeError) as error:
                invalid.add(path)
                log.warning('Tasks', f'Leaving invalid task unchanged: {path.name}: {error}')
        upgrading = [path for path in upgrading if path not in invalid]
        state.setdefault('shipped', {})
        state.setdefault('deleted', {})
        for path in sorted(bundled.glob('*.json.example')):
            definition = validate_task(_read(path, bundled))
            task_id = definition['id']
            filename = path.name.removesuffix('.example')
            target = directory / filename
            old_hash = state['shipped'].get(task_id)
            shipped_hash = digest(definition)
            if target in invalid:
                # The catalog reports the file error. Never replace a user's
                # broken file with a default or follow its symbolic link.
                state['shipped'][task_id] = shipped_hash
                continue
            _safe(target, directory, exists=False)
            if task_id not in state['deleted']:
                if not target.exists() and old_hash is None:
                    updates[target] = definition
                elif target.exists() and old_hash is not None:
                    current = existing.get(target)
                    # Update only a whole file that still matches its recorded
                    # bundled version. Every field in a customized file is owned
                    # by the user, even fields that retain older stock values.
                    if current is not None and digest(current) == old_hash and old_hash != shipped_hash:
                        updates[target] = definition
                # A manually deleted file stays deleted. The shipped hash prevents recreation.
            state['shipped'][task_id] = shipped_hash
        retiring = []
        for task_id, prior in _RETIRED_TASKS.items():
            target = directory / (task_id + '.json')
            if target in invalid:
                state.setdefault('missing', {})[task_id] = [task_id, prior['name'], *prior['aliases']]
                continue
            if task_id in state['deleted']:
                continue
            current = existing.get(target)
            if current is None:
                if task_id in state['shipped'] and task_id not in state.get('retired', {}):
                    # A manually removed preset must not reappear via an alias.
                    state.setdefault('missing', {})[task_id] = [task_id, prior['name'], *prior['aliases']]
                continue
            state.get('missing', {}).pop(task_id, None)
            candidate = updates.get(target, current)
            try:
                validate_task(candidate)
            except ValueError:
                state.setdefault('missing', {})[task_id] = [task_id, prior['name'], *prior['aliases']]
                continue
            if digest(candidate) != prior['hash'] and digest(current) != state['shipped'].get(task_id):
                continue
            _safe(target, directory)
            retiring.append(target)
            updates.pop(target, None)
            state.setdefault('retired', {})[task_id] = target.name
        # Preflight finished. Copy originals before any replacement, including
        # removed-task archives so Restore cannot bring back examples/profiles.
        for path in upgrading:
            _backup(path, directory / '.legacy' / 'v1' / ('removed' if path.parent.name == '.trash' else 'tasks'), directory)
        for path in retiring:
            _backup(path, directory / '.legacy' / 'duration-presets', directory)
        if upgrading and state_path.exists():
            _backup(state_path, directory / '.legacy' / 'v1', directory)
        for target, value in updates.items():
            write_json_object(target, value)
        for target in retiring:
            target.unlink()
        state['version'] = 1
        if digest(state) != original_state_hash:
            write_json_object(state_path, state, private=True)
    invalidate_catalog()
