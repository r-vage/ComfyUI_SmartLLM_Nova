"""Task endpoints and prompt inspection, scoped to SmartLLM Nova."""
import asyncio

from aiohttp import web

from .task_library import (
    TaskConflict,
    catalog_metadata,
    digest,
    load_catalog,
    new_task,
    prompt_config,
    read_task,
    remove_task,
    restore_task,
    save_task,
    task_scope,
    validate_plan,
)
from .task_operations import FLORENCE_OPERATIONS, HANDLERS


def preview(reference, user='', *, direct_system=''):
    from .prompt_flow import user_message
    from .tasks import get_system_prompt

    with task_scope():
        task = load_catalog().require(reference)
        config = prompt_config(task.id)
        system = direct_system if task.handler == 'direct_chat' and direct_system.strip() else get_system_prompt(task.id)
        return {'id': task.id, 'name': task.name, 'system': system, 'user': user_message(task, user, template=config['instruction_template']),
                    'operation': task.florence_token, 'revision': task.revision}


def fingerprint():
    catalog = load_catalog()
    return digest({'tasks': {key: task.revision for key, task in catalog.tasks.items()},
                   'errors': catalog.errors, 'deleted': catalog.deleted})


def register(routes, read_json, mutation_denial):
    def entry_for(name):
        from .model_registry import get_model_entry
        if not isinstance(name, str) or len(name) > 2048:
            raise ValueError('Invalid model selection.')
        entry = get_model_entry(name) if name else {}
        if entry is None:
            raise ValueError('The selected model is unavailable. Refresh or choose another model.')
        return entry

    @routes.get('/smartllm-nova/tasks')
    async def catalog(request):
        try:
            kind = request.query.get('kind', 'loader')
            if kind not in {'loader', 'detection'}:
                raise ValueError('Unknown task node.')
            entry = await asyncio.to_thread(entry_for, request.query.get('model', ''))
            data = await asyncio.to_thread(catalog_metadata, entry, kind)
            data.update(operations={key: {'token': op.token, 'text': op.text, 'spatial': op.spatial} for key, op in FLORENCE_OPERATIONS.items()},
                        handlers=HANDLERS, template=new_task(kind))
            return web.json_response(data)
        except (ValueError, OSError) as error:
            return web.json_response({'error': str(error)}, status=400)

    @routes.get('/smartllm-nova/tasks/read')
    async def read(request):
        try:
            return web.json_response(await asyncio.to_thread(read_task, request.query.get('id', '')))
        except (ValueError, OSError) as error:
            return web.json_response({'error': str(error)}, status=400)

    @routes.post('/smartllm-nova/tasks/preview')
    async def inspect(request):
        try:
            data = await read_json(request)
            if set(data) - {'id', 'user', 'direct_system'} or not isinstance(data.get('id'), str):
                raise ValueError('Invalid preview request.')
            if any(not isinstance(data.get(key, ''), str) for key in ('user', 'direct_system')):
                raise ValueError('Invalid prompt preview.')
            result = await asyncio.to_thread(preview, data['id'], data.get('user', ''),
                                             direct_system=data.get('direct_system', ''))
            return web.json_response(result)
        except (ValueError, OSError) as error:
            return web.json_response({'error': str(error)}, status=400)

    @routes.get('/smartllm-nova/tasks/preview')
    async def inspect_default(request):
        try:
            result = await asyncio.to_thread(preview, request.query.get('id', ''))
            return web.json_response(result)
        except (ValueError, OSError) as error:
            return web.json_response({'error': str(error)}, status=400)

    @routes.post('/smartllm-nova/tasks/validate')
    async def validate(request):
        try:
            data = await read_json(request)
            references = data.get('tasks')
            if (not isinstance(references, list) or not 1 <= len(references) <= 32
                    or any(not isinstance(value, str) for value in references)
                    or data.get('kind', 'loader') not in {'loader', 'detection'}
                    or type(data.get('scene_json', False)) is not bool):
                raise ValueError('Invalid task selection.')
            entry = await asyncio.to_thread(entry_for, data.get('model', ''))
            selected = await asyncio.to_thread(validate_plan, references, entry, data.get('kind', 'loader'),
                                               scene_json=data.get('scene_json', False))
            return web.json_response({'tasks': [task.id for task in selected]})
        except (ValueError, OSError) as error:
            return web.json_response({'error': str(error)}, status=400)

    @routes.post('/smartllm-nova/tasks/edit')
    async def edit(request):
        denial = mutation_denial(request)
        if denial is not None:
            return denial
        try:
            data = await read_json(request)
            if set(data) - {'action', 'id', 'revision', 'changes'} or not isinstance(data.get('id', ''), str):
                raise ValueError('Invalid task edit request.')
            action = data.get('action')
            if action in {'save', 'duplicate', 'create'}:
                result = await asyncio.to_thread(save_task, data.get('id') if action != 'create' else None,
                                                 data.get('revision'), data.get('changes', {}), copy_as_new=action == 'duplicate')
            elif action == 'remove':
                await asyncio.to_thread(remove_task, data['id'], data.get('revision'))
                result = {'removed': data['id']}
            elif action == 'restore':
                result = await asyncio.to_thread(restore_task, data['id'])
            else:
                raise ValueError('Unknown task edit action.')
            return web.json_response(result)
        except TaskConflict as error:
            return web.json_response({'error': str(error)}, status=409)
        except (ValueError, OSError) as error:
            return web.json_response({'error': str(error)}, status=400)
