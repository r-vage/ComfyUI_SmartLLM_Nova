"""One task/system/user composition step before any generation backend."""
from contextvars import ContextVar
from functools import wraps

from .creative_inputs import observing
from .task_library import get_task, prompt_config

prepared = ContextVar('smartllm_prompt_prepared', default=False)


def user_message(task, text, *, template=None):
    text = str(text or '')
    if task.handler in {'tags_to_text', 'text_to_tags'}:
        text = text.replace('_', ' ')
    template = prompt_config(task.id)['instruction_template'] if template is None else template
    return template.replace('{prompt}', text) if '{prompt}' in template else f'{template} {text}' if template else text


def with_prompt_flow(function):
    @wraps(function)
    def generate(instance, **kwargs):
        from .song import active_source
        from .tasks import (
            get_system_prompt,
            push_system_prompt_override,
            reset_system_prompt_override,
        )

        task = get_task(kwargs['task_name'])
        if task is None:
            raise ValueError(f"Task is missing: {kwargs['task_name']}")
        source = active_source.get()
        reference = source.prompt_key if source is not None and source.mode == 'convert' else task.id
        config = prompt_config(reference)
        system = kwargs.get('system_prompt')
        if system is None:
            system = get_system_prompt(task.id)
        kwargs['prompt'] = (str(kwargs.get('prompt') or '') if observing.get() else
                            user_message(task, kwargs.get('prompt'), template=config['instruction_template']))
        kwargs['system_prompt'] = system
        token = push_system_prompt_override(system)
        flow_token = prepared.set(True)
        try:
            return function(instance, **kwargs)
        finally:
            prepared.reset(flow_token)
            reset_system_prompt_override(token)
    return generate
