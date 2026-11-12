"""Deterministic cleanup for task results that contain only tags."""
import re

from .task_library import get_task


def _split_tags(text):
    """Split list separators without splitting escaped or weighted tag groups."""
    start, closing, escaped = 0, [], False
    brackets = {'(': ')', '[': ']', '{': '}'}
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char in brackets:
            closing.append(brackets[char])
        elif closing and char == closing[-1]:
            closing.pop()
        elif not closing and char in ',\r\n':
            yield text[start:index]
            start = index + 1
    yield text[start:]


def deduplicate_tags(text):
    """Keep the first spelling/order; weights and different phrases stay distinct."""
    tags, seen = [], set()
    for part in _split_tags(text):
        tag = part.strip()
        key = re.sub(r'[\s_]+', ' ', tag).casefold()
        if key and key not in seen:
            seen.add(key)
            tags.append(tag)
    return ', '.join(tags)


def clean_tag_output(task_name, text):
    task = get_task(task_name)
    if task and (task.handler == 'text_to_tags' or task.florence_id == 'prompt_gen_tags'):
        return deduplicate_tags(text)
    return text
