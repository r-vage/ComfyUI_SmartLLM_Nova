"""Bound native Qwen caption reasoning without restarting image generation."""

from collections import deque
from contextlib import contextmanager
from functools import partial

import torch

from .task_library import get_task


class _ThinkingBudget:
    def __init__(self, *, limit, opening, closing, transition, stops, prefilled):
        self.limit = limit
        self.opening = opening
        self.closing = closing
        self.transition = transition
        self.stops = stops
        self.thinking = prefilled
        self.used = 0
        self.pending = deque()

    def _finish_thinking(self):
        self.pending.extend(self.transition)
        self.thinking = False

    def sample(self, original, logits, *args, **kwargs):
        if self.thinking and self.used >= self.limit:
            self._finish_thinking()
        if self.pending:
            return torch.full((logits.shape[0], 1), self.pending.popleft(),
                              dtype=torch.long, device=logits.device)
        token = original(logits, *args, **kwargs)
        value = token.item()  # Native generation already uses a single sequence.
        if value == self.opening:
            if self.thinking:
                self.used += 1
            self.thinking = True
        elif value == self.closing:
            self.thinking = False
        elif self.thinking and value in self.stops:
            # An EOS during reasoning would otherwise leave no caption at all.
            self._finish_thinking()
            return torch.full_like(token, self.pending.popleft())
        elif self.thinking:
            self.used += 1
        return token


@contextmanager
def description_thinking_budget(clip, architecture, task_name, max_tokens, *, prefilled, enabled=True):
    """Caller holds the encoder's generation lock for the scoped sampler override.

    Only the description tasks on native Qwen3/3.5 use this policy. Other tasks,
    raw Qwen3.5 chat templates and other native families retain their behavior.
    Forced closing tokens count towards the same total as model-generated tokens.
    """
    task = get_task(task_name)
    caption = task is not None and task.florence_id in {'caption', 'detailed_caption', 'more_detailed_caption'}
    eligible = enabled and caption and architecture.startswith(("QWEN3_", "QWEN35_"))
    if not eligible or architecture.startswith("QWEN35_") and not prefilled:
        yield {}
        return

    model = clip.cond_stage_model
    inner = getattr(model, getattr(model, "clip", ""), None)
    decoder = getattr(inner, "transformer", None)
    original = getattr(decoder, "sample_token", None)
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer = getattr(tokenizer, getattr(tokenizer, "clip", ""), None)
    tokenizer = getattr(tokenizer, "tokenizer", None)
    if not callable(original) or not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("This ComfyUI encoder does not expose native Qwen thinking-budget support. Update ComfyUI.")

    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    transition = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    if (len(opening) != 1 or len(closing) != 1 or opening == closing
            or not transition or transition[0] != closing[0]
            or tokenizer.decode(opening) != "<think>" or tokenizer.decode(closing) != "</think>"):
        raise RuntimeError("The native Qwen tokenizer has unsupported thinking delimiters.")

    state = _ThinkingBudget(
        limit=min(128, max_tokens // 4), opening=opening[0], closing=closing[0],
        transition=transition, stops=set(decoder.model.config.stop_tokens), prefilled=prefilled,
    )
    had_override = "sample_token" in vars(decoder)
    previous = vars(decoder).get("sample_token")
    decoder.sample_token = partial(state.sample, original)
    try:
        # Speculative MTP bypasses sample_token; use the ordinary cached decoder
        # for this bounded request. No weights, shared classes or KV state change.
        yield {"mtp": False} if getattr(decoder, "mtp", None) is not None else {}
    finally:
        if had_override:
            decoder.sample_token = previous
        else:
            del decoder.sample_token
