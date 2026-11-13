# Cancellation belongs to an execution, never to a saved node configuration.

import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from .logger import log

_lock = threading.RLock()
_runs = {}
_revision = 0
_current_run = ContextVar("smartllm_generation", default=None)
DOCKER_BACKENDS = frozenset({"ollama", "vllm", "sglang", "llamacpp"})


@dataclass
class Generation:
    node_id: str
    display_node_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    backend: str = ""
    cancelled: threading.Event = field(default_factory=threading.Event)
    # Keep the execution alive until its stop operation finishes. A late stop
    # must never interrupt the following queued prompt or kill its container.
    stop_lock: threading.RLock = field(default_factory=threading.RLock)

    def public(self):
        return {"run_id": self.run_id, "node_id": self.node_id,
                "display_node_id": self.display_node_id, "backend": self.backend,
                "stopping": self.cancelled.is_set()}


def generation_status():
    with _lock:
        return {"revision": _revision, "runs": [run.public() for run in _runs.values()]}


def _publish_locked():
    global _revision
    _revision += 1
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            PromptServer.instance.send_sync("smartllm-nova/generation", generation_status())
    except (ImportError, AttributeError):
        pass  # Standalone execution has no browser connection.


def check_cancelled():
    _check_run(_current_run.get())


def _check_run(run):
    if run is not None:
        from comfy import model_management
        if run.cancelled.is_set():
            raise model_management.InterruptProcessingException()
        model_management.throw_exception_if_processing_interrupted()


def token_stopping_kwargs(runtime):
    run = _current_run.get()
    if run is None:
        return {}
    # Capture the run for runtimes that evaluate criteria on a worker thread.
    def interrupt_on_token(*_args, **_kwargs):
        _check_run(run)
        return False
    if runtime == "transformers":
        from transformers import StoppingCriteriaList
    elif runtime == "gguf":
        from llama_cpp import StoppingCriteriaList
    else:
        raise ValueError("Unsupported token stopping runtime")
    return {"stopping_criteria": StoppingCriteriaList([interrupt_on_token])}


def cancellable(function):
    # ComfyUI's interruption inherits BaseException, so backend error recovery
    # cannot turn cancellation into a retry or a successful partial response.
    @wraps(function)
    def wrapper(*args, **kwargs):
        check_cancelled()
        try:
            return function(*args, **kwargs)
        finally:
            check_cancelled()
    return wrapper


def set_generation_backend(backend):
    run = _current_run.get()
    if run is not None:
        with _lock:
            run.backend = backend
            _publish_locked()
    check_cancelled()


def _force_stop(backend):
    if backend in DOCKER_BACKENDS:
        from .generation_docker_stop import force_stop_backend
        return force_stop_backend(backend)
    return 0


def stop_generation(run_id):
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        raise LookupError("This LM execution has already finished.")
    with run.stop_lock:
        with _lock:
            if _runs.get(run_id) is not run:
                raise LookupError("This LM execution has already finished.")
            run.cancelled.set()
            from comfy import model_management
            model_management.interrupt_current_processing(True)
            _publish_locked()
        stopped = _force_stop(run.backend)
        return {"success": True, "run_id": run_id, "stopped_containers": stopped,
                "message": "Stop requested. Waiting for LM cleanup."}


@contextmanager
def generation_scope(node_id, display_node_id=None):
    run = Generation(str(node_id), str(display_node_id if display_node_id is not None else node_id))
    token = _current_run.set(run)
    with _lock:
        _runs[run.run_id] = run
        _publish_locked()
    try:
        try:
            yield run
        finally:
            check_cancelled()
    finally:
        # Runs outside the loader's cleanup scope. Also catches a container
        # whose already-running Docker start command completed after Stop.
        with run.stop_lock:
            try:
                if run.cancelled.is_set():
                    try:
                        _force_stop(run.backend)
                    except RuntimeError as error:
                        log.error("Stop LM", str(error))
            finally:
                with _lock:
                    _runs.pop(run.run_id, None)
                    _publish_locked()
                try:
                    check_cancelled()
                finally:
                    _current_run.reset(token)


def with_generation_control(function):
    @wraps(function)
    def wrapper(cls, *args, **kwargs):
        hidden = cls.hidden
        node_id = getattr(hidden, "unique_id", None)
        if isinstance(node_id, list):
            node_id = node_id[0] if node_id else None
        display_node_id = node_id
        dynprompt = getattr(hidden, "dynprompt", None)
        if isinstance(dynprompt, list):
            dynprompt = dynprompt[0] if dynprompt else None
        if dynprompt is not None and node_id is not None:
            display_node_id = dynprompt.get_display_node_id(str(node_id))
        with generation_scope(node_id, display_node_id):
            return function(cls, *args, **kwargs)
    return wrapper
