# Emergency stop of this checkout's managed inference containers only.

import re
import subprocess

from .container_spec import container_is_owned, managed_container_filters
from .generation_control import DOCKER_BACKENDS


def _docker(args, timeout=5):
    try:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace", timeout=min(timeout, 5),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not reach Docker to force-stop the LM. Check the Docker daemon.") from error
    return result.returncode == 0, result.stdout.strip()


def force_stop_backend(backend):
    if backend not in DOCKER_BACKENDS:
        raise ValueError("Unsupported managed Docker backend")
    success, output = _docker(["ps", "--quiet", "--no-trunc", *managed_container_filters(backend)])
    if not success:
        raise RuntimeError("Docker could not list the managed LM containers.")
    identities = output.splitlines()
    if len(identities) > 16 or any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in identities):
        raise RuntimeError("Docker returned an invalid managed container identity.")
    stopped = 0
    for identity in identities:
        # Inspect and kill the same immutable ID, never a potentially reused name.
        if not container_is_owned(identity, backend, _docker):
            raise RuntimeError("Refusing to stop a container without matching SmartLLM Nova ownership.")
        success, _ = _docker(["kill", identity])
        if not success:
            exists, running = _docker(["inspect", identity, "--format", "{{.State.Running}}"])
            if not exists or running != "false":
                raise RuntimeError("Docker could not force-stop the LM container. Check the Docker daemon.")
        stopped += 1
    return stopped
