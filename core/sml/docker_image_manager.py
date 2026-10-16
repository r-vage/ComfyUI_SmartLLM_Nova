# Docker installation overview and managed image operations for the SmartLLM UI.

from __future__ import annotations

import getpass
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .docker_image_policy import (
    OLLAMA_RUNTIME_IMAGES,
    RELEASE_DOCKER_IMAGES,
    resolve_managed_docker_image,
)
from .docker_utils import CREATE_NO_WINDOW, IS_WINDOWS, validate_docker_image
from .json_store import JsonStoreError, read_json_object, update_json_object
from .logger import log

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "docker_config.json"
_INSTALLER_PATH = _REPO_ROOT / "scripts" / "install-docker-engine.sh"
_GUIDE_PATH = _REPO_ROOT / "Readme" / "Docker_Installation_Guide_Linux.md"
_GUIDE_URL = (
    "https://github.com/r-vage/ComfyUI_SmartLLM_Nova/blob/main/"
    "Readme/Docker_Installation_Guide_Linux.md"
)
_DOCKER_OPERATION_LOCK = threading.Lock()
_LOG_PREFIX = "Docker Image Manager"
_MAX_DEBUG_OUTPUT_LINES = 2_000
_MAX_CAPTURED_OUTPUT_LINES = 64
_MAX_OUTPUT_LINE_LENGTH = 512
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SENSITIVE_OUTPUT_RE = re.compile(
    r"(?:authorization|bearer\s|password|passwd|secret|token|credential|cookie|"
    r"api[_-]?key|access[_-]?key|x-registry-auth)",
    re.IGNORECASE,
)

_BACKENDS: dict[str, dict[str, Any]] = {
    "vllm": {
        "label": "vLLM",
        "description": "High-throughput OpenAI-compatible inference server",
        "port": 8000,
    },
    "sglang": {
        "label": "SGLang",
        "description": "Fast server backend with RadixAttention",
        "port": 30000,
    },
    "ollama": {
        "label": "Ollama",
        "description": "Model server used by SmartLLM's Ollama backend",
        "port": 11434,
    },
    "llamacpp": {
        "label": "llama.cpp",
        "description": "Lightweight GGUF inference server",
        "port": 8080,
    },
}
_VENDOR_ALIASES = {
    "auto": "auto",
    "nvidia": "nvidia",
    "amd": "amd",
    "rocm": "amd",
    "none": "none",
    "cpu": "none",
}
_RECOMMENDED_OLLAMA_RUNTIME_VERSION = "0.33.1"
_OLLAMA_RUNTIME_LABELS = {
    "0.33.1": "0.33.1 (recommended)",
    "0.20.2": "0.20.2 (legacy compatibility)",
}


class DockerImageManagerError(RuntimeError):
    pass


class DockerImageManagerBusy(DockerImageManagerError):
    pass


class DockerImageInUse(DockerImageManagerError):
    def __init__(self, containers: list[dict[str, str]]):
        self.containers = containers
        names = ", ".join(item["name"] for item in containers)
        super().__init__(f"Remove the containers using this image first: {names}")


def _run(command: list[str], *, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
        check=False,
    )


def _sanitize_docker_output(line: str) -> str:
    sanitized = _ANSI_ESCAPE_RE.sub("", line)
    sanitized = _CONTROL_CHARACTER_RE.sub("", sanitized).strip()
    if not sanitized:
        return ""
    if _SENSITIVE_OUTPUT_RE.search(sanitized):
        return "[redacted sensitive Docker output]"
    return sanitized[:_MAX_OUTPUT_LINE_LENGTH]


def _run_streaming(
    command: list[str],
    *,
    timeout: int,
    operation: str,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
    except OSError as error:
        raise DockerImageManagerError("Could not start the Docker command") from error

    output_queue: queue.Queue[str | None] = queue.Queue(maxsize=256)
    reader_stop = threading.Event()
    captured_output: deque[str] = deque(maxlen=_MAX_CAPTURED_OUTPUT_LINES)

    def read_output() -> None:
        try:
            if process.stdout is not None:
                for output_line in process.stdout:
                    while not reader_stop.is_set():
                        try:
                            output_queue.put(output_line, timeout=0.25)
                            break
                        except queue.Full:
                            continue
                    if reader_stop.is_set():
                        break
        finally:
            while not reader_stop.is_set():
                try:
                    output_queue.put(None, timeout=0.25)
                    break
                except queue.Full:
                    continue

    reader = threading.Thread(
        target=read_output,
        name="SmartLLM-DockerOutput",
        daemon=True,
    )
    reader.start()
    deadline = time.monotonic() + timeout
    output_finished = False
    debug_line_count = 0
    timed_out = False

    while not output_finished or process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            output_line = output_queue.get(timeout=min(0.25, remaining))
        except queue.Empty:
            continue
        if output_line is None:
            output_finished = True
            continue
        sanitized = _sanitize_docker_output(output_line)
        if not sanitized:
            continue
        captured_output.append(sanitized)
        if debug_line_count < _MAX_DEBUG_OUTPUT_LINES:
            log.debug(_LOG_PREFIX, f"{operation}: {sanitized}")
        elif debug_line_count == _MAX_DEBUG_OUTPUT_LINES:
            log.debug(
                _LOG_PREFIX,
                f"{operation}: additional Docker output omitted",
            )
        debug_line_count += 1

    if timed_out:
        reader_stop.set()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    else:
        process.wait()
    reader.join(timeout=1)

    output = "\n".join(captured_output)
    if timed_out:
        return subprocess.CompletedProcess(
            command,
            124,
            output,
            f"Docker command timed out after {timeout} seconds",
        )
    return subprocess.CompletedProcess(command, process.returncode, output, "")


def _command_error(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    if not lines:
        return "Docker command failed"
    return lines[-1][:512]


def _normalize_vendor(vendor: object) -> str:
    if not isinstance(vendor, str):
        raise TypeError("vendor must be a string")
    normalized = _VENDOR_ALIASES.get(vendor.strip().lower())
    if normalized is None:
        raise ValueError("vendor must be auto, nvidia, amd, rocm, cpu, or none")
    if normalized == "auto":
        from .device import detect_gpu_vendor

        return detect_gpu_vendor()
    return normalized


def _normalize_backend(backend: object) -> str:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    normalized = backend.strip().lower()
    if normalized not in _BACKENDS:
        raise ValueError("Unsupported managed Docker backend")
    return normalized


def normalize_managed_image_selection(
    backend: object,
    vendor: object = "auto",
) -> tuple[str, str]:
    normalized_backend = _normalize_backend(backend)
    normalized_vendor = _normalize_vendor(vendor)
    if normalized_backend not in _backends_for_vendor(normalized_vendor):
        raise ValueError(
            f"{_BACKENDS[normalized_backend]['label']} is not offered in CPU mode"
        )
    return normalized_backend, normalized_vendor


def _backends_for_vendor(vendor: str) -> tuple[str, ...]:
    if vendor == "none":
        return ("ollama", "llamacpp")
    return tuple(_BACKENDS)


def _policy_vendor(backend: str, vendor: str) -> str:
    # The script's CPU mode intentionally uses llama.cpp's audited non-CUDA image.
    if backend == "llamacpp" and vendor == "none":
        return "amd"
    return vendor


def _load_config() -> dict[str, Any]:
    return read_json_object(_CONFIG_PATH, default={})


def _managed_image_reference(
    backend: str,
    vendor: str,
    config: dict[str, Any] | None = None,
    runtime_version: object = None,
) -> str:
    backend, vendor = normalize_managed_image_selection(backend, vendor)

    effective_vendor = _policy_vendor(backend, vendor)
    release_vendor = "amd" if effective_vendor == "amd" else "nvidia"
    if runtime_version is not None:
        if backend != "ollama":
            raise ValueError("Runtime version selection is supported only for Ollama")
        if not isinstance(runtime_version, str):
            raise TypeError("runtime_version must be a string")
        normalized_version = runtime_version.strip()
        if normalized_version not in OLLAMA_RUNTIME_IMAGES:
            raise ValueError("Unsupported managed Ollama runtime version")
        return validate_docker_image(
            OLLAMA_RUNTIME_IMAGES[normalized_version][release_vendor]
        )
    image_key = "docker_image_rocm" if release_vendor == "amd" else "docker_image"
    full_config = _load_config() if config is None else config
    backend_config = full_config.get(backend, {})
    if not isinstance(backend_config, dict):
        backend_config = {}
    configured_image = backend_config.get(
        image_key,
        RELEASE_DOCKER_IMAGES[backend][release_vendor],
    )
    return validate_docker_image(
        resolve_managed_docker_image(
            backend,
            configured_image,
            effective_vendor,
            allow_unpinned=full_config.get("allow_unpinned_docker_images", False),
        )
    )


def _ollama_runtime_version(image_reference: str, vendor: str) -> str:
    effective_vendor = _policy_vendor("ollama", vendor)
    release_vendor = "amd" if effective_vendor == "amd" else "nvidia"
    for version, images in OLLAMA_RUNTIME_IMAGES.items():
        if image_reference == images[release_vendor]:
            return version
    return ""


def _ollama_runtime_options(image_reference: str, vendor: str) -> list[dict[str, Any]]:
    selected_version = _ollama_runtime_version(image_reference, vendor)
    effective_vendor = _policy_vendor("ollama", vendor)
    release_vendor = "amd" if effective_vendor == "amd" else "nvidia"
    return [
        {
            "version": version,
            "label": _OLLAMA_RUNTIME_LABELS[version],
            "image": images[release_vendor],
            "selected": version == selected_version,
            "recommended": version == _RECOMMENDED_OLLAMA_RUNTIME_VERSION,
        }
        for version, images in OLLAMA_RUNTIME_IMAGES.items()
    ]


def _persist_ollama_runtime_image(image_reference: str, vendor: str) -> None:
    effective_vendor = _policy_vendor("ollama", vendor)
    image_key = "docker_image_rocm" if effective_vendor == "amd" else "docker_image"

    def update_ollama_config(config: dict[str, Any]) -> None:
        ollama_config = config.get("ollama")
        if not isinstance(ollama_config, dict):
            ollama_config = {}
            config["ollama"] = ollama_config
        ollama_config[image_key] = image_reference

    try:
        update_json_object(_CONFIG_PATH, update_ollama_config, default={})
    except (JsonStoreError, OSError) as error:
        raise DockerImageManagerError(
            "Ollama image was installed, but SmartLLM could not save the selected "
            "runtime version"
        ) from error


def _format_size(size: object) -> str:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return ""


def _inspect_image(image_reference: str) -> dict[str, Any]:
    template = (
        '{"id":{{json .Id}},"size":{{json .Size}},'
        '"created":{{json .Created}},"repo_digests":{{json .RepoDigests}}}'
    )
    result = _run(
        ["docker", "image", "inspect", image_reference, "--format", template],
        timeout=15,
    )
    if result.returncode != 0:
        return {"installed": False}
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise DockerImageManagerError("Docker returned malformed image metadata") from error
    image_id = metadata.get("id", "")
    size = metadata.get("size")
    return {
        "installed": True,
        "image_id": image_id,
        "short_id": image_id.removeprefix("sha256:")[:12],
        "size_bytes": size if isinstance(size, int) and not isinstance(size, bool) else None,
        "size": _format_size(size),
        "created": metadata.get("created", ""),
        "repo_digests": metadata.get("repo_digests") or [],
    }


def _active_group_names() -> set[str]:
    if os.name == "nt":
        return set()
    try:
        import grp

        return {grp.getgrgid(group_id).gr_name for group_id in os.getgroups()}
    except (ImportError, KeyError, OSError):
        return set()


def _docker_group_configured(username: str) -> bool:
    if os.name == "nt":
        return False
    try:
        import grp

        docker_group = grp.getgrnam("docker")
        return username in docker_group.gr_mem
    except (ImportError, KeyError, OSError):
        return False


def _installation_overview() -> dict[str, Any]:
    system = platform.system()
    username = getpass.getuser()
    docker_path = shutil.which("docker")
    docker_installed = docker_path is not None
    docker_version = ""
    daemon_accessible = False
    daemon_version = ""
    daemon_error = ""

    if docker_installed:
        version_result = _run(["docker", "--version"], timeout=5)
        if version_result.returncode == 0:
            docker_version = version_result.stdout.strip()
        info_result = _run(
            ["docker", "info", "--format", "{{.ServerVersion}}"], timeout=8
        )
        daemon_accessible = info_result.returncode == 0
        if daemon_accessible:
            daemon_version = info_result.stdout.strip()
        else:
            daemon_error = _command_error(info_result)

    active_groups = _active_group_names()
    docker_group_active = "docker" in active_groups
    docker_group_configured = _docker_group_configured(username)
    installer_available = system == "Linux" and _INSTALLER_PATH.is_file()
    setup_state = "ready"
    setup_message = "Docker Engine is installed and accessible to ComfyUI."
    restart_required = False

    if not docker_installed:
        setup_state = "not_installed"
        setup_message = (
            "Docker Engine is not installed. Open SmartLLM's Linux Docker guide "
            "for terminal installation steps."
        )
    elif not daemon_accessible:
        error_lower = daemon_error.lower()
        if "permission denied" in error_lower or (
            docker_group_configured and not docker_group_active
        ):
            setup_state = "permission"
            if docker_group_configured and not docker_group_active:
                setup_message = (
                    "Docker group membership is configured, but this ComfyUI process "
                    "has not inherited it. Log out or reboot, then restart ComfyUI."
                )
                restart_required = True
            else:
                setup_message = (
                    "Docker is installed, but this user cannot access the Docker "
                    "daemon. Open the Linux Docker guide to configure access."
                )
                if system == "Linux":
                    restart_required = True
        else:
            setup_state = "daemon_unavailable"
            setup_message = (
                "Docker is installed, but its daemon is not available. Start Docker "
                "outside ComfyUI; Linux setup steps are available in the guide."
            )

    gpu_vendor = _normalize_vendor("auto")
    gpu: dict[str, Any] = {"vendor": gpu_vendor}
    if gpu_vendor == "nvidia":
        gpu["nvidia_container_toolkit"] = shutil.which("nvidia-ctk") is not None
        nvidia_result = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=8,
        ) if shutil.which("nvidia-smi") else None
        gpu["driver_accessible"] = bool(
            nvidia_result is not None and nvidia_result.returncode == 0
        )
        gpu["devices"] = (
            nvidia_result.stdout.strip().splitlines() if gpu["driver_accessible"] else []
        )
    elif gpu_vendor == "amd":
        gpu["kfd_available"] = Path("/dev/kfd").exists()
        gpu["dri_available"] = Path("/dev/dri").is_dir()

    return {
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "docker": {
            "installed": docker_installed,
            "path": docker_path or "",
            "version": docker_version,
            "daemon_accessible": daemon_accessible,
            "daemon_version": daemon_version,
            "group_configured": docker_group_configured,
            "group_active": docker_group_active,
        },
        "gpu": gpu,
        "setup": {
            "state": setup_state,
            "message": setup_message,
            "installer_available": installer_available,
            # Retain empty legacy keys so older frontends fail closed instead of
            # rendering a command supplied by the server.
            "installer_command": "",
            "command_label": "",
            "restart_required": restart_required,
            "guide_path": str(_GUIDE_PATH) if system == "Linux" else "",
            "guide_url": _GUIDE_URL if system == "Linux" else "",
        },
    }


def get_docker_manager_overview(vendor: object = "auto") -> dict[str, Any]:
    selected_vendor = _normalize_vendor(vendor)
    installation = _installation_overview()
    config = _load_config()
    images = []
    daemon_accessible = installation["docker"]["daemon_accessible"]
    for backend in _backends_for_vendor(selected_vendor):
        details = _BACKENDS[backend]
        image_reference = _managed_image_reference(backend, selected_vendor, config)
        image = {
            "backend": backend,
            "label": details["label"],
            "description": details["description"],
            "port": details["port"],
            "image": image_reference,
            "installed": False,
        }
        if backend == "ollama":
            image["runtime_version"] = _ollama_runtime_version(
                image_reference,
                selected_vendor,
            )
            image["runtime_versions"] = _ollama_runtime_options(
                image_reference,
                selected_vendor,
            )
        if daemon_accessible:
            image.update(_inspect_image(image_reference))
        images.append(image)
    return {
        "success": True,
        "selected_vendor": selected_vendor,
        "installation": installation,
        "images": images,
    }


def _ensure_docker_accessible() -> None:
    if shutil.which("docker") is None:
        log.error(_LOG_PREFIX, "Docker Engine is not installed.")
        raise DockerImageManagerError("Docker Engine is not installed")
    result = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=8)
    if result.returncode != 0:
        detail = _sanitize_docker_output(_command_error(result))
        if detail:
            log.debug(_LOG_PREFIX, f"Docker access check: {detail}")
        log.error(
            _LOG_PREFIX,
            "Docker daemon access failed; check the Docker setup overview.",
        )
        raise DockerImageManagerError(
            "Docker daemon is unavailable to the ComfyUI process; check the setup overview"
        )


@contextmanager
def _docker_operation() -> Iterator[None]:
    if not _DOCKER_OPERATION_LOCK.acquire(blocking=False):
        log.warning(
            _LOG_PREFIX,
            "Another Docker image or container operation is already running.",
        )
        raise DockerImageManagerBusy(
            "Another Docker image or container operation is already running"
        )
    try:
        yield
    finally:
        _DOCKER_OPERATION_LOCK.release()


def pull_managed_image(
    backend: object,
    vendor: object = "auto",
    runtime_version: object = None,
) -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    normalized_vendor = _normalize_vendor(vendor)
    normalized_runtime_version = (
        runtime_version.strip() if isinstance(runtime_version, str) else runtime_version
    )
    image_reference = _managed_image_reference(
        normalized_backend,
        normalized_vendor,
        runtime_version=normalized_runtime_version,
    )
    backend_label = _BACKENDS[normalized_backend]["label"]
    with _docker_operation():
        _ensure_docker_accessible()
        log.msg(
            _LOG_PREFIX,
            f"Installing or updating {backend_label} image: {image_reference}",
        )
        result = _run_streaming(
            ["docker", "pull", image_reference],
            timeout=3600,
            operation=f"{backend_label} pull",
        )
        if result.returncode != 0:
            log.error(
                _LOG_PREFIX,
                f"{backend_label} image installation failed; Docker details are "
                "available with SmartLLM debug logging enabled.",
            )
            raise DockerImageManagerError(_command_error(result))
        image = _inspect_image(image_reference)
        if not image.get("installed"):
            log.error(
                _LOG_PREFIX,
                f"{backend_label} pull finished, but Docker could not inspect the image.",
            )
            raise DockerImageManagerError("Docker pull completed without an inspectable image")
        image_details = ", ".join(
            detail
            for detail in (image.get("short_id", ""), image.get("size", ""))
            if detail
        )
        detail_suffix = f" ({image_details})" if image_details else ""
        log.msg(
            _LOG_PREFIX,
            f"{backend_label} image is ready{detail_suffix}.",
        )
        if normalized_runtime_version is not None:
            _persist_ollama_runtime_image(image_reference, normalized_vendor)
            log.msg(
                _LOG_PREFIX,
                f"Selected Ollama runtime {normalized_runtime_version}; the managed container "
                "will be recreated on its next start.",
            )
    return {
        "success": True,
        "backend": normalized_backend,
        "vendor": normalized_vendor,
        "image": image_reference,
        "runtime_version": normalized_runtime_version,
        "status": image,
    }


def _containers_using_image(image_id: str) -> list[dict[str, str]]:
    result = _run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"ancestor={image_id}",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Status}}",
        ],
        timeout=15,
    )
    if result.returncode != 0:
        raise DockerImageManagerError(_command_error(result))
    containers = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            containers.append({"id": parts[0], "name": parts[1], "status": parts[2]})
    return containers


def remove_managed_image(backend: object, vendor: object = "auto") -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    normalized_vendor = _normalize_vendor(vendor)
    image_reference = _managed_image_reference(normalized_backend, normalized_vendor)
    backend_label = _BACKENDS[normalized_backend]["label"]
    with _docker_operation():
        _ensure_docker_accessible()
        image = _inspect_image(image_reference)
        if not image.get("installed"):
            log.msg(
                _LOG_PREFIX,
                f"{backend_label} image is already absent; nothing to remove.",
            )
            return {
                "success": True,
                "backend": normalized_backend,
                "vendor": normalized_vendor,
                "image": image_reference,
                "removed": False,
            }
        containers = _containers_using_image(image["image_id"])
        if containers:
            log.warning(
                _LOG_PREFIX,
                f"Cannot remove {backend_label}; {len(containers)} container(s) still "
                "reference the image.",
            )
            raise DockerImageInUse(containers)
        log.msg(_LOG_PREFIX, f"Removing {backend_label} image: {image_reference}")
        result = _run_streaming(
            ["docker", "image", "rm", image_reference],
            timeout=120,
            operation=f"{backend_label} removal",
        )
        if result.returncode != 0:
            log.error(
                _LOG_PREFIX,
                f"{backend_label} image removal failed; Docker details are available "
                "with SmartLLM debug logging enabled.",
            )
            raise DockerImageManagerError(_command_error(result))
        log.msg(_LOG_PREFIX, f"{backend_label} image was removed.")
    return {
        "success": True,
        "backend": normalized_backend,
        "vendor": normalized_vendor,
        "image": image_reference,
        "removed": True,
    }


def stop_managed_containers(
    backend: object,
    vendor: object = "auto",
) -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    normalized_vendor = _normalize_vendor(vendor)
    backend_label = _BACKENDS[normalized_backend]["label"]

    with _docker_operation():
        _ensure_docker_accessible()
        if normalized_backend == "vllm":
            from .backend_vllm_docker import stop_vllm_container

            stop_succeeded = stop_vllm_container()
        elif normalized_backend == "sglang":
            from .backend_sglang_docker import stop_sglang_container

            stop_succeeded = stop_sglang_container()
        elif normalized_backend == "ollama":
            from .backend_ollama_docker import stop_ollama_container

            stop_succeeded = stop_ollama_container()
        else:
            from .backend_llamacpp_docker import stop_llamacpp_container

            stop_succeeded = stop_llamacpp_container()

        if not stop_succeeded:
            log.error(
                _LOG_PREFIX,
                f"Could not stop the managed {backend_label} container(s).",
            )
            raise DockerImageManagerError(
                f"Could not stop managed {backend_label} container(s)"
            )
        log.msg(
            _LOG_PREFIX,
            f"Managed {backend_label} container(s) are stopped.",
        )

    return {
        "success": True,
        "backend": normalized_backend,
        "vendor": normalized_vendor,
        "stopped": True,
    }


__all__ = [
    "DockerImageInUse",
    "DockerImageManagerBusy",
    "DockerImageManagerError",
    "get_docker_manager_overview",
    "normalize_managed_image_selection",
    "pull_managed_image",
    "remove_managed_image",
    "stop_managed_containers",
]
