"""Release retained SmartLLM resources at an explicit workflow boundary."""

from .lifecycle import model_maintenance_if_idle
from .logger import log
from .model_cache import clear_all_model_caches, stop_all_docker_containers


def unload_retained_models() -> None:
    # The node must not enter with_execution_cleanup: maintenance needs an idle
    # inference gate, and reserves it until every backend has been released.
    with model_maintenance_if_idle() as acquired:
        if not acquired:
            raise RuntimeError(
                "SmartLLM models are still in use. Connect Smart LM Unload after "
                "all LM tasks or after the completed loop."
            )
        errors = []
        try:
            clear_all_model_caches()
        except Exception as error:  # noqa: BLE001 - still release Docker resources
            errors.append(error)
        try:
            stop_all_docker_containers(strict=True)
        except Exception as error:  # noqa: BLE001 - preserve both cleanup failures
            errors.append(error)
        if errors:
            raise RuntimeError(
                "Smart LM Unload could not release every retained model. "
                "Check the backend cleanup error before continuing."
            ) from errors[0]
        log.msg("Smart LM Unload", "Retained SmartLLM models unloaded")
