# Optional, client-scoped notices for reviewed execution warnings.
# Console logging remains in logger.py and never depends on a connected browser.

def send_execution_notice(summary: str, message: str, severity: str) -> None:
    if severity not in {"warn", "error"}:
        return
    try:
        from comfy_execution.utils import get_executing_context
        from server import PromptServer

        context = get_executing_context()
        server = PromptServer.instance
        client_id = getattr(server, "client_id", None)
        if context is None or not client_id:
            return
        server.send_sync(
            "smartllm-nova/notification",
            {
                "summary": summary[:200],
                "message": message[:1200],
                "severity": severity,
                "node_id": context.node_id,
                "prompt_id": context.prompt_id,
            },
            client_id,
        )
    except Exception:  # noqa: BLE001 - optional UI must not affect execution
        return

