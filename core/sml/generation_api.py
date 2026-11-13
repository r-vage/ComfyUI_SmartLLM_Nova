# Small request boundary for execution-scoped Stop LM controls.

import asyncio
import re

from aiohttp import web

from .generation_control import generation_status, stop_generation


def register(routes, read_json, mutation_denial):
    @routes.get("/smartllm-nova/generation")
    async def status(request):
        return web.json_response(generation_status(), headers={"Cache-Control": "no-store"})

    @routes.post("/smartllm-nova/generation/stop")
    async def stop(request):
        denial = mutation_denial(request)
        if denial is not None:
            return denial
        data = await read_json(request)
        run_id = data.get("run_id")
        if set(data) != {"run_id"} or not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return web.json_response({"success": False, "error": "A valid LM execution ID is required."}, status=400)
        try:
            result = await asyncio.to_thread(stop_generation, run_id)
        except LookupError as error:
            return web.json_response({"success": False, "error": str(error)}, status=409)
        except RuntimeError as error:
            return web.json_response({"success": False, "error": str(error)}, status=503)
        return web.json_response(result)
