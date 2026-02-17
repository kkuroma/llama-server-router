import time
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response, HTMLResponse
from router import LLMRouter

app = FastAPI()
router: LLMRouter | None = None
gpu_monitor = None  # set by main.py
status_timeline = None  # set by main.py
translation_service = None  # set by main.py

def get_router() -> LLMRouter:
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return router

# Custom endpoints for the router

@app.get("/router")
async def router_status():
    '''
        Return current router status and the ports of all live subprocesses
    '''
    r = get_router()
    return {
        "status": r.status.value,
        "ports": sorted(r.processes.keys()),
    }

@app.get("/router/start")
async def router_start():
    r = get_router()
    if not r.status.value in ["inactive", "error"]:
        return {"success": True, "status": r.status.value}
    await r.start()
    return {"success": True, "status": r.status.value}

@app.get("/router/stop")
async def router_stop():
    r = get_router()
    if r.status.value in ["inactive", "error"]:
        return {"success": True, "status": r.status.value}
    await r.stop()
    return {"success": True, "status": r.status.value}

@app.get("/router/restart")
async def router_restart():
    r = get_router()
    await r.restart()
    return {"success": True, "status": r.status.value}

# overwrites /model/load /model/unload
@app.post("/models/unload")
async def router_unload(body: dict):
    '''
        Signal llamacpp's /models/unload to ALL instances
        Expects body: {"model": "<model_id>"}
    '''
    r = get_router()
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=422, detail="Missing 'model' field")
    if not(model_id in r.router_config["LLM"].keys()):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    await r.unload_model(model_id)
    return {"success": True}

@app.post("/models/load")
async def router_load(body: dict):
    '''
        Signal llamacpp's /models/load to ALL instances, adjusting the number of instances accordingly
        Expects body: {"model": "<model_id>"}
    '''
    r = get_router()
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=422, detail="Missing 'model' field")
    if not(model_id in r.router_config["LLM"].keys()):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    result = await r.load_model(model_id)
    return {"success": result}


# History endpoints

@app.get("/router/history")
async def router_history(model: str | None = None):
    '''Return request history, optionally filtered by model'''
    r = get_router()
    return await r.get_history(model)

@app.get("/router/reset_history")
async def router_reset_history():
    '''Clear all request history'''
    r = get_router()
    await r.reset_history()
    return {"success": True}

# Dashboard

@app.get("/dash")
async def dashboard():
    '''Serve the dashboard SPA'''
    html_path = Path(__file__).parent / "dash" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(html_path.read_text())

@app.get("/router/models")
async def router_models():
    '''Return all configured models with their current load status'''
    r = get_router()
    if not r.processes:
        return {"models": []}
    port = next(iter(r.processes.keys()))
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://0.0.0.0:{port}/models", timeout=5.0)
            return resp.json()
    except Exception:
        return {"models": []}

@app.get("/router/gpu")
async def router_gpu():
    '''Return GPU utilization and VRAM history'''
    if gpu_monitor is None:
        return {"error": "GPU monitor not available"}
    return {
        "total_vram_mb": gpu_monitor.total_vram_mb,
        "util_history": gpu_monitor.util_history,
        "vram_history": gpu_monitor.vram_history,
    }

@app.get("/router/status_timeline")
async def router_status_timeline():
    '''Return router status change timeline'''
    if status_timeline is None:
        return {"entries": []}
    return {"entries": status_timeline.entries}

# Translation

@app.get("/translate")
async def translate_page():
    '''Serve the translation SPA'''
    html_path = Path(__file__).parent / "translate" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Translate app not found")
    return HTMLResponse(html_path.read_text())

@app.get("/router/languages")
async def router_languages(lang: str | None = None):
    '''Return language list, optionally filtered by lang_name'''
    if translation_service is None:
        raise HTTPException(status_code=503, detail="Translation service not available")
    return translation_service.get_languages(lang)

@app.post("/router/translate")
async def router_translate(request: Request):
    '''Translate text using a model via the router queue.
    Body: {model_id, source_id, target_id, text, additionals?, stream?}'''
    import json as _json
    r = get_router()
    if translation_service is None:
        raise HTTPException(status_code=503, detail="Translation service not available")

    body = await request.json()
    model_id = body.get("model_id")
    source_id = body.get("source_id")
    target_id = body.get("target_id")
    text = body.get("text")
    additionals = body.get("additionals", "")
    is_streaming = body.get("stream", False)

    if not all([model_id, source_id, target_id, text]):
        raise HTTPException(status_code=422, detail="Missing required fields: model_id, source_id, target_id, text")
    if model_id not in r.router_config["LLM"]:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    if source_id not in translation_service.lang_map:
        raise HTTPException(status_code=422, detail=f"Invalid source language: {source_id}")
    if target_id not in translation_service.lang_map:
        raise HTTPException(status_code=422, detail=f"Invalid target language: {target_id}")

    messages = translation_service.build_messages(source_id, target_id, text, additionals)

    envelope = {
        "path": "/v1/chat/completions",
        "method": "POST",
        "body": _json.dumps({
            "model": model_id,
            "messages": messages,
            "cache_prompt": True,
            "stream": is_streaming,
        }),
        "headers": {"Content-Type": "application/json"},
        "model": model_id,
        "is_streaming": is_streaming,
    }

    future = await r.add_request(envelope)
    result = await future

    if is_streaming:
        async def stream_chunks():
            while True:
                chunk = await result.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        return StreamingResponse(stream_chunks(), media_type="text/event-stream")
    else:
        return JSONResponse(content=result.json(), status_code=result.status_code)

# proxies everything else to the backend

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(full_path: str, request: Request):
    '''
        Proxies all requests through the router.
        If the upstream request wants streaming (body has "stream": true), stream back with SSE
        Otherwise passes the JSON through
    '''
    r = get_router()
    if r.status.value in ["inactive", "error"]:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"The server is currently inactive or down.",
                    "code": "service_unavailable"
                }
            }
        )

    raw_body = await request.body()
    query_string = request.url.query
    path_with_query = f"/{full_path}"
    if query_string:
        path_with_query += f"?{query_string}"
    envelope = {
        "path": path_with_query,
        "method": request.method,
        "body": raw_body,
        "headers": dict(request.headers),
        "is_streaming": False,
    }

    # detect streaming: try to parse JSON and check "stream" field
    try:
        import json
        parsed = json.loads(raw_body)
        envelope["is_streaming"] = parsed.get("stream", False)
        # include parsed model in envelope so the router can check it
        if "model" in parsed:
            envelope["model"] = parsed["model"]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # model doesn't exist
    requested_model = envelope.get("model", None)
    if not(requested_model is None) and not(requested_model in r.router_config["LLM"].keys()):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model `{requested_model}` does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        )

    is_streaming = envelope["is_streaming"]

    # enqueue and await result
    future = await r.add_request(envelope)
    result = await future

    if is_streaming:
        async def stream_chunks():
            while True:
                chunk = await result.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        return StreamingResponse(stream_chunks(), media_type="text/event-stream")
    else:
        content_type = result.headers.get("Content-Type", "")
        safe_headers = {
            k: v for k, v in result.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")
        }
        if "application/json" in content_type:
            try:
                return JSONResponse(
                    content=result.json(),
                    status_code=result.status_code,
                    headers=safe_headers,
                )
            except Exception:
                pass

        return Response(
            content=result.content,
            status_code=result.status_code,
            media_type=content_type,
            headers=safe_headers,
        )
