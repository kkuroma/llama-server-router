import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from router import LLMRouter

app = FastAPI()

# Shared web UI assets (fonts + design-language CSS/JS) for /dash and /translate.
# Mounted before the catch-all proxy so /webui/* is served locally, not forwarded.
app.mount("/webui", StaticFiles(directory=Path(__file__).parent / "webui"), name="webui")
router: LLMRouter | None = None
gpu_monitor = None  # set by main.py
status_timeline = None  # set by main.py
translation_service = None  # set by main.py


def getRouter() -> LLMRouter:
    """
    Returns the initialized router or raises if it is not ready yet

    Returns:
        The active LLMRouter instance

    Raises:
        HTTPException: 503 if the router has not been initialized
    """
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return router

# Custom endpoints for the router

@app.get("/")
async def root():
    """Lands on the dashboard instead of falling into the proxy catch-all."""
    return RedirectResponse(url="/dash")

@app.get("/router")
async def routerStatus():
    """
    Returns the current router status and the ports of all live subprocesses
    """
    r = getRouter()
    return {
        "status": r.status.value,
        "ports": sorted(r.processes.keys()),
        "instances": {str(p): m for p, m in sorted(r.port_model.items())},
        "num_gpus": r.num_gpus,
        "max_models_per_gpu": r.MAX_MODELS_PER_GPU,
        "eviction_policy": r.EVICTION_POLICY,
        "model_gpus": r.model_gpus,
    }

@app.get("/router/start")
async def routerStart():
    """Starts the router if it is inactive or errored."""
    r = getRouter()
    if not r.status.value in ["inactive", "error"]:
        return {"success": True, "status": r.status.value}
    await r.start()
    return {"success": True, "status": r.status.value}

@app.get("/router/stop")
async def routerStop():
    """Stops the router unless it is already inactive or errored."""
    r = getRouter()
    if r.status.value in ["inactive", "error"]:
        return {"success": True, "status": r.status.value}
    await r.stop()
    return {"success": True, "status": r.status.value}

@app.get("/router/restart")
async def routerRestart():
    """Restarts the router by stopping and starting again."""
    r = getRouter()
    await r.restart()
    return {"success": True, "status": r.status.value}

# overwrites /model/load /model/unload
@app.post("/models/unload")
async def routerUnload(body: dict):
    """
    Unloads a model by killing all of its replica processes

    Args:
        body (dict): Request body expected to contain {"model": "<model_id>"}

    Raises:
        HTTPException: 422 if "model" is missing, 404 if the model is unknown
    """
    r = getRouter()
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=422, detail="Missing 'model' field")
    if not(model_id in r.router_config["LLM"].keys()):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    await r.unloadModel(model_id)
    return {"success": True}

@app.post("/models/load")
async def routerLoad(body: dict):
    """
    Loads a model, spawning its replica processes and evicting as needed

    Args:
        body (dict): Request body expected to contain {"model": "<model_id>"}

    Raises:
        HTTPException: 422 if "model" is missing, 404 if the model is unknown
    """
    r = getRouter()
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=422, detail="Missing 'model' field")
    if not(model_id in r.router_config["LLM"].keys()):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    result = await r.loadModel(model_id)
    return {"success": result}


# History endpoints

@app.get("/router/history")
async def routerHistory(
    model: str | None = None,
    limit: int = 10000,
    since: float | None = None,
    until: float | None = None,
):
    """
    Returns request history rows within a time window, optionally filtered by model

    Args:
        model (str | None)  : The model to filter by, or None for all models
        limit (int)         : Max rows to return, clamped to [0, 100000]; 0 means everything
        since (float | None): Only rows with request_time >= this unix ts, if set
        until (float | None): Only rows with request_time <= this unix ts, if set

    Returns:
        The list of history rows, most recent first
    """
    r = getRouter()
    limit = max(0, min(limit, 100000))
    return await r.getHistory(model, limit, since, until)

@app.get("/router/reset_history")
async def routerResetHistory():
    """Clears all request history."""
    r = getRouter()
    await r.resetHistory()
    return {"success": True}

# Dashboard

@app.get("/dash")
async def dashboard():
    """
    Serves the dashboard single-page app

    Raises:
        HTTPException: 404 if the dashboard HTML is missing
    """
    html_path = Path(__file__).parent / "dash" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(html_path.read_text())

# Chat (themed shell + embedded llama.cpp UIs)

@app.get("/chat")
async def chatPage():
    """
    Serves the chat single-page app

    The shell (navbar + Aa popover + one tab per instance) is ours; each tab
    embeds a live llama-server replica's own web UI via /instance/{port}/ — a
    same-origin pass-through proxy that forwards straight to that replica and
    bypasses the scheduler/RWLock. Same-origin (not the raw instance port) is
    what makes it work over Tailscale and behind a reverse proxy; bypassing the
    lock is what keeps the embedded UI's polling from starving the write-lock
    unloadModel needs (which would leave the router stuck "serving").

    Raises:
        HTTPException: 404 if the chat HTML is missing
    """
    html_path = Path(__file__).parent / "chat" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Chat app not found")
    return HTMLResponse(html_path.read_text())

# Injected into the proxied llama.cpp index so its ABSOLUTE API paths (/v1/*,
# /props, ...) get rewritten to /instance/{port}/* and stay pinned to this
# replica (its relative ./_app/* assets already resolve under the prefix). The
# __PORT__ token is substituted per request. Without this the UI's API calls
# would escape the prefix back to the router root — re-entering the scheduler
# and the very RWLock starvation this design avoids.
_INSTANCE_SHIM = """<script>
(function () {
  var P = '/instance/__PORT__';
  function rw(u) {
    if (typeof u !== 'string') return u;
    var s = u;
    if (s.indexOf(location.origin) === 0) s = s.slice(location.origin.length);
    if (s.charAt(0) !== '/') return u;              // relative or cross-origin — leave
    if (s === P || s.indexOf(P + '/') === 0) return u;  // already prefixed
    return P + s;
  }
  if (window.fetch) {
    var of = window.fetch;
    window.fetch = function (input, init) {
      try {
        if (typeof input === 'string') input = rw(input);
        else if (input && input.url) { var n = rw(input.url); if (n !== input.url) input = new Request(n, input); }
      } catch (e) {}
      return of.call(this, input, init);
    };
  }
  if (window.EventSource) {
    var OE = window.EventSource;
    window.EventSource = function (u, c) { return new OE(rw(u), c); };
    window.EventSource.prototype = OE.prototype;
  }
  if (window.XMLHttpRequest) {
    var oo = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function (m, u) {
      try { u = rw(u); } catch (e) {}
      return oo.call(this, m, u, arguments[3], arguments[4], arguments[5]);
    };
  }
})();
</script>"""

@app.api_route("/instance/{port}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def instanceProxy(port: int, path: str, request: Request):
    """
    Same-origin pass-through proxy to a single llama-server replica

    Forwards the request straight to http://127.0.0.1:{port}/{path} WITHOUT
    touching router.addRequest / the RWLock, so the embedded chat UI's polling
    and streaming never take reader slots that would starve unload. Keeping it on
    the router's own port (rather than the raw instance port) is what lets it work
    over Tailscale and behind a reverse proxy. The instance's index HTML is
    buffered so the API-path shim can be injected; everything else streams through
    (SSE included, so tok/s tick live).

    Args:
        port (int)          : The replica port; must be a live instance
        path (str)          : The remaining path to forward to that replica
        request (Request)   : The incoming request to forward

    Raises:
        HTTPException: 404 if no live instance owns that port, 502 if it is unreachable
    """
    r = getRouter()
    if port not in r.processes:
        raise HTTPException(status_code=404, detail=f"No live instance on port {port}")

    url = f"http://127.0.0.1:{port}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body = await request.body()

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = client.build_request(request.method, url, content=body, headers=fwd_headers)
        resp = await client.send(upstream, stream=True)
    except Exception as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Instance {port} unreachable: {exc}")

    content_type = resp.headers.get("content-type", "")

    # HTML index: buffer, decompress (httpx decodes .aread()), inject the shim.
    if "text/html" in content_type:
        raw = await resp.aread()
        status = resp.status_code
        await resp.aclose()
        await client.aclose()
        text = raw.decode("utf-8", "replace")
        shim = _INSTANCE_SHIM.replace("__PORT__", str(port))
        # Inject right after the opening <head> so it runs before the app bundle.
        new_text, n = re.subn(r"(<head[^>]*>)", r"\1" + shim, text, count=1, flags=re.IGNORECASE)
        text = new_text if n else shim + text
        return HTMLResponse(text, status_code=status)

    # Everything else streams through raw (content-encoding preserved).
    safe_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "connection", "keep-alive")}

    async def streamUpstream():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        streamUpstream(),
        status_code=resp.status_code,
        media_type=content_type or None,
        headers=safe_headers,
    )

@app.get("/router/models")
async def routerModels():
    """
    Returns all configured models with their current load status

    Synthesized from router state since ports are per-model now

    Returns:
        An OpenAI-style list with each model's status, gpus, and live ports
    """
    r = getRouter()
    loaded = await r.getLoadedModels()
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "status": {"value": "loaded" if mid in loaded else "unloaded"},
                "gpus": r.model_gpus.get(mid, [0]),
                "ports": r._modelPorts(mid),
            }
            for mid in r.router_config["LLM"]
        ],
    }

@app.get("/v1/models")
async def v1Models():
    """
    Returns an OpenAI-compatible model listing served by the router itself

    Works even when no llama-server replica is running

    Returns:
        An OpenAI-style list of the configured model ids
    """
    r = getRouter()
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": created,
                "owned_by": "llama-router",
            }
            for mid in r.router_config["LLM"]
        ],
    }

@app.get("/router/gpu")
async def routerGpu():
    """Returns per-GPU utilization and VRAM history for every detected GPU."""
    if gpu_monitor is None:
        return {"error": "GPU monitor not available", "gpus": []}
    return {"gpus": gpu_monitor.snapshot()}

@app.get("/router/status_timeline")
async def routerStatusTimeline():
    """Returns the router status change timeline."""
    if status_timeline is None:
        return {"entries": []}
    return {"entries": status_timeline.entries}

# Translation

@app.get("/translate")
async def translatePage():
    """
    Serves the translation single-page app

    Raises:
        HTTPException: 404 if the translate HTML is missing
    """
    html_path = Path(__file__).parent / "translate" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Translate app not found")
    return HTMLResponse(html_path.read_text())

@app.get("/router/languages")
async def routerLanguages(lang: str | None = None):
    """
    Returns the language list, optionally filtered by language name

    Args:
        lang (str | None): The language name to match, or None for all

    Raises:
        HTTPException: 503 if the translation service is unavailable
    """
    if translation_service is None:
        raise HTTPException(status_code=503, detail="Translation service not available")
    return translation_service.getLanguages(lang)

@app.post("/router/translate")
async def routerTranslate(request: Request):
    """
    Translates text with a model routed through the request queue

    Validates the model and languages, builds cache-friendly messages, enqueues
    the request, and streams or returns the result

    Args:
        request (Request): JSON body with model_id, source_id, target_id, text, additionals?, stream?

    Raises:
        HTTPException: 503 if translation is unavailable, 422 on bad fields, 404 on unknown model
    """
    import json as _json
    r = getRouter()
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

    messages = translation_service.buildMessages(source_id, target_id, text, additionals)

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

    future = await r.addRequest(envelope)
    result = await future

    if is_streaming:
        async def streamChunks():
            while True:
                chunk = await result.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        return StreamingResponse(streamChunks(), media_type="text/event-stream")
    else:
        return JSONResponse(content=result.json(), status_code=result.status_code)

# proxies everything else to the backend

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(full_path: str, request: Request):
    """
    Proxies all remaining requests through the router

    Wraps the request into an envelope, detects streaming from the body's
    "stream" field, and streams back with SSE or passes the JSON through

    Args:
        full_path (str)     : The path segment captured by the catch-all route
        request (Request)   : The incoming request to forward

    Returns:
        A streaming, JSON, or raw response mirroring the upstream reply
    """
    r = getRouter()
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
    future = await r.addRequest(envelope)
    try:
        result = await future
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": str(e),
                    "code": "upstream_error"
                }
            }
        )

    if is_streaming:
        async def streamChunks():
            while True:
                chunk = await result.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        return StreamingResponse(streamChunks(), media_type="text/event-stream")
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
