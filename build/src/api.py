from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from router import LLMRouter

app = FastAPI()
router: LLMRouter | None = None

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
    result = await r.load_model(model_id)
    return {"success": result}


# proxies everything else to the backend

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(full_path: str, request: Request):
    '''
        Proxies all requests through the router. 
        If the upstream request wants streaming (body has "stream": true), stream back with SSE
        Otherwise passes the JSON through
    '''
    r = get_router()
    raw_body = await request.body()
    envelope = {
        "path": f"/{full_path}",
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
        return JSONResponse(
            content=result.json(),
            status_code=result.status_code,
            headers={
                k: v for k, v in result.headers.items()
                if k.lower() not in ("content-length", "transfer-encoding")
            },
        )
