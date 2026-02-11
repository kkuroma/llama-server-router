# LLaMa.cpp HTTP Server Router

A fastapi router and scheduler for single GPU deployment with [LLaMa.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) that supports varying number of instances of multiple LLMs, in the case where only a single model can fit in VRAM at once.
- [] TODO add more scheduler and support for custom schedulers

## Functionalities

- 🌟 **Smart scheduling and swapping of multiple LLMs**: the LRU scheduler prioritizes requests where the requested model is already loaded to VRAM, significantly improving turnaround time.
- 🌟 **Per-LLM number of instance configuration**: allows smaller LLMs to be deployed with multiple instances, maximizing GPU utilization. Uses round robin to distribute request across underlying instances.
- 🌟 **Seamless integration with LLaMa.cpp**: proxies API request directly to background LLaMa.cpp instances, allowing for
- 🌟 **Up-to-date LLM Deployment**:  uses the latest LLaMa.cpp release, allowing newer LLM releases to be deployed as soom as LLaMa.cpp does.

## Usage
The repository contains an example docker build file for a deployment-ready image. Add your `.gguf` LLMs weights to `./llm/models`, then configure `./llm/configs/config.json` and `./llm/configs/presets.ini` accordingly.

## API Endpoints
- `GET /router/` returns the router status.
- `POST /model/load` expects the body `{"model_id": model_id}` and returns `{"success": true/false}`, loads the requested model to VRAM while unloading others (if not already serving other requests).
- `POST /model/unload` expects the body `{"model_id": model_id}` and returns `{"success": true/false}`, unloads the requested model if loaded to VRAM.