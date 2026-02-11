# llama-router: a cache-aware LLaMa.cpp scheduler

A FastAPI router and scheduler for single GPU deployment with [LLaMa.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) that supports varying number of instances of multiple LLMs, in the case where only a single model can fit in VRAM at once. It intelligently schedules requests to maximize cache hits and GPU utilizations by prioritizing request to GPU already loaded to VRAM, while swapping if necessary.

## Functionalities

- 🌟 **Cache-aware Scheduling:** Drains request of already loaded models before processing other request (unless waited for too long) to avoid unnecessary model swaps.
- 🌟 **Easy swapping of multiple LLMs**: Automatically swaps the loaded model if not serving without requiring separate `/load` or `/unload` calls.
- 🌟 **Per-LLM number of instance configuration**: allows smaller LLMs to be deployed with multiple instances, maximizing GPU utilization. Uses round robin to distribute request across underlying instances. Seamlessly use one instance to deploy a 32B model and switch to 4 instances of 4B models in an instant!
- 🌟 **Seamless integration with LLaMa.cpp**: proxies API request directly to background LLaMa.cpp instances, being a drop in replacement for `llama-server`
- 🌟 **Up-to-date LLM deployment**:  uses the latest LLaMa.cpp release, allowing newer LLM releases to be deployed as soom as LLaMa.cpp does.
- 🌟 **Text-based configuration**: allows configuration to be stored in persistent storage, ideal for deployment!

## Why llama-router
`llama-router` supports number of instances to be declared per-model and automatically proxies and distributes workloads to underlying services. These are additional benefits over commonly used LLM deployment engine:
- **VS llama-server**: Hot swapping, implements scheduling not present in `llama-server`
- **VS ollama**: Direct LLaMa.cpp integration means greater customizability, cache-aware scheduler outperforms ollama's FIFO
- **VS llama-swap (go)**: Python-based (easier to configure without recompiling), allows a wider range of custom schedulers

## Usage
**Requirements** Docker and NVIDIA-cuda-container. It is recommended and required to serve `llama-router` via 

The repository contains an example docker build file for a deployment-ready image. Add your `.gguf` LLMs weights to `./llm/models`, then configure `./llm/configs/config.json` and `./llm/configs/presets.ini` accordingly.

## API endpoints
- `GET /router/` returns the router status.
- `POST /model/load` expects the body `{"model_id": model_id}` and returns `{"success": true/false}`, loads the requested model to VRAM while unloading others (if not already serving other requests).
- `POST /model/unload` expects the body `{"model_id": model_id}` and returns `{"success": true/false}`, unloads the requested model if loaded to VRAM.\

## Future Plans
- [] Implement other scheduler types
- [] Add support for other hardware configuration (currently supports NVIDIA GPUs)
- [] Enable multi-GPU support