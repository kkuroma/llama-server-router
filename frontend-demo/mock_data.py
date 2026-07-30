"""
Static fake-fleet definition for the frontend demo harness.

This is pure data, no logic. `launch.py` reads it to synthesize the router API
the real SPAs poll (`/router`, `/router/models`, `/router/gpu`, ...). Edit the
lists here to stress different shapes: more GPUs, wider VRAM spreads, longer
model names, MoE active-param tags, models with no size token, etc.

Nothing here spawns a process; there is no llama-server. The values are only
believable enough to exercise every dashboard widget.
"""

# Physical GPUs advertised to the dashboard. `total_vram_mb` drives the VRAM
# chart's y-axis and the "near cap" red line; deliberately mixed sizes so the
# per-GPU selector, legend and VRAM scaling all get exercised.
GPUS = [
    {"index": 0, "name": "NVIDIA GeForce RTX 4080 SUPER", "total_vram_mb": 16376, "power_limit_w": 320},
    {"index": 1, "name": "NVIDIA GeForce RTX 4090",       "total_vram_mb": 24564, "power_limit_w": 450},
    {"index": 2, "name": "NVIDIA GeForce RTX 4080 SUPER", "total_vram_mb": 16376, "power_limit_w": 320},
    {"index": 3, "name": "NVIDIA GeForce RTX 3090",       "total_vram_mb": 24576, "power_limit_w": 350},
]

# Configured models. `gpus` is the fixed placement (must reference indices in
# GPUS); `vram_mb` is the model's TOTAL resident footprint, split evenly across
# its GPUs by the sim; `tps` is a nominal decode speed used to fabricate
# believable request durations. The id set intentionally covers every tag shape
# the dashboard parser handles: plain dense (`27b`), MoE (`35b` + `a3b`),
# trailing descriptors (`uncensored`), size-less names (`glm-4-7-flash`,
# `laguna-s-2-1`), a 4-GPU spanner, and small single-GPU models.
MODELS = [
    {"id": "qwen-3-6-35b-a3b",              "gpus": [0, 2],       "vram_mb": 22000, "tps": 95},
    {"id": "qwen-3-6-35b-a3b-uncensored",   "gpus": [0, 2],       "vram_mb": 22000, "tps": 95},
    {"id": "gemma-4-12b",                   "gpus": [1],          "vram_mb": 11000, "tps": 70},
    {"id": "gemma-4-12b-uncensored",        "gpus": [1],          "vram_mb": 11000, "tps": 70},
    {"id": "glm-4-7-flash",                 "gpus": [0, 2],       "vram_mb": 20000, "tps": 110},
    {"id": "thinkingcap-qwen-3-6-27b",      "gpus": [0, 2],       "vram_mb": 24000, "tps": 45},
    {"id": "wordslop-qwen-3-6-27b",         "gpus": [0, 2],       "vram_mb": 26000, "tps": 40},
    {"id": "gemma-4-26b-a4b",               "gpus": [0, 2],       "vram_mb": 18000, "tps": 80},
    {"id": "gemma-4-26b-a4b-uncensored",    "gpus": [0, 2],       "vram_mb": 18000, "tps": 80},
    {"id": "gemma-4-31b",                   "gpus": [0, 2],       "vram_mb": 25000, "tps": 50},
    {"id": "gemma-4-31b-uncensored",        "gpus": [0, 2],       "vram_mb": 25000, "tps": 50},
    {"id": "laguna-s-2-1",                  "gpus": [0, 1, 2],    "vram_mb": 33000, "tps": 30},
    {"id": "deepseek-r1-671b-a37b",         "gpus": [0, 1, 2, 3], "vram_mb": 60000, "tps": 22},
    {"id": "mistral-nemo-12b",              "gpus": [3],          "vram_mb": 9000,  "tps": 85},
    {"id": "phi-4-mini-3.8b-instruct",      "gpus": [3],          "vram_mb": 4200,  "tps": 140},
]

# Models resident when the demo boots (a disjoint set that lights every GPU so
# the first paint is already busy). Everything else the sim loads/unloads over
# time. Must be a subset of MODELS ids with non-overlapping GPUs.
INITIAL_LOADED = ["wordslop-qwen-3-6-27b", "gemma-4-12b", "phi-4-mini-3.8b-instruct"]

# Scheduler knobs echoed back on /router (cosmetic: the dashboard displays them
# but the sim enforces its own single-model-per-GPU residency regardless).
MAX_MODELS_PER_GPU = 1
EVICTION_POLICY = "lru"
LLM_BASE_PORT = 30000
