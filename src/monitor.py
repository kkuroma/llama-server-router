import time
from typing import Any, TypedDict, cast

try:
    import pynvml
except ImportError:
    pynvml = None

WINDOW_SECONDS = 7200  # 2 hours
FLUSH_INTERVAL = 10.0  # flush every 10s for higher resolution


class GpuSnapshot(TypedDict):
    """One GPU's JSON-serializable history, as returned to the dashboard."""
    index: int
    name: str
    total_vram_mb: float
    power_limit_w: float
    util_history: list[tuple[float, float]]
    vram_history: list[tuple[float, float]]
    temp_history: list[tuple[float, float]]
    power_history: list[tuple[float, float]]


class GPUMonitor:
    """
    Samples every NVIDIA GPU's utilization, VRAM, temperature, and power draw
    once per second, keeping a rolling per-GPU window

    Detects all devices via NVML at startup and, for each, buffers per-second
    samples and flushes their maxima every FLUSH_INTERVAL to keep history compact.
    All GPUs flush on one shared cadence so their timestamps stay aligned.
    Temp/power reads are optional per device, so an unsupported sensor leaves
    that history empty.
    """

    def __init__(self):
        if pynvml is None:
            raise RuntimeError("pynvml is not installed")
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            raise RuntimeError("no NVIDIA GPUs detected")
        # One state dict per GPU. The _samples lists buffer the current interval.
        self.gpus: list[dict[str, Any]] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Name is cosmetic, so never let a pynvml quirk abort the whole monitor.
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):  # older pynvml returns bytes
                    name = name.decode()
            except Exception:
                name = f"GPU {i}"
            try:
                power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                power_limit_w = 0.0
            self.gpus.append({
                "index": i,
                "handle": handle,
                "name": name,
                "total_vram_mb": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total) / (1024 ** 2),
                "power_limit_w": power_limit_w,
                "util_history": [],   # (unix_ts, percent)
                "vram_history": [],   # (unix_ts, used_mb)
                "temp_history": [],   # (unix_ts, deg_c)
                "power_history": [],  # (unix_ts, watts)
                "_util_samples": [],
                "_vram_samples": [],
                "_temp_samples": [],
                "_power_samples": [],
            })
        self._last_flush_time: float = time.time()

    @property
    def gpu_count(self) -> int:
        """Returns the number of detected GPUs."""
        return len(self.gpus)

    def poll(self):
        """
        Records one util/VRAM/temp/power sample per GPU, flushing maxima on the interval

        Called every 1s from a background thread; on flush each GPU appends its
        window maxima and drops history older than WINDOW_SECONDS
        """
        assert pynvml is not None  # GPUMonitor is only constructed when pynvml imported
        for g in self.gpus:
            util = pynvml.nvmlDeviceGetUtilizationRates(g["handle"])
            mem = pynvml.nvmlDeviceGetMemoryInfo(g["handle"])
            g["_util_samples"].append(util.gpu)
            g["_vram_samples"].append(int(mem.used) / (1024 ** 2))
            try:
                g["_temp_samples"].append(pynvml.nvmlDeviceGetTemperature(g["handle"], pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass  # sensor unsupported on this device
            try:
                g["_power_samples"].append(pynvml.nvmlDeviceGetPowerUsage(g["handle"]) / 1000.0)
            except Exception:
                pass

        now = time.time()
        if now - self._last_flush_time >= FLUSH_INTERVAL:
            cutoff = now - WINDOW_SECONDS
            for g in self.gpus:
                for metric in ("util", "vram", "temp", "power"):
                    samples = g[f"_{metric}_samples"]
                    if samples:
                        g[f"{metric}_history"].append((now, round(max(samples), 1)))
                        samples.clear()
                    g[f"{metric}_history"] = [(t, v) for t, v in g[f"{metric}_history"] if t > cutoff]
            self._last_flush_time = now

    def snapshot(self) -> list[GpuSnapshot]:
        """
        Returns the JSON-serializable per-GPU histories for the API

        Returns:
            list[GpuSnapshot]: one entry per GPU with index, name, total_vram_mb,
                power_limit_w, and the rolling windows (NVML handles omitted)
        """
        # self.gpus holds dynamic NVML-derived values, so widen before the cast.
        return [
            cast(GpuSnapshot, cast(object, {
                "index": g["index"],
                "name": g["name"],
                "total_vram_mb": g["total_vram_mb"],
                "power_limit_w": g["power_limit_w"],
                "util_history": g["util_history"],
                "vram_history": g["vram_history"],
                "temp_history": g["temp_history"],
                "power_history": g["power_history"],
            }))
            for g in self.gpus
        ]


class StatusTimeline:
    """
    Records per-GPU router status changes over a rolling window

    Keeps one independent timeline per GPU index. For each GPU it only appends
    when that GPU's status actually changes, dropping entries older than
    WINDOW_SECONDS on each change.
    """

    def __init__(self):
        self.entries: dict[int, list[tuple[float, str]]] = {}  # gpu -> [(unix_ts, status_value), ...]
        self._last_status: dict[int, str] = {}

    def record(self, statuses: dict[int, str]):
        """
        Appends an entry per GPU whose status differs from its previous value

        Args:
            statuses (dict[int, str]): The current {gpu_index: status_value} map
        """
        now = time.time()
        cutoff = now - WINDOW_SECONDS
        for gpu, status_value in statuses.items():
            if status_value != self._last_status.get(gpu):
                lane = self.entries.setdefault(gpu, [])
                lane.append((now, status_value))
                self._last_status[gpu] = status_value
                self.entries[gpu] = [(t, s) for t, s in lane if t > cutoff]
