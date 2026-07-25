import time

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

WINDOW_SECONDS = 7200  # 2 hours
FLUSH_INTERVAL = 10.0  # flush every 10s for higher resolution


class GPUMonitor:
    """
    Samples every NVIDIA GPU's utilization and VRAM once per second, keeping a
    rolling per-GPU window

    Detects all devices via NVML at startup and, for each, buffers per-second
    samples and flushes their maxima every FLUSH_INTERVAL to keep history compact.
    All GPUs flush on one shared cadence so their timestamps stay aligned.
    """

    def __init__(self):
        if not _PYNVML_AVAILABLE:
            raise RuntimeError("pynvml is not installed")
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetDeviceCount()
        if count == 0:
            raise RuntimeError("no NVIDIA GPUs detected")
        # One state dict per GPU. util_history/vram_history are the flushed rolling
        # windows; the _samples lists buffer the current interval's per-second reads.
        self.gpus: list[dict] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Name is cosmetic — never let a driver/pynvml quirk here abort the
            # whole monitor (that would drop GPU detection entirely).
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):  # older pynvml returns bytes
                    name = name.decode()
            except Exception:
                name = f"GPU {i}"
            self.gpus.append({
                "index": i,
                "handle": handle,
                "name": name,
                "total_vram_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 2),
                "util_history": [],   # (unix_ts, percent)
                "vram_history": [],   # (unix_ts, used_mb)
                "_util_samples": [],
                "_vram_samples": [],
            })
        self._last_flush_time: float = time.time()

    @property
    def gpu_count(self) -> int:
        """Returns the number of detected GPUs."""
        return len(self.gpus)

    def poll(self):
        """
        Records one utilization/VRAM sample per GPU, flushing maxima on the interval

        Called every 1s from a background thread; on flush each GPU appends its
        window maxima and drops history older than WINDOW_SECONDS
        """
        for g in self.gpus:
            util = pynvml.nvmlDeviceGetUtilizationRates(g["handle"])
            mem = pynvml.nvmlDeviceGetMemoryInfo(g["handle"])
            g["_util_samples"].append(util.gpu)
            g["_vram_samples"].append(mem.used / (1024 ** 2))

        now = time.time()
        if now - self._last_flush_time >= FLUSH_INTERVAL:
            cutoff = now - WINDOW_SECONDS
            for g in self.gpus:
                if g["_util_samples"]:
                    g["util_history"].append((now, round(max(g["_util_samples"]), 1)))
                    g["vram_history"].append((now, round(max(g["_vram_samples"]), 1)))
                    g["_util_samples"].clear()
                    g["_vram_samples"].clear()
                g["util_history"] = [(t, v) for t, v in g["util_history"] if t > cutoff]
                g["vram_history"] = [(t, v) for t, v in g["vram_history"] if t > cutoff]
            self._last_flush_time = now

    def snapshot(self) -> list[dict]:
        """
        Returns the JSON-serializable per-GPU histories for the API

        Returns:
            list[dict]: one entry per GPU with index, name, total_vram_mb, and the
                util_history / vram_history rolling windows (NVML handles omitted)
        """
        return [
            {
                "index": g["index"],
                "name": g["name"],
                "total_vram_mb": g["total_vram_mb"],
                "util_history": g["util_history"],
                "vram_history": g["vram_history"],
            }
            for g in self.gpus
        ]


class StatusTimeline:
    """
    Records router status changes over a rolling window

    Only appends when the status actually changes, dropping entries older than
    WINDOW_SECONDS on each change
    """

    def __init__(self):
        self.entries: list[tuple[float, str]] = []  # (unix_ts, status_value)
        self._last_status: str | None = None

    def record(self, status_value: str):
        """
        Appends a status entry when the value differs from the previous one

        Args:
            status_value (str): The current router status value to record
        """
        if status_value != self._last_status:
            now = time.time()
            self.entries.append((now, status_value))
            self._last_status = status_value
            cutoff = now - WINDOW_SECONDS
            self.entries = [(t, s) for t, s in self.entries if t > cutoff]
