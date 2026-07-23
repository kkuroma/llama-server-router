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
    Samples GPU utilization and VRAM once per second and keeps a rolling window

    Reads GPU index 0 only via NVML, buffering per-second samples and flushing
    their maxima every FLUSH_INTERVAL to keep the history compact
    """

    def __init__(self):
        if not _PYNVML_AVAILABLE:
            raise RuntimeError("pynvml is not installed")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total_vram_mb = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / (1024 ** 2)
        self.util_history: list[tuple[float, float]] = []   # (unix_ts, percent)
        self.vram_history: list[tuple[float, float]] = []   # (unix_ts, used_mb)
        self._util_samples: list[float] = []
        self._vram_samples: list[float] = []
        self._last_flush_time: float = time.time()

    def poll(self):
        """
        Records one utilization/VRAM sample, flushing maxima on the interval

        Called every 1s from a background thread; on flush it appends the window
        maxima and drops history older than WINDOW_SECONDS
        """
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self._util_samples.append(util.gpu)
        self._vram_samples.append(mem.used / (1024 ** 2))

        now = time.time()
        if now - self._last_flush_time >= FLUSH_INTERVAL:
            max_util = max(self._util_samples)
            max_vram = max(self._vram_samples)
            self.util_history.append((now, round(max_util, 1)))
            self.vram_history.append((now, round(max_vram, 1)))
            self._util_samples.clear()
            self._vram_samples.clear()
            self._last_flush_time = now
            cutoff = now - WINDOW_SECONDS
            self.util_history = [(t, v) for t, v in self.util_history if t > cutoff]
            self.vram_history = [(t, v) for t, v in self.vram_history if t > cutoff]


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
