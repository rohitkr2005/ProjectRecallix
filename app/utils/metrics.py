from contextlib import contextmanager
import time
from typing import Dict


class LatencyTracker:
    """
    11.4 Performance & Latency Metrics

    Tracks execution time in milliseconds across pipeline components.
    """

    def __init__(self):
        self.metrics: Dict[str, float] = {}

    @contextmanager
    def timer(self, label: str):
        """Context manager to record elapsed duration in milliseconds."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.metrics[label] = round(elapsed_ms, 2)

    def record(self, label: str, duration_ms: float):
        """Manually record a latency value."""
        self.metrics[label] = round(duration_ms, 2)

    def get_metrics(self) -> Dict[str, float]:
        """Return a copy of all recorded latency metrics including total_ms."""
        res = dict(self.metrics)
        if "total_ms" not in res and res:
            res["total_ms"] = round(sum(res.values()), 2)
        return res

    def formatted(self) -> str:
        """Return formatted key=value string of metrics."""
        m = self.get_metrics()
        return ", ".join(f"{k}={v:.2f}ms" for k, v in m.items())

    def reset(self):
        """Clear all metrics."""
        self.metrics.clear()
