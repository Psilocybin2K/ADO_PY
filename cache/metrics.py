from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import threading


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    operation_times: List[float] = field(default_factory=list)
    last_cleanup: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_operation_time(self) -> float:
        return sum(self.operation_times) / len(self.operation_times) if self.operation_times else 0.0


class MetricsCollector:
    def __init__(self):
        self._metrics = CacheMetrics()
        self._lock = threading.Lock()

    def record_hit(self, operation_time: float):
        with self._lock:
            self._metrics.hits += 1
            self._metrics.operation_times.append(operation_time)

    def record_miss(self, operation_time: float):
        with self._lock:
            self._metrics.misses += 1
            self._metrics.operation_times.append(operation_time)

    def record_set(self, operation_time: float, data_size: int):
        with self._lock:
            self._metrics.sets += 1
            self._metrics.total_size_bytes += data_size
            self._metrics.operation_times.append(operation_time)

    def record_error(self):
        with self._lock:
            self._metrics.errors += 1

    def get_metrics(self) -> CacheMetrics:
        with self._lock:
            return self._metrics

    def reset_metrics(self):
        with self._lock:
            self._metrics = CacheMetrics()


