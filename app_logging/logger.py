from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, List

from config.settings import LoggingOptions


@dataclass
class ExecutionMetrics:
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class EnhancedLogger:
    def __init__(self, config: LoggingOptions):
        self.config = config
        self.logger = self._setup_logger()
        self.metrics: List[ExecutionMetrics] = []

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ado_analysis")
        logger.setLevel(getattr(logging, self.config.level, logging.INFO))

        # Avoid duplicate handlers if logger recreated
        if logger.handlers:
            return logger

        formatter = logging.Formatter(self.config.log_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if self.config.log_to_file:
            file_handler = logging.FileHandler(self.config.log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @contextmanager
    def operation_timer(self, operation_name: str, **additional_data):
        metric = ExecutionMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            additional_data=additional_data,
        )

        start_time_ts = time.time()
        self.logger.info("Starting operation: %s", operation_name)

        try:
            yield metric
            metric.success = True
        except Exception as exc:  # re-raise after logging
            metric.success = False
            metric.error_message = str(exc)
            self.logger.error("Operation failed: %s - %s", operation_name, exc)
            raise
        finally:
            end_time_ts = time.time()
            metric.end_time = datetime.now()
            metric.duration_seconds = end_time_ts - start_time_ts

            if self.config.verbose_timing:
                self.logger.info(
                    "Completed operation: %s in %.2fs", operation_name, metric.duration_seconds or 0.0
                )

            if self.config.performance_metrics:
                self.metrics.append(metric)


def log_execution_time(operation_name: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = logging.getLogger("ado_analysis")

            start_time_ts = time.time()
            logger.debug("Starting %s", name)
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time_ts
                logger.info("Completed %s in %.2fs", name, duration)
                return result
            except Exception as exc:
                duration = time.time() - start_time_ts
                logger.error("Failed %s after %.2fs: %s", name, duration, exc)
                raise

        return wrapper

    return decorator


