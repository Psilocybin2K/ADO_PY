from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Callable, Any, TypeVar, Generic

from config.settings import ProcessingOptions
from app_logging.logger import EnhancedLogger

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult:
    successful: List[Any]
    failed: List[tuple]
    total_items: int
    processing_time: float


class BatchProcessor(Generic[T, R]):
    def __init__(self, config: ProcessingOptions, logger: EnhancedLogger):
        self.config = config
        self.logger = logger

    def process_batch_sync(self, items: List[T], process_func: Callable[[T], R]) -> BatchResult:
        start_time_ts = time.time()
        successful: List[Any] = []
        failed: List[tuple] = []

        with self.logger.operation_timer(f"batch_process_{len(items)}_items"):
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
                future_to_item = {executor.submit(self._process_with_retry, process_func, item): item for item in items}

                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        successful.append(result)
                        self.logger.logger.debug(f"Successfully processed item: {str(item)[:50]}...")
                    except Exception as exc:
                        failed.append((item, exc))
                        self.logger.logger.error(f"Failed to process item {str(item)[:50]}...: {exc}")

        processing_time = time.time() - start_time_ts

        batch_result = BatchResult(
            successful=successful,
            failed=failed,
            total_items=len(items),
            processing_time=processing_time,
        )

        self.logger.logger.info(
            f"Batch processing completed: {len(successful)} successful, {len(failed)} failed, {processing_time:.2f}s total"
        )
        return batch_result

    def _process_with_retry(self, process_func: Callable[[T], R], item: T) -> R:
        last_exception: Exception | None = None
        for attempt in range(self.config.retry_attempts):
            try:
                return process_func(item)
            except Exception as exc:
                last_exception = exc
                if attempt < self.config.retry_attempts - 1:
                    self.logger.logger.warning(
                        f"Retry {attempt + 1}/{self.config.retry_attempts} for item: {exc}"
                    )
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.logger.logger.error(f"All retry attempts failed for item: {exc}")
        assert last_exception is not None
        raise last_exception


def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


