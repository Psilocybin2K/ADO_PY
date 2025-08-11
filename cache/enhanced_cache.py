from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

from config.settings import CacheOptions
from cache.metrics import MetricsCollector
from app_logging.logger import EnhancedLogger


class EnhancedFileCache:
    def __init__(self, config: CacheOptions, metrics_collector: MetricsCollector, logger: EnhancedLogger):
        self.config = config
        self.metrics = metrics_collector
        self.logger = logger
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, *args, **kwargs) -> str:
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(content.encode()).hexdigest()

    # Public helper to avoid accessing protected member externally
    def compute_cache_key(self, *args, **kwargs) -> str:
        return self._get_cache_key(*args, **kwargs)

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.meta"

    def _is_expired(self, metadata_path: Path) -> bool:
        if not metadata_path.exists():
            return True
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            created_time = datetime.fromisoformat(metadata["created_at"])
            expiry_time = created_time + timedelta(hours=self.config.ttl_hours)
            return datetime.now() > expiry_time
        except Exception:
            return True

    def get(self, cache_key: str) -> Optional[Any]:
        start_time = time.time()
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            if not cache_path.exists() or self._is_expired(metadata_path):
                if self.config.collect_metrics:
                    self.metrics.record_miss(time.time() - start_time)
                return None
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self.config.collect_metrics:
                self.metrics.record_hit(time.time() - start_time)
            self.logger.logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return data
        except Exception as exc:
            if self.config.collect_metrics:
                self.metrics.record_error()
            self.logger.logger.error(f"Cache get error for key {cache_key[:8]}...: {exc}")
            return None

    def set(self, cache_key: str, data: Any) -> None:
        start_time = time.time()
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            size_bytes = int(self._get_cache_path(cache_key).stat().st_size)
            metadata = {
                "created_at": datetime.now().isoformat(),
                "size_bytes": size_bytes,
                "cache_key": cache_key,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            if self.config.collect_metrics:
                self.metrics.record_set(time.time() - start_time, size_bytes)
            self.logger.logger.debug(f"Cache set for key: {cache_key[:8]}...")
        except Exception as exc:
            if self.config.collect_metrics:
                self.metrics.record_error()
            self.logger.logger.error(f"Cache set error for key {cache_key[:8]}...: {exc}")

    def get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs):
        cached_data = self.get(cache_key)
        if cached_data is not None:
            return cached_data
        self.logger.logger.info(f"Fetching new data for key: {cache_key[:8]}...")
        data = fetch_func(*args, **kwargs)
        self.set(cache_key, data)
        return data


