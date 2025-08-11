from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
from platformdirs import user_cache_dir
 

@dataclass
class ProcessingOptions:
    batch_size: int = 50
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    include_commit_diffs: bool = True
    include_iterations: bool = True
    max_pull_requests: Optional[int] = None
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None


@dataclass
class CacheOptions:
    enabled: bool = True
    cache_dir: str = field(default_factory=lambda: user_cache_dir("ado_git"))
    ttl_hours: int = 24
    max_cache_size_mb: int = 1024
    collect_metrics: bool = True


@dataclass
class LoggingOptions:
    level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = field(default_factory=lambda: "ado_analysis.log")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    verbose_timing: bool = True
    performance_metrics: bool = True


@dataclass
class ApplicationConfig:
    organization_url: str = ""
    personal_access_token: str = ""
    target_repositories: List[str] = field(default_factory=list)
    processing: ProcessingOptions = field(default_factory=ProcessingOptions)
    cache: CacheOptions = field(default_factory=CacheOptions)
    logging: LoggingOptions = field(default_factory=LoggingOptions)

    @classmethod
    def from_file(cls, config_path: str) -> "ApplicationConfig":
        # Avoid import at module import/type-check time; import lazily here
        from src.config.loader import ConfigurationLoader
        return ConfigurationLoader.load_config(config_path)

    @classmethod
    def from_env(cls) -> "ApplicationConfig":
        def get_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        org_url = os.getenv("ADO_ORG_URL", os.getenv("ADO_ORGANIZATION_URL", "")).strip()
        pat = os.getenv("ADO_PAT", os.getenv("ADO_PERSONAL_ACCESS_TOKEN", "")).strip()
        repos_env = os.getenv("ADO_TARGET_REPOS", "").strip()
        repos = [r.strip() for r in repos_env.split(",") if r.strip()] if repos_env else []

        processing = ProcessingOptions(
            batch_size=int(os.getenv("ADO_BATCH_SIZE", "50")),
            max_concurrent_requests=int(os.getenv("ADO_MAX_CONCURRENCY", "5")),
            retry_attempts=int(os.getenv("ADO_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("ADO_RETRY_DELAY", "1.0")),
            include_commit_diffs=get_bool("ADO_INCLUDE_DIFFS", True),
            include_iterations=get_bool("ADO_INCLUDE_ITERATIONS", True),
            max_pull_requests=(int(os.getenv("ADO_MAX_PRS", "0")) or None),
            date_range_start=os.getenv("ADO_DATE_START"),
            date_range_end=os.getenv("ADO_DATE_END"),
        )

        cache = CacheOptions(
            enabled=get_bool("ADO_CACHE_ENABLED", True),
            cache_dir=os.getenv("ADO_CACHE_DIR", user_cache_dir("ado_git")),
            ttl_hours=int(os.getenv("ADO_CACHE_TTL_HOURS", "24")),
            max_cache_size_mb=int(os.getenv("ADO_CACHE_MAX_MB", "1024")),
            collect_metrics=get_bool("ADO_CACHE_METRICS", True),
        )

        logging = LoggingOptions(
            level=os.getenv("ADO_LOG_LEVEL", "INFO"),
            log_to_file=get_bool("ADO_LOG_TO_FILE", True),
            log_file_path=os.getenv("ADO_LOG_FILE", "ado_analysis.log"),
            verbose_timing=get_bool("ADO_VERBOSE_TIMING", True),
            performance_metrics=get_bool("ADO_PERF_METRICS", True),
        )

        return cls(
            organization_url=org_url,
            personal_access_token=pat,
            target_repositories=repos,
            processing=processing,
            cache=cache,
            logging=logging,
        )


