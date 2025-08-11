# ADO Git Analysis Tool Refactor Plan

## 1. Configuration Management

### 1.1 Configuration Structure
```python
# config/settings.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
from pathlib import Path

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
    cache_dir: str = field(default_factory=lambda: os.path.join(os.environ.get('APPDATA', ''), '.cache_ADO_GIT'))
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
    organization_url: str
    personal_access_token: str
    target_repositories: List[str] = field(default_factory=list)
    processing: ProcessingOptions = field(default_factory=ProcessingOptions)
    cache: CacheOptions = field(default_factory=CacheOptions)
    logging: LoggingOptions = field(default_factory=LoggingOptions)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ApplicationConfig':
        """Load configuration from JSON/YAML file"""
        # Implementation for loading from file
        pass
    
    @classmethod
    def from_env(cls) -> 'ApplicationConfig':
        """Load configuration from environment variables"""
        # Implementation for loading from env vars
        pass
```

### 1.2 Configuration Loader
```python
# config/loader.py
import json
import yaml
from pathlib import Path
from typing import Union

class ConfigurationLoader:
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> ApplicationConfig:
        """Load configuration from file with format auto-detection"""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.json':
            return ConfigurationLoader._load_json(config_path)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            return ConfigurationLoader._load_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def _load_json(config_path: Path) -> ApplicationConfig:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return ApplicationConfig(**data)
    
    @staticmethod
    def _load_yaml(config_path: Path) -> ApplicationConfig:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return ApplicationConfig(**data)
```

## 2. Logging and Metrics Framework

### 2.1 Enhanced Logging System
```python
# logging/logger.py
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ExecutionMetrics:
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    additional_data: Dict[str, Any] = None

class EnhancedLogger:
    def __init__(self, config: LoggingOptions):
        self.config = config
        self.logger = self._setup_logger()
        self.metrics: List[ExecutionMetrics] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger('ado_analysis')
        logger.setLevel(getattr(logging, self.config.level))
        
        formatter = logging.Formatter(self.config.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            file_handler = logging.FileHandler(self.config.log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @contextmanager
    def operation_timer(self, operation_name: str, **additional_data):
        """Context manager for timing operations"""
        metric = ExecutionMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            additional_data=additional_data
        )
        
        start_time = time.time()
        self.logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield metric
            metric.success = True
        except Exception as e:
            metric.success = False
            metric.error_message = str(e)
            self.logger.error(f"Operation failed: {operation_name} - {str(e)}")
            raise
        finally:
            end_time = time.time()
            metric.end_time = datetime.now()
            metric.duration_seconds = end_time - start_time
            
            if self.config.verbose_timing:
                self.logger.info(f"Completed operation: {operation_name} in {metric.duration_seconds:.2f}s")
            
            if self.config.performance_metrics:
                self.metrics.append(metric)

def log_execution_time(operation_name: str = None):
    """Decorator for logging function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = logging.getLogger('ado_analysis')
            
            start_time = time.time()
            logger.debug(f"Starting {name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {name} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {name} after {duration:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator
```

### 2.2 Cache Metrics System
```python
# cache/metrics.py
from dataclasses import dataclass, field
from typing import Dict, List
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
```

## 3. Enhanced Cache System

### 3.1 Cache with Metrics and TTL
```python
# cache/enhanced_cache.py
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

class EnhancedFileCache:
    def __init__(self, config: CacheOptions, metrics_collector: MetricsCollector, logger: EnhancedLogger):
        self.config = config
        self.metrics = metrics_collector
        self.logger = logger
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if config.collect_metrics:
            self._schedule_cleanup()
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate SHA256 hash for cache key from arguments"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path for cache file"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path for cache metadata file"""
        return self.cache_dir / f"{cache_key}.meta"
    
    def _is_expired(self, metadata_path: Path) -> bool:
        """Check if cache entry is expired"""
        if not metadata_path.exists():
            return True
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_time = datetime.fromisoformat(metadata['created_at'])
            expiry_time = created_time + timedelta(hours=self.config.ttl_hours)
            
            return datetime.now() > expiry_time
        except Exception:
            return True
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache with metrics tracking"""
        start_time = time.time()
        
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            if not cache_path.exists() or self._is_expired(metadata_path):
                if self.config.collect_metrics:
                    self.metrics.record_miss(time.time() - start_time)
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.config.collect_metrics:
                self.metrics.record_hit(time.time() - start_time)
            
            self.logger.logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return data
            
        except Exception as e:
            if self.config.collect_metrics:
                self.metrics.record_error()
            self.logger.logger.error(f"Cache get error for key {cache_key[:8]}...: {str(e)}")
            return None
    
    def set(self, cache_key: str, data: Any) -> None:
        """Store data in cache with metadata"""
        start_time = time.time()
        
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            # Write data
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Write metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'size_bytes': cache_path.stat().st_size,
                'cache_key': cache_key
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            
            if self.config.collect_metrics:
                self.metrics.record_set(time.time() - start_time, metadata['size_bytes'])
            
            self.logger.logger.debug(f"Cache set for key: {cache_key[:8]}...")
            
        except Exception as e:
            if self.config.collect_metrics:
                self.metrics.record_error()
            self.logger.logger.error(f"Cache set error for key {cache_key[:8]}...: {str(e)}")
    
    def get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs):
        """Get data from cache or fetch and cache it"""
        cached_data = self.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        self.logger.logger.info(f"Fetching new data for key: {cache_key[:8]}...")
        data = fetch_func(*args, **kwargs)
        self.set(cache_key, data)
        return data
```

## 4. Batch Processing System

### 4.1 Batch Processor
```python
# processing/batch_processor.py
import asyncio
from typing import List, Callable, Any, TypeVar, Generic
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class BatchResult:
    successful: List[Any]
    failed: List[tuple]  # (item, exception)
    total_items: int
    processing_time: float

class BatchProcessor(Generic[T, R]):
    def __init__(self, config: ProcessingOptions, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
    
    def process_batch_sync(self, items: List[T], process_func: Callable[[T], R]) -> BatchResult:
        """Process items in batches synchronously with threading"""
        start_time = time.time()
        successful = []
        failed = []
        
        with self.logger.operation_timer(f"batch_process_{len(items)}_items"):
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
                # Submit all items
                future_to_item = {
                    executor.submit(self._process_with_retry, process_func, item): item 
                    for item in items
                }
                
                # Collect results
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        successful.append(result)
                        self.logger.logger.debug(f"Successfully processed item: {str(item)[:50]}...")
                    except Exception as e:
                        failed.append((item, e))
                        self.logger.logger.error(f"Failed to process item {str(item)[:50]}...: {str(e)}")
        
        processing_time = time.time() - start_time
        
        batch_result = BatchResult(
            successful=successful,
            failed=failed,
            total_items=len(items),
            processing_time=processing_time
        )
        
        self.logger.logger.info(
            f"Batch processing completed: {len(successful)} successful, "
            f"{len(failed)} failed, {processing_time:.2f}s total"
        )
        
        return batch_result
    
    def _process_with_retry(self, process_func: Callable[[T], R], item: T) -> R:
        """Process single item with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return process_func(item)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    self.logger.logger.warning(
                        f"Retry {attempt + 1}/{self.config.retry_attempts} for item: {str(e)}"
                    )
                    time.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.logger.error(f"All retry attempts failed for item: {str(e)}")
        
        raise last_exception

def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """Split items into batches of specified size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
```

## 5. Enhanced ADO Client

### 5.1 Repository Lifetime Analysis Client
```python
# client/enhanced_ado_client.py
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from azure.devops.v7_1.git.models import GitPullRequestSearchCriteria

class LifetimeAnalysisClient(CachedADOGitClient):
    def __init__(self, config: ApplicationConfig, logger: EnhancedLogger, cache: EnhancedFileCache):
        super().__init__(config.organization_url, config.personal_access_token)
        self.config = config
        self.logger = logger
        self.cache = cache
        self.batch_processor = BatchProcessor(config.processing, logger)
    
    @log_execution_time("get_repository_lifetime_prs")
    def get_repository_lifetime_pull_requests(self, repository_id: str) -> List[Dict]:
        """Get ALL pull requests for a repository from first to last"""
        
        with self.logger.operation_timer("fetch_all_pull_requests", repository_id=repository_id):
            # Get total count first
            initial_search = GitPullRequestSearchCriteria(
                status='all',
                include_links=True
            )
            
            # Fetch in batches to avoid timeouts
            all_prs = []
            skip = 0
            batch_size = self.config.processing.batch_size
            
            while True:
                self.logger.logger.info(f"Fetching PRs batch: skip={skip}, top={batch_size}")
                
                batch_prs = self._get_pr_batch(repository_id, initial_search, skip, batch_size)
                
                if not batch_prs:
                    break
                
                all_prs.extend(batch_prs)
                skip += batch_size
                
                # Apply max limit if configured
                if (self.config.processing.max_pull_requests and 
                    len(all_prs) >= self.config.processing.max_pull_requests):
                    all_prs = all_prs[:self.config.processing.max_pull_requests]
                    break
            
            # Sort by creation date (first to last)
            all_prs.sort(key=lambda pr: pr.get('creation_date', ''))
            
            self.logger.logger.info(f"Retrieved {len(all_prs)} pull requests for repository {repository_id}")
            return all_prs
    
    def _get_pr_batch(self, repository_id: str, search_criteria, skip: int, top: int) -> List[Dict]:
        """Get a batch of pull requests with caching"""
        cache_key = self.cache._get_cache_key("pr_batch", repository_id, str(search_criteria), skip, top)
        
        def fetch_batch():
            prs = self.git_client.get_pull_requests(
                repository_id=repository_id,
                search_criteria=search_criteria,
                skip=skip,
                top=top
            )
            return [self._serialize_pull_request(pr) for pr in prs]
        
        return self.cache.get_cached_or_fetch(cache_key, fetch_batch)
    
    @log_execution_time("process_prs_with_iterations")
    def process_pull_requests_with_iterations(self, repository_id: str, pull_requests: List[Dict]) -> List[Dict]:
        """Process pull requests with iterations using batch processing"""
        
        def process_single_pr(pr_data: Dict) -> Dict:
            return self._process_pull_request_with_iterations(repository_id, pr_data)
        
        # Process in batches
        pr_batches = batch_items(pull_requests, self.config.processing.batch_size)
        all_processed_prs = []
        
        for batch_idx, pr_batch in enumerate(pr_batches):
            self.logger.logger.info(f"Processing PR batch {batch_idx + 1}/{len(pr_batches)}")
            
            batch_result = self.batch_processor.process_batch_sync(pr_batch, process_single_pr)
            all_processed_prs.extend(batch_result.successful)
            
            # Log batch statistics
            self.logger.logger.info(
                f"Batch {batch_idx + 1} completed: "
                f"{len(batch_result.successful)} successful, "
                f"{len(batch_result.failed)} failed"
            )
            
            # Log any failures
            for failed_item, exception in batch_result.failed:
                self.logger.logger.error(
                    f"Failed to process PR {failed_item.get('pull_request_id', 'unknown')}: {str(exception)}"
                )
        
        return all_processed_prs
    
    def _process_pull_request_with_iterations(self, repository_id: str, pr_data: Dict) -> Dict:
        """Process a single pull request with its iterations"""
        pr_id = pr_data['pull_request_id']
        
        with self.logger.operation_timer(f"process_pr_{pr_id}", pr_id=pr_id):
            # Get iterations
            if self.config.processing.include_iterations:
                iterations_data = self.get_pr_iterations(repository_id, pr_id, include_commits=True)
                
                # Process each iteration
                processed_iterations = []
                for iteration_data in iterations_data:
                    processed_iteration = self._process_iteration(repository_id, pr_id, iteration_data)
                    processed_iterations.append(processed_iteration)
                
                pr_data['iterations'] = processed_iterations
            else:
                pr_data['iterations'] = []
            
            return pr_data
    
    def _process_iteration(self, repository_id: str, pr_id: int, iteration_data: Dict) -> Dict:
        """Process a single iteration with changes and diffs"""
        iteration_id = iteration_data['id']
        
        # Get iteration changes
        changes_data = self.get_pr_iteration_changes(repository_id, pr_id, iteration_id)
        iteration_data['changes'] = changes_data
        
        # Get commit diffs if enabled and available
        if (self.config.processing.include_commit_diffs and 
            iteration_data.get('source_ref_commit_id') and 
            iteration_data.get('target_ref_commit_id')):
            
            commit_diffs_data = self.get_commit_diffs(repository_id)
            iteration_data['commit_diffs'] = commit_diffs_data
        
        return iteration_data
```

## 6. Main Application Orchestrator

### 6.1 Application Entry Point
```python
# main.py
import sys
from pathlib import Path
from typing import Dict, List

class ADOAnalysisApplication:
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize components
        self.logger = EnhancedLogger(self.config.logging)
        self.metrics_collector = MetricsCollector()
        self.cache = EnhancedFileCache(self.config.cache, self.metrics_collector, self.logger)
        self.client = LifetimeAnalysisClient(self.config, self.logger, self.cache)
    
    def _load_configuration(self, config_path: Optional[str]) -> ApplicationConfig:
        """Load configuration from file or environment"""
        if config_path:
            return ConfigurationLoader.load_config(config_path)
        elif os.getenv('ADO_CONFIG_PATH'):
            return ConfigurationLoader.load_config(os.getenv('ADO_CONFIG_PATH'))
        else:
            return ApplicationConfig.from_env()
    
    def analyze_repositories(self, repository_names: Optional[List[str]] = None) -> Dict:
        """Analyze specified repositories or all configured repositories"""
        
        target_repos = repository_names or self.config.target_repositories
        if not target_repos:
            raise ValueError("No repositories specified for analysis")
        
        with self.logger.operation_timer("full_analysis", repositories=target_repos):
            results = {}
            
            for repo_name in target_repos:
                try:
                    self.logger.logger.info(f"Starting analysis for repository: {repo_name}")
                    repo_result = self._analyze_single_repository(repo_name)
                    results[repo_name] = repo_result
                    
                except Exception as e:
                    self.logger.logger.error(f"Failed to analyze repository {repo_name}: {str(e)}")
                    results[repo_name] = {
                        "error": str(e),
                        "analysis_timestamp": datetime.now().isoformat()
                    }
            
            # Generate final report
            final_report = self._generate_analysis_report(results)
            
            # Save results
            self._save_analysis_results(final_report)
            
            return final_report
    
    def _analyze_single_repository(self, repo_name: str) -> Dict:
        """Analyze a single repository completely"""
        
        with self.logger.operation_timer(f"analyze_repository_{repo_name}"):
            # 1. Find repository
            repositories = self.client.get_repositories()
            target_repo = next((repo for repo in repositories if repo.name == repo_name), None)
            
            if not target_repo:
                raise ValueError(f"Repository '{repo_name}' not found")
            
            # 2. Get all pull requests (lifetime)
            all_prs = self.client.get_repository_lifetime_pull_requests(target_repo.id)
            
            # 3. Process pull requests with iterations
            processed_prs = self.client.process_pull_requests_with_iterations(target_repo.id, all_prs)
            
            # 4. Generate repository analysis
            analysis_result = {
                "repository": {
                    "id": target_repo.id,
                    "name": target_repo.name,
                    "default_branch": getattr(target_repo, 'default_branch', None),
                    "size": getattr(target_repo, 'size', None),
                    "url": getattr(target_repo, 'url', None)
                },
                "pull_requests": processed_prs,
                "statistics": self._generate_repository_statistics(processed_prs),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return analysis_result
    
    def _generate_repository_statistics(self, pull_requests: List[Dict]) -> Dict:
        """Generate statistical summary of repository analysis"""
        total_prs = len(pull_requests)
        total_iterations = sum(len(pr.get('iterations', [])) for pr in pull_requests)
        
        # Status distribution
        status_counts = {}
        for pr in pull_requests:
            status = pr.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Date range
        creation_dates = [pr.get('creation_date') for pr in pull_requests if pr.get('creation_date')]
        date_range = {
            "earliest": min(creation_dates) if creation_dates else None,
            "latest": max(creation_dates) if creation_dates else None
        }
        
        return {
            "total_pull_requests": total_prs,
            "total_iterations": total_iterations,
            "status_distribution": status_counts,
            "date_range": date_range,
            "cache_metrics": self.metrics_collector.get_metrics().__dict__ if self.config.cache.collect_metrics else None
        }
    
    def _generate_analysis_report(self, results: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        return {
            "analysis_summary": {
                "total_repositories": len(results),
                "successful_analyses": len([r for r in results.values() if 'error' not in r]),
                "failed_analyses": len([r for r in results.values() if 'error' in r]),
                "analysis_timestamp": datetime.now().isoformat(),
                "configuration": {
                    "processing_options": self.config.processing.__dict__,
                    "cache_options": self.config.cache.__dict__
                }
            },
            "repositories": results,
            "execution_metrics": [m.__dict__ for m in self.logger.metrics],
            "cache_metrics": self.metrics_collector.get_metrics().__dict__ if self.config.cache.collect_metrics else None
        }
    
    def _save_analysis_results(self, report: Dict):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.cache.cache_dir) / f"analysis_report_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.logger.info(f"Analysis report saved to: {output_file}")
        except Exception as e:
            self.logger.logger.error(f"Failed to save analysis report: {str(e)}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ADO Git Repository Analysis Tool')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--repositories', nargs='+', help='Repository names to analyze')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        app = ADOAnalysisApplication(args.config)
        
        if args.verbose:
            app.config.logging.level = "DEBUG"
            app.logger = EnhancedLogger(app.config.logging)
        
        results = app.analyze_repositories(args.repositories)
        
        print(f"Analysis completed successfully. Processed {len(results)} repositories.")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 7. Configuration Files

### 7.1 Example Configuration (config.json)
```json
{
  "organization_url": "https://dev.azure.com/OptimalBlue",
  "personal_access_token": "${ADO_PAT}",
  "target_repositories": ["HedgePlatform", "AnotherRepo"],
  "processing": {
    "batch_size": 50,
    "max_concurrent_requests": 5,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "include_commit_diffs": true,
    "include_iterations": true,
    "max_pull_requests": null,
    "date_range_start": null,
    "date_range_end": null
  },
  "cache": {
    "enabled": true,
    "cache_dir": "./cache",
    "ttl_hours": 24,
    "max_cache_size_mb": 1024,
    "collect_metrics": true
  },
  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_file_path": "ado_analysis.log",
    "verbose_timing": true,
    "performance_metrics": true
  }
}
```

## 8. Implementation Plan

### Phase 1: Configuration and Logging
1. Implement `ApplicationConfig` and `ConfigurationLoader`
2. Create `EnhancedLogger` with timing and metrics
3. Add `MetricsCollector` for cache statistics

### Phase 2: Enhanced Caching
1. Implement `EnhancedFileCache` with TTL and metrics
2. Add cache cleanup and size management
3. Integrate cache metrics collection

### Phase 3: Batch Processing
1. Create `BatchProcessor` with retry logic
2. Implement concurrent processing with threading
3. Add comprehensive error handling and reporting

### Phase 4: Lifetime Analysis Client
1. Extend `CachedADOGitClient` to `LifetimeAnalysisClient`
2. Implement repository lifetime pull request fetching
3. Add batch processing for PR iteration analysis

### Phase 5: Application Orchestrator
1. Create main `ADOAnalysisApplication` class
2. Implement repository analysis workflow
3. Add comprehensive reporting and statistics

### Phase 6: Testing and Documentation
1. Create unit tests for all components
2. Add integration tests for full workflows
3. Document configuration options and usage examples