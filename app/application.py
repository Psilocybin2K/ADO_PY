from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.cache.enhanced_cache import EnhancedFileCache
from src.cache.metrics import MetricsCollector
from src.client.enhanced_ado_client import LifetimeAnalysisClient
from src.config.loader import ConfigurationLoader
from src.config.settings import ApplicationConfig
from src.app_logging.logger import EnhancedLogger


class ADOAnalysisApplication:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_configuration(config_path)
        self.logger = EnhancedLogger(self.config.logging)
        self.metrics_collector = MetricsCollector()
        self.cache = EnhancedFileCache(self.config.cache, self.metrics_collector, self.logger)
        self.client = LifetimeAnalysisClient(self.config, self.logger, self.cache)

    def _load_configuration(self, config_path: Optional[str]) -> ApplicationConfig:
        if config_path:
            return ConfigurationLoader.load_config(config_path)
        elif os.getenv("ADO_CONFIG_PATH"):
            return ConfigurationLoader.load_config(os.getenv("ADO_CONFIG_PATH", ""))
        else:
            return ApplicationConfig.from_env()

    def analyze_repositories(self, repository_names: Optional[List[str]] = None) -> Dict:
        target_repos = repository_names or self.config.target_repositories
        if not target_repos:
            raise ValueError("No repositories specified for analysis")

        with self.logger.operation_timer("full_analysis", repositories=target_repos):
            results: Dict[str, Dict] = {}
            for repo_name in target_repos:
                try:
                    self.logger.logger.info("Starting analysis for repository: %s", repo_name)
                    repo_result = self._analyze_single_repository(repo_name)
                    results[repo_name] = repo_result
                except Exception as exc:  # noqa: BLE001 - intentional: continue on per-repo failure
                    self.logger.logger.error("Failed to analyze repository %s: %s", repo_name, exc)
                    results[repo_name] = {
                        "error": str(exc),
                        "analysis_timestamp": datetime.now().isoformat(),
                    }

            final_report = self._generate_analysis_report(results)
            self._save_analysis_results(final_report)
            return final_report

    def _analyze_single_repository(self, repo_name: str) -> Dict:
        with self.logger.operation_timer(f"analyze_repository_{repo_name}"):
            repositories = self.client.get_repositories()
            target_repo = next((r for r in repositories if r.get("name") == repo_name), None)
            if not target_repo:
                raise ValueError(f"Repository '{repo_name}' not found")

            all_prs = self.client.get_repository_lifetime_pull_requests(target_repo["id"])
            processed_prs = self.client.process_pull_requests_with_iterations(target_repo["id"], all_prs)

            analysis_result = {
                "repository": {
                    "id": target_repo["id"],
                    "name": target_repo["name"],
                    "default_branch": target_repo.get("default_branch"),
                    "size": target_repo.get("size"),
                    "url": target_repo.get("url"),
                },
                "pull_requests": processed_prs,
                "statistics": self._generate_repository_statistics(processed_prs),
                "analysis_timestamp": datetime.now().isoformat(),
            }
            return analysis_result

    def _generate_repository_statistics(self, pull_requests: List[Dict]) -> Dict:
        total_prs = len(pull_requests)
        total_iterations = sum(len(pr.get("iterations", [])) for pr in pull_requests)
        status_counts: Dict[str, int] = {}
        for pr in pull_requests:
            status = pr.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        creation_dates_raw = [pr.get("creation_date") for pr in pull_requests]
        creation_dates = [d for d in creation_dates_raw if isinstance(d, str) and d]
        if creation_dates:
            earliest = min(creation_dates)
            latest = max(creation_dates)
        else:
            earliest = None
            latest = None
        date_range = {"earliest": earliest, "latest": latest}
        return {
            "total_pull_requests": total_prs,
            "total_iterations": total_iterations,
            "status_distribution": status_counts,
            "date_range": date_range,
            "cache_metrics": self.metrics_collector.get_metrics().__dict__ if self.config.cache.collect_metrics else None,
        }

    def _generate_analysis_report(self, results: Dict) -> Dict:
        return {
            "analysis_summary": {
                "total_repositories": len(results),
                "successful_analyses": len([r for r in results.values() if "error" not in r]),
                "failed_analyses": len([r for r in results.values() if "error" in r]),
                "analysis_timestamp": datetime.now().isoformat(),
                "configuration": {
                    "processing_options": self.config.processing.__dict__,
                    "cache_options": self.config.cache.__dict__,
                },
            },
            "repositories": results,
            "execution_metrics": [m.__dict__ for m in self.logger.metrics],
            "cache_metrics": self.metrics_collector.get_metrics().__dict__ if self.config.cache.collect_metrics else None,
        }

    def _save_analysis_results(self, report: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.cache.cache_dir) / f"analysis_report_{timestamp}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.logger.info("Analysis report saved to: %s", output_file)
        except Exception as exc:  # noqa: BLE001 - log and continue without raising
            self.logger.logger.error("Failed to save analysis report: %s", exc)


