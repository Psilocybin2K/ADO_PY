from __future__ import annotations

from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # for type hints only; avoid import-time errors
    from azure.devops.connection import Connection  # type: ignore
    from msrest.authentication import BasicAuthentication  # type: ignore
    from azure.devops.v7_1.git.models import GitPullRequestSearchCriteria  # type: ignore

from cache.enhanced_cache import EnhancedFileCache
from config.settings import ApplicationConfig
from app_logging.logger import EnhancedLogger, log_execution_time
from processing.batch_processor import BatchProcessor, batch_items


class CachedADOGitClient:
    def __init__(self, base_url: str, personal_access_token: str):
        # Import lazily via importlib to avoid import-time linter errors
        import importlib
        msrest_auth = importlib.import_module('msrest.authentication')
        ado_conn_mod = importlib.import_module('azure.devops.connection')
        BasicAuthentication = getattr(msrest_auth, 'BasicAuthentication')
        Connection = getattr(ado_conn_mod, 'Connection')

        credentials = BasicAuthentication("", personal_access_token)
        connection = Connection(base_url=base_url, creds=credentials)
        self.git_client = connection.clients_v7_1.get_git_client()

    # Minimal serializers to stable dicts for caching/processing
    @staticmethod
    def _serialize_pull_request(pr) -> Dict:
        return {
            "pull_request_id": pr.pull_request_id,
            "title": pr.title,
            "status": str(pr.status) if getattr(pr, "status", None) else None,
            "created_by_display_name": getattr(getattr(pr, "created_by", None), "display_name", None),
            "creation_date": (
                pr.creation_date.isoformat() if getattr(pr, "creation_date", None) else None
            ),
            "source_ref_name": getattr(pr, "source_ref_name", None),
            "target_ref_name": getattr(pr, "target_ref_name", None),
        }

    @staticmethod
    def _serialize_repository(repo) -> Dict:
        return {
            "id": repo.id,
            "name": repo.name,
            "default_branch": getattr(repo, "default_branch", None),
            "size": getattr(repo, "size", None),
            "url": getattr(repo, "url", None),
        }

    @staticmethod
    def _serialize_iteration(iteration) -> Dict:
        def to_iso(dt):
            return dt.isoformat() if dt else None

        return {
            "id": iteration.id,
            "created_date": to_iso(getattr(iteration, "created_date", None)),
            "updated_date": to_iso(getattr(iteration, "updated_date", None)),
            "description": getattr(iteration, "description", None),
            "author_display_name": getattr(getattr(iteration, "author", None), "display_name", None),
            "source_ref_commit_id": getattr(getattr(iteration, "source_ref_commit", None), "commit_id", None),
            "target_ref_commit_id": getattr(getattr(iteration, "target_ref_commit", None), "commit_id", None),
        }


class LifetimeAnalysisClient(CachedADOGitClient):
    def __init__(self, config: ApplicationConfig, logger: EnhancedLogger, cache: EnhancedFileCache):
        super().__init__(config.organization_url, config.personal_access_token)
        self.config = config
        self.logger = logger
        self.cache = cache
        self.batch_processor: BatchProcessor = BatchProcessor(config.processing, logger)

    def get_repositories(self) -> List[Dict]:
        cache_key = self.cache.compute_cache_key("repositories")

        def fetch():
            repos = self.git_client.get_repositories()
            return [self._serialize_repository(r) for r in repos]

        return self.cache.get_cached_or_fetch(cache_key, fetch)

    @log_execution_time("get_repository_lifetime_prs")
    def get_repository_lifetime_pull_requests(self, repository_id: str) -> List[Dict]:
        with self.logger.operation_timer("fetch_all_pull_requests", repository_id=repository_id):
            import importlib
            git_models = importlib.import_module('azure.devops.v7_1.git.models')
            GitPullRequestSearchCriteria = getattr(git_models, 'GitPullRequestSearchCriteria')
            search = GitPullRequestSearchCriteria(status="all", include_links=True)

            all_prs: List[Dict] = []
            skip = 0
            batch_size = self.config.processing.batch_size
            while True:
                batch = self._get_pr_batch(repository_id, search, skip, batch_size)
                if not batch:
                    break
                all_prs.extend(batch)
                skip += batch_size

                if self.config.processing.max_pull_requests and len(all_prs) >= self.config.processing.max_pull_requests:
                    all_prs = all_prs[: self.config.processing.max_pull_requests]
                    break

            all_prs.sort(key=lambda pr: pr.get("creation_date", ""))
            self.logger.logger.info("Retrieved %s pull requests for repository %s", len(all_prs), repository_id)
            return all_prs

    def _get_pr_batch(self, repository_id: str, search_criteria, skip: int, top: int) -> List[Dict]:
        key = self.cache.compute_cache_key(
            "pr_batch",
            repository_id,
            f"status=={getattr(search_criteria, 'status', None)}",
            f"include_links=={getattr(search_criteria, 'include_links', None)}",
            skip,
            top,
        )

        def fetch():
            prs = self.git_client.get_pull_requests(
                repository_id=repository_id,
                search_criteria=search_criteria,
                skip=skip,
                top=top,
            )
            return [self._serialize_pull_request(pr) for pr in prs]

        return self.cache.get_cached_or_fetch(key, fetch)

    def get_pr_iterations(self, repository_id: str, pull_request_id: int, include_commits: bool = True) -> List[Dict]:
        key = self.cache.compute_cache_key("pr_iterations", repository_id, pull_request_id, include_commits)

        def fetch():
            iterations = self.git_client.get_pull_request_iterations(
                repository_id=repository_id, pull_request_id=pull_request_id, include_commits=include_commits
            )
            return [self._serialize_iteration(i) for i in iterations]

        return self.cache.get_cached_or_fetch(key, fetch)

    def get_pr_iteration_changes_full(self, repository_id: str, pull_request_id: int, iteration_id: int) -> Dict:
        changes: List[Dict] = []
        skip: Optional[int] = 0
        page_size = 2000
        total = 0
        while True:
            key = self.cache.compute_cache_key(
                "pr_iteration_changes", repository_id, pull_request_id, iteration_id, skip or 0, page_size
            )

            def fetch_page():
                resp = self.git_client.get_pull_request_iteration_changes(
                    repository_id=repository_id,
                    pull_request_id=pull_request_id,
                    iteration_id=iteration_id,
                    skip=skip,
                    top=page_size,
                )
                return {
                    "entries": [
                        {"change_tracking_id": c.change_tracking_id} for c in getattr(resp, "change_entries", [])
                    ],
                    "next_skip": getattr(resp, "next_skip", None),
                    "next_top": getattr(resp, "next_top", None),
                }

            page = self.cache.get_cached_or_fetch(key, fetch_page)
            page_entries = page.get("entries", [])
            changes.extend(page_entries)
            total += len(page_entries)
            if not page.get("next_skip"):
                break
            skip = page.get("next_skip")

        return {"change_entries": changes, "total_changes": total}

    def process_pull_requests_with_iterations(self, repository_id: str, pull_requests: List[Dict]) -> List[Dict]:
        def process_single_pr(pr_data: Dict) -> Dict:
            return self._process_pull_request_with_iterations(repository_id, pr_data)

        pr_batches = batch_items(pull_requests, self.config.processing.batch_size)
        all_processed: List[Dict] = []
        for idx, pr_batch in enumerate(pr_batches):
            self.logger.logger.info("Processing PR batch %s/%s", idx + 1, len(pr_batches))
            batch_result = self.batch_processor.process_batch_sync(pr_batch, process_single_pr)
            all_processed.extend(batch_result.successful)
            for failed_item, exc in batch_result.failed:
                self.logger.logger.error(
                    "Failed to process PR %s: %s",
                    failed_item.get("pull_request_id", "unknown"),
                    exc,
                )
        return all_processed

    def _process_pull_request_with_iterations(self, repository_id: str, pr_data: Dict) -> Dict:
        pr_id = pr_data["pull_request_id"]
        with self.logger.operation_timer(f"process_pr_{pr_id}", pr_id=pr_id):
            iterations = []
            if self.config.processing.include_iterations:
                iterations_data = self.get_pr_iterations(repository_id, pr_id, include_commits=True)
                for iteration in iterations_data:
                    processed_iteration = self._process_iteration(repository_id, pr_id, iteration)
                    iterations.append(processed_iteration)
            pr_data["iterations"] = iterations
            return pr_data

    def _process_iteration(self, repository_id: str, pr_id: int, iteration_data: Dict) -> Dict:
        iteration_id = iteration_data["id"]
        changes = self.get_pr_iteration_changes_full(repository_id, pr_id, iteration_id)
        iteration_data["changes"] = changes
        return iteration_data
