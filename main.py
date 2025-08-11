from __future__ import annotations

import argparse
import sys


from src.app.application import ADOAnalysisApplication


def main():
    parser = argparse.ArgumentParser(description="ADO Git Repository Analysis Tool")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--repositories", nargs="+", help="Repository names to analyze")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        app = ADOAnalysisApplication(args.config)
        if args.verbose:
            app.config.logging.level = "DEBUG"
            app.logger = app.logger
        results = app.analyze_repositories(args.repositories)
        print(f"Analysis completed successfully. Processed {len(results)} repositories.")
    except Exception as exc:  # keep broad here to exit non-zero in CLI
        print(f"Analysis failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()