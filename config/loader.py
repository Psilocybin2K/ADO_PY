import yaml

import json
from pathlib import Path
from typing import Union

from src.config.settings import ApplicationConfig

class ConfigurationLoader:
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> ApplicationConfig:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            return ConfigurationLoader._load_json(path)
        elif suffix in (".yml", ".yaml"):
            return ConfigurationLoader._load_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    @staticmethod
    def _load_json(config_path: Path) -> ApplicationConfig:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ApplicationConfig(**data)

    @staticmethod
    def _load_yaml(config_path: Path) -> ApplicationConfig:
        # Import lazily to reduce type-checker noise
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is not installed; cannot load YAML config")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ApplicationConfig(**data)


