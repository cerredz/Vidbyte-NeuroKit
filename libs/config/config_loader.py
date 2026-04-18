from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from libs.dataclasses import TribeConfig


class ConfigLoader:
    def __init__(self, config: TribeConfig | None = None, config_path: str | Path | None = None) -> None:
        self.config = config or TribeConfig()
        self.config_path = self._resolve_config_path(config_path)

    def load(self) -> TribeConfig:
        defaults = self._read_defaults()
        overrides = self.config.to_dict()
        merged = {**defaults, **{key: value for key, value in overrides.items() if value is not None}}
        return self._validate(merged)

    def _resolve_config_path(self, config_path: str | Path | None) -> Path:
        if config_path is None:
            return Path(__file__).resolve().with_name("tribe_runner.yaml")
        return Path(config_path).expanduser().resolve()

    def _read_defaults(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a top-level mapping: {self.config_path}")
        return payload

    def _validate(self, payload: dict[str, Any]) -> TribeConfig:
        required_keys = {
            "model_name",
            "cache_dir",
            "output_dir",
            "checkpoint_name",
            "device",
            "cluster",
            "config_update",
        }
        missing_keys = sorted(required_keys - payload.keys())
        if missing_keys:
            raise ValueError(f"Config file is missing required keys: {missing_keys}")

        config_update = payload["config_update"]
        if config_update is not None and not isinstance(config_update, dict):
            raise ValueError("config_update must be a mapping or null.")

        return TribeConfig(
            model_name=str(payload["model_name"]),
            cache_dir=payload["cache_dir"],
            output_dir=payload["output_dir"],
            checkpoint_name=str(payload["checkpoint_name"]),
            device=str(payload["device"]),
            cluster=payload["cluster"],
            config_update=config_update,
        )
