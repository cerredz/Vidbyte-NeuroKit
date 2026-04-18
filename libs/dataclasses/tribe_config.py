from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TribeConfig:
    model_name: str | None = None
    cache_dir: str | Path | None = None
    output_dir: str | Path | None = None
    checkpoint_name: str | None = None
    device: str | None = None
    cluster: str | None = None
    config_update: Mapping[str, Any] | None = None
    input_path: str | Path | None = None
    save_to: str | Path | None = None
    verbose: bool | None = None
    include_brain_stimulus_csv: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env_file(cls, env_path: str | Path | None = None) -> "TribeConfig":
        resolved_env_path = cls._resolve_env_path(env_path)
        if not resolved_env_path.is_file():
            return cls()

        payload: dict[str, Any] = {}
        for raw_line in resolved_env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env line in {resolved_env_path}: {raw_line}")

            key, raw_value = line.split("=", 1)
            field_name = cls._env_to_field_name(key.strip())
            if field_name is None:
                continue
            payload[field_name] = cls._parse_env_value(field_name, raw_value.strip())

        return cls(**payload)

    @staticmethod
    def _resolve_env_path(env_path: str | Path | None) -> Path:
        if env_path is None:
            return Path.cwd() / ".env"
        return Path(env_path).expanduser().resolve()

    @staticmethod
    def _env_to_field_name(key: str) -> str | None:
        mapping = {
            "TRIBE_MODEL_NAME": "model_name",
            "TRIBE_CACHE_DIR": "cache_dir",
            "TRIBE_OUTPUT_DIR": "output_dir",
            "TRIBE_CHECKPOINT_NAME": "checkpoint_name",
            "TRIBE_DEVICE": "device",
            "TRIBE_CLUSTER": "cluster",
            "TRIBE_CONFIG_UPDATE_JSON": "config_update",
            "TRIBE_INPUT_PATH": "input_path",
            "TRIBE_SAVE_TO": "save_to",
            "TRIBE_VERBOSE": "verbose",
            "TRIBE_INCLUDE_BRAIN_STIMULUS_CSV": "include_brain_stimulus_csv",
        }
        return mapping.get(key)

    @staticmethod
    def _parse_env_value(field_name: str, raw_value: str) -> Any:
        value = raw_value.strip().strip("\"'")
        if value.lower() in {"", "none", "null"}:
            return None
        if field_name == "config_update":
            payload = json.loads(value)
            if not isinstance(payload, dict):
                raise ValueError("TRIBE_CONFIG_UPDATE_JSON must decode to a JSON object.")
            return payload
        if field_name in {"verbose", "include_brain_stimulus_csv"}:
            normalized = value.lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Invalid boolean value for {field_name}: {raw_value}")
        return value
