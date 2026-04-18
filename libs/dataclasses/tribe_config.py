from __future__ import annotations

from dataclasses import asdict, dataclass
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
