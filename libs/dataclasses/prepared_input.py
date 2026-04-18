from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from libs.enums import InputKind


@dataclass(frozen=True, slots=True)
class PreparedInput:
    original_path: Path
    model_path: Path
    kind: InputKind
    model_kwargs: dict[str, str]
