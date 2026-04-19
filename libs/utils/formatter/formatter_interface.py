from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from libs.enums import DataFormat


class FormatterInterface(ABC):
    format_type: DataFormat

    @abstractmethod
    def matches_path(self, path: Path) -> bool:
        pass

    @abstractmethod
    def can_convert_from(self, source_format: DataFormat) -> bool:
        pass

    @abstractmethod
    def from_(self, source: Any) -> Any:
        pass

    @abstractmethod
    def to(self, data: Any, *, output_path: str | Path | None = None) -> Any:
        pass
