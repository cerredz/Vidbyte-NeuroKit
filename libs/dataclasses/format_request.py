from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from libs.enums import DataFormat


@dataclass(frozen=True, slots=True)
class FormatRequest:
    source: DataFormat | None = None
    target: DataFormat | None = None

    def __post_init__(self) -> None:
        if self.source is None and self.target is None:
            raise ValueError("FormatRequest requires at least one of source or target.")

    @classmethod
    def from_value(
        cls,
        value: "FormatRequest | DataFormat | str | Mapping[str, Any] | None",
    ) -> "FormatRequest | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, DataFormat):
            return cls(target=value)
        if isinstance(value, str):
            return cls(target=cls._coerce_format(value))
        if isinstance(value, Mapping):
            source = value.get("source", value.get("from"))
            target = value.get("target", value.get("to", value.get("format")))
            return cls(
                source=cls._coerce_optional_format(source),
                target=cls._coerce_optional_format(target),
            )
        raise TypeError(
            "format must be a supported format string, DataFormat enum, mapping, or FormatRequest."
        )

    @staticmethod
    def _coerce_optional_format(value: Any) -> DataFormat | None:
        if value is None:
            return None
        return FormatRequest._coerce_format(value)

    @staticmethod
    def _coerce_format(value: Any) -> DataFormat:
        if isinstance(value, DataFormat):
            return value
        if not isinstance(value, str):
            raise TypeError(f"Unsupported format value: {value!r}")
        try:
            return DataFormat(value.strip().lower())
        except ValueError as exc:
            supported_formats = ", ".join(sorted(format_.value for format_ in DataFormat))
            raise ValueError(
                f"Unsupported format '{value}'. Expected one of: {supported_formats}."
            ) from exc
