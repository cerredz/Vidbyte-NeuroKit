from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class BaseProviderCredentials(ABC):
    provider_key: ClassVar[str]

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BaseProviderCredentials":
        return cls(**payload)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def to_redacted_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self.to_payload().items():
            if isinstance(value, str) and any(token in key.lower() for token in ("token", "secret", "key")):
                payload[key] = self._redact(value)
            else:
                payload[key] = value
        return payload

    @staticmethod
    def _redact(value: str) -> str:
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-4:]}"
