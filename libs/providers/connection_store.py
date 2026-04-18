from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProviderConnectionStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Path.home() / ".tribe" / "provider_connections.json").expanduser().resolve()

    def load(self, provider_key: str) -> dict[str, Any] | None:
        payload = self._read_all()
        entry = payload.get(provider_key)
        if not isinstance(entry, dict):
            return None
        credentials = entry.get("credentials")
        if not isinstance(credentials, dict):
            return None
        return dict(credentials)

    def save(self, provider_key: str, credentials: dict[str, Any]) -> None:
        payload = self._read_all()
        payload[provider_key] = {"credentials": dict(credentials)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _read_all(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Provider connection store must contain a JSON object: {self.path}")
        return raw
