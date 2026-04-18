from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.dataclasses.provider_models import DownloadedProviderAsset, DropboxAccount, DropboxEntry
from libs.providers.http_client import ProviderHttpClient

from .credentials import DropboxCredentials


class DropboxProvider:
    def __init__(
        self,
        credentials: DropboxCredentials | str,
        *,
        http_client: ProviderHttpClient | None = None,
        download_dir: str | Path | None = None,
    ) -> None:
        self.credentials = credentials if isinstance(credentials, DropboxCredentials) else DropboxCredentials(token=credentials)
        self.http_client = http_client or ProviderHttpClient()
        self.download_dir = Path(download_dir or Path.cwd() / "cache" / "providers" / "dropbox").expanduser().resolve()

    def get_account(self) -> DropboxAccount:
        payload = self.http_client.post_json(
            "https://api.dropboxapi.com/2/users/get_current_account",
            headers=self._headers(),
        )
        name = payload.get("name", {}) if isinstance(payload.get("name"), dict) else {}
        account_type = payload.get("account_type", {}) if isinstance(payload.get("account_type"), dict) else {}
        return DropboxAccount(
            account_id=str(payload.get("account_id", "")),
            email=payload.get("email"),
            display_name=name.get("display_name"),
            account_type=account_type.get(".tag"),
        )

    def get_entry(self, path: str) -> DropboxEntry:
        payload = self.http_client.post_json(
            "https://api.dropboxapi.com/2/files/get_metadata",
            headers=self._headers(),
            body={"path": path, "include_deleted": False, "include_has_explicit_shared_members": False},
        )
        return self._parse_entry(payload)

    def list_folder(self, *, path: str = "", recursive: bool = False) -> tuple[DropboxEntry, ...]:
        payload = self.http_client.post_json(
            "https://api.dropboxapi.com/2/files/list_folder",
            headers=self._headers(),
            body={"path": path, "recursive": recursive},
        )
        items = [self._parse_entry(entry) for entry in payload.get("entries", []) if isinstance(entry, dict)]
        cursor = payload.get("cursor")
        has_more = bool(payload.get("has_more"))
        while has_more and cursor:
            payload = self.http_client.post_json(
                "https://api.dropboxapi.com/2/files/list_folder/continue",
                headers=self._headers(),
                body={"cursor": cursor},
            )
            items.extend(self._parse_entry(entry) for entry in payload.get("entries", []) if isinstance(entry, dict))
            cursor = payload.get("cursor")
            has_more = bool(payload.get("has_more"))
        return tuple(items)

    def fetch_media(self, path: str) -> DownloadedProviderAsset:
        entry = self.get_entry(path)
        if entry.entry_type != "file":
            raise ValueError(f"Dropbox path {path!r} does not point to a file.")
        if entry.is_downloadable is False:
            raise ValueError(f"Dropbox path {path!r} is not directly downloadable. Export flows are not yet implemented.")
        safe_name = entry.name or "dropbox-file"
        destination = self.download_dir / safe_name
        payload = self.http_client.request_bytes(
            "POST",
            "https://content.dropboxapi.com/2/files/download",
            headers={
                **self._headers(),
                "Dropbox-API-Arg": json.dumps({"path": path}),
            },
            data=b"",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return DownloadedProviderAsset(
            provider="dropbox",
            remote_id=entry.entry_id,
            name=entry.name,
            local_path=destination,
            source_url=path,
            size_bytes=entry.size_bytes,
            metadata={"entry": entry},
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.credentials.token}", "Content-Type": "application/json"}

    @staticmethod
    def _parse_entry(payload: dict[str, Any]) -> DropboxEntry:
        return DropboxEntry(
            entry_type=str(payload.get(".tag", "")),
            entry_id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            path_display=payload.get("path_display"),
            path_lower=payload.get("path_lower"),
            server_modified=payload.get("server_modified"),
            size_bytes=DropboxProvider._coerce_int(payload.get("size")),
            is_downloadable=payload.get("is_downloadable"),
            content_hash=payload.get("content_hash"),
            metadata=payload,
        )

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        return None if value in (None, "") else int(value)
