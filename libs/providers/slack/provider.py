from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from libs.dataclasses.provider_models import DownloadedProviderAsset, SlackAccount, SlackFile
from libs.enums import ProviderBaseUrl
from libs.providers.http_client import ProviderHttpClient

from .credentials import SlackCredentials


class SlackProvider:
    def __init__(
        self,
        credentials: SlackCredentials | str,
        *,
        http_client: ProviderHttpClient | None = None,
        download_dir: str | Path | None = None,
    ) -> None:
        self.credentials = credentials if isinstance(credentials, SlackCredentials) else SlackCredentials(token=credentials)
        self.http_client = http_client or ProviderHttpClient()
        self.download_dir = Path(download_dir or Path.cwd() / "cache" / "providers" / "slack").expanduser().resolve()

    def get_account(self) -> SlackAccount:
        payload = self._get("auth.test")
        return SlackAccount(
            url=payload.get("url"),
            team=payload.get("team"),
            team_id=payload.get("team_id"),
            user=payload.get("user"),
            user_id=payload.get("user_id"),
            bot_id=payload.get("bot_id"),
        )

    def get_file(self, file_id: str) -> SlackFile:
        payload = self._get("files.info", file=file_id)
        file_payload = payload.get("file")
        if not isinstance(file_payload, dict):
            raise ValueError(f"Slack files.info returned no file payload for file {file_id!r}.")
        return self._parse_file(file_payload)

    def list_files(self, *, channel: str | None = None, user: str | None = None, count: int = 100) -> tuple[SlackFile, ...]:
        payload = self._get("files.list", channel=channel, user=user, count=count)
        return tuple(self._parse_file(item) for item in payload.get("files", []) if isinstance(item, dict))

    def fetch_media(self, file_id: str) -> DownloadedProviderAsset:
        file = self.get_file(file_id)
        download_url = file.url_private_download or file.url_private
        if not download_url:
            raise ValueError(f"Slack file {file.file_id} does not expose a private download URL.")
        suffix = self._infer_suffix(download_url, file.mimetype)
        destination = self.download_dir / f"{self._safe_name(file.title or file.name or file.file_id)}{suffix}"
        self.http_client.download(download_url, destination, headers=self._headers())
        return DownloadedProviderAsset(
            provider="slack",
            remote_id=file.file_id,
            name=file.title or file.name or file.file_id,
            local_path=destination,
            mime_type=file.mimetype,
            source_url=file.permalink or download_url,
            size_bytes=file.size_bytes,
            metadata={"file": file},
        )

    def _get(self, method: str, **params: Any) -> dict[str, Any]:
        payload = self.http_client.get_json(f"{ProviderBaseUrl.SLACK_API.value}/{method}", headers=self._headers(), params=params)
        if not isinstance(payload, dict):
            raise ValueError(f"Slack API returned a non-object response for {method}.")
        if payload.get("ok") is False:
            raise ValueError(f"Slack API {method} failed: {payload.get('error', 'unknown_error')}")
        return payload

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.credentials.token}"}

    @staticmethod
    def _parse_file(payload: dict[str, Any]) -> SlackFile:
        return SlackFile(
            file_id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            title=payload.get("title"),
            mimetype=payload.get("mimetype"),
            filetype=payload.get("filetype"),
            pretty_type=payload.get("pretty_type"),
            mode=payload.get("mode"),
            size_bytes=SlackProvider._coerce_int(payload.get("size")),
            created=SlackProvider._coerce_int(payload.get("created")),
            user_id=payload.get("user"),
            is_public=payload.get("is_public"),
            permalink=payload.get("permalink"),
            url_private=payload.get("url_private"),
            url_private_download=payload.get("url_private_download"),
            metadata=payload,
        )

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        return None if value in (None, "") else int(value)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in name).strip("_") or "slack-file"

    @staticmethod
    def _infer_suffix(url: str, mime_type: str | None) -> str:
        path_suffix = Path(urlsplit(url).path).suffix
        if path_suffix:
            return path_suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        return guessed or ""
