from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from libs.dataclasses.provider_models import DownloadedProviderAsset, VimeoAccount, VimeoVideo, VimeoVideoFile
from libs.providers.http_client import ProviderHttpClient

from .credentials import VimeoCredentials


class VimeoProvider:
    def __init__(
        self,
        credentials: VimeoCredentials | str,
        *,
        http_client: ProviderHttpClient | None = None,
        download_dir: str | Path | None = None,
    ) -> None:
        self.credentials = credentials if isinstance(credentials, VimeoCredentials) else VimeoCredentials(token=credentials)
        self.http_client = http_client or ProviderHttpClient()
        self.download_dir = Path(download_dir or Path.cwd() / "cache" / "providers" / "vimeo").expanduser().resolve()

    def get_account(self) -> VimeoAccount:
        payload = self.http_client.get_json("https://api.vimeo.com/me", headers=self._headers())
        return VimeoAccount(
            user_id=str(payload.get("uri", "")).split("/")[-1],
            name=str(payload.get("name", "")),
            uri=str(payload.get("uri", "")),
            account_type=self._coerce_nested(payload, "account", "type"),
        )

    def get_video(self, video_id: str) -> VimeoVideo:
        payload = self.http_client.get_json(
            f"https://api.vimeo.com/videos/{video_id}",
            headers=self._headers(),
            params={
                "fields": "uri,name,description,link,duration,width,height,created_time,modified_time,files,download,pictures",
            },
        )
        files = tuple(self._parse_video_file(item) for item in payload.get("files", []) if isinstance(item, dict))
        download_links = tuple(
            str(item["link"])
            for item in payload.get("download", [])
            if isinstance(item, dict) and item.get("link")
        )
        return VimeoVideo(
            video_id=str(video_id),
            uri=str(payload.get("uri", "")),
            name=str(payload.get("name", video_id)),
            description=payload.get("description"),
            duration_s=self._coerce_float(payload.get("duration")),
            width=self._coerce_int(payload.get("width")),
            height=self._coerce_int(payload.get("height")),
            created_time=payload.get("created_time"),
            modified_time=payload.get("modified_time"),
            embed_link=payload.get("link"),
            files=files,
            download_links=download_links,
            metadata=payload,
        )

    def list_videos(self, *, page: int = 1, per_page: int = 25) -> tuple[VimeoVideo, ...]:
        payload = self.http_client.get_json(
            "https://api.vimeo.com/me/videos",
            headers=self._headers(),
            params={
                "page": page,
                "per_page": per_page,
                "fields": "uri,name,description,link,duration,width,height,created_time,modified_time,files,download,pictures",
            },
        )
        return tuple(
            self.get_video(str(item.get("uri", "")).split("/")[-1])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("uri")
        )

    def fetch_media(self, video_id: str) -> DownloadedProviderAsset:
        video = self.get_video(video_id)
        selected = self._select_download_url(video)
        suffix = self._infer_suffix(selected[0], selected[1], default=".mp4")
        destination = self.download_dir / f"{video.video_id}{suffix}"
        self.http_client.download(selected[0], destination, headers=self._headers())
        return DownloadedProviderAsset(
            provider="vimeo",
            remote_id=video.video_id,
            name=video.name,
            local_path=destination,
            mime_type=selected[1],
            source_url=selected[0],
            size_bytes=selected[2],
            metadata={"video": video},
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credentials.token}",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        }

    @staticmethod
    def _parse_video_file(payload: dict[str, Any]) -> VimeoVideoFile:
        return VimeoVideoFile(
            link=str(payload.get("link", "")),
            public_name=str(payload.get("public_name", "")),
            quality=str(payload.get("quality", "")),
            rendition=str(payload.get("rendition", "")),
            type=payload.get("type"),
            width=VimeoProvider._coerce_int(payload.get("width")),
            height=VimeoProvider._coerce_int(payload.get("height")),
            fps=VimeoProvider._coerce_float(payload.get("fps")),
            size_bytes=VimeoProvider._coerce_int(payload.get("size")),
            source_link=payload.get("source_link"),
            expires=payload.get("expires"),
        )

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        return None if value in (None, "") else int(value)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        return None if value in (None, "") else float(value)

    @staticmethod
    def _coerce_nested(payload: dict[str, Any], *keys: str) -> str | None:
        current: Any = payload
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return None if current is None else str(current)

    @staticmethod
    def _infer_suffix(url: str, mime_type: str | None, *, default: str) -> str:
        path_suffix = Path(urlsplit(url).path).suffix
        if path_suffix:
            return path_suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        return guessed or default

    @staticmethod
    def _select_download_url(video: VimeoVideo) -> tuple[str, str | None, int | None]:
        preferred_file = next((item for item in video.files if item.source_link), None)
        if preferred_file is not None:
            return preferred_file.source_link or preferred_file.link, preferred_file.type, preferred_file.size_bytes
        best_file = next((item for item in video.files if item.link), None)
        if best_file is not None:
            return best_file.link, best_file.type, best_file.size_bytes
        if video.download_links:
            return video.download_links[0], "video/mp4", None
        raise ValueError(f"Vimeo video {video.video_id} does not expose a downloadable file link.")
