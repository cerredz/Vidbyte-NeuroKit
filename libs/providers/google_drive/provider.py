from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from libs.dataclasses.provider_models import DownloadedProviderAsset, GoogleDriveAccount, GoogleDriveFile
from libs.enums import ProviderBaseUrl
from libs.providers.http_client import ProviderHttpClient

from .credentials import GoogleDriveCredentials


GOOGLE_DRIVE_FIELDS = ",".join(
    [
        "id",
        "name",
        "mimeType",
        "fileExtension",
        "createdTime",
        "modifiedTime",
        "size",
        "thumbnailLink",
        "webViewLink",
        "webContentLink",
        "capabilities/canDownload",
    ]
)


class GoogleDriveProvider:
    def __init__(
        self,
        credentials: GoogleDriveCredentials | str,
        *,
        http_client: ProviderHttpClient | None = None,
        download_dir: str | Path | None = None,
    ) -> None:
        self.credentials = credentials if isinstance(credentials, GoogleDriveCredentials) else GoogleDriveCredentials(token=credentials)
        self.http_client = http_client or ProviderHttpClient()
        self.download_dir = Path(download_dir or Path.cwd() / "cache" / "providers" / "google-drive").expanduser().resolve()

    def get_account(self) -> GoogleDriveAccount:
        payload = self.http_client.get_json(
            f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/about",
            headers=self._headers(),
            params={"fields": "user,storageQuota"},
        )
        user = payload.get("user", {}) if isinstance(payload.get("user"), dict) else {}
        quota = payload.get("storageQuota", {}) if isinstance(payload.get("storageQuota"), dict) else {}
        return GoogleDriveAccount(
            display_name=user.get("displayName"),
            email_address=user.get("emailAddress"),
            permission_id=user.get("permissionId"),
            storage_quota_limit=self._coerce_int(quota.get("limit")),
            storage_quota_usage=self._coerce_int(quota.get("usage")),
        )

    def get_file(self, file_id: str) -> GoogleDriveFile:
        payload = self.http_client.get_json(
            f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/files/{file_id}",
            headers=self._headers(),
            params={"fields": GOOGLE_DRIVE_FIELDS, "supportsAllDrives": "true"},
        )
        return self._parse_file(payload)

    def list_files(self, *, query: str | None = None, page_size: int = 100) -> tuple[GoogleDriveFile, ...]:
        payload = self.http_client.get_json(
            f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/files",
            headers=self._headers(),
            params={
                "fields": f"files({GOOGLE_DRIVE_FIELDS})",
                "pageSize": page_size,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
                "q": query,
            },
        )
        return tuple(self._parse_file(item) for item in payload.get("files", []) if isinstance(item, dict))

    def fetch_media(self, file_id: str) -> DownloadedProviderAsset:
        file = self.get_file(file_id)
        if file.is_google_workspace_document:
            if file.mime_type != "application/vnd.google-apps.document":
                raise ValueError(
                    f"Google Drive file {file.file_id} is a Google Workspace document with MIME type {file.mime_type!r}. "
                    "Only Google Docs export to plain text is supported in this initial implementation."
                )
            destination = self.download_dir / f"{self._safe_name(file.name)}.txt"
            payload = self.http_client.request_bytes(
                "GET",
                f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/files/{file.file_id}/export",
                headers=self._headers(),
                params={"mimeType": "text/plain"},
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            mime_type = "text/plain"
            source_url = f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/files/{file.file_id}/export"
        else:
            if file.can_download is False:
                raise ValueError(f"Google Drive file {file.file_id} is not downloadable for the connected account.")
            suffix = self._infer_suffix(file)
            destination = self.download_dir / f"{self._safe_name(file.name)}{suffix}"
            self.http_client.download(
                f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/files/{file.file_id}",
                destination,
                headers=self._headers(),
                params={"alt": "media", "supportsAllDrives": "true"},
            )
            mime_type = file.mime_type
            source_url = f"{ProviderBaseUrl.GOOGLE_DRIVE_API.value}/files/{file.file_id}?alt=media"
        return DownloadedProviderAsset(
            provider="google-drive",
            remote_id=file.file_id,
            name=file.name,
            local_path=destination,
            mime_type=mime_type,
            source_url=source_url,
            size_bytes=file.size_bytes,
            metadata={"file": file},
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.credentials.token}"}

    @staticmethod
    def _parse_file(payload: dict[str, Any]) -> GoogleDriveFile:
        mime_type = str(payload.get("mimeType", ""))
        capabilities = payload.get("capabilities", {}) if isinstance(payload.get("capabilities"), dict) else {}
        return GoogleDriveFile(
            file_id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            mime_type=mime_type,
            file_extension=payload.get("fileExtension"),
            created_time=payload.get("createdTime"),
            modified_time=payload.get("modifiedTime"),
            size_bytes=GoogleDriveProvider._coerce_int(payload.get("size")),
            thumbnail_link=payload.get("thumbnailLink"),
            web_view_link=payload.get("webViewLink"),
            web_content_link=payload.get("webContentLink"),
            can_download=capabilities.get("canDownload"),
            is_google_workspace_document=mime_type.startswith("application/vnd.google-apps."),
            metadata=payload,
        )

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        return None if value in (None, "") else int(value)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in name).strip("_") or "drive-file"

    @staticmethod
    def _infer_suffix(file: GoogleDriveFile) -> str:
        if file.file_extension:
            return f".{file.file_extension.lstrip('.')}"
        guessed = mimetypes.guess_extension(file.mime_type)
        return guessed or ""
