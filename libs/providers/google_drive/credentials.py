from __future__ import annotations

from dataclasses import dataclass

from libs.providers.base_credentials import BaseProviderCredentials


@dataclass(frozen=True, slots=True)
class GoogleDriveCredentials(BaseProviderCredentials):
    provider_key = "google-drive"

    token: str

    def validate(self) -> None:
        if not self.token.strip():
            raise ValueError("Google Drive token must not be empty.")
