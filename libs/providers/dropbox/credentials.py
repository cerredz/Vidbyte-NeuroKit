from __future__ import annotations

from dataclasses import dataclass

from libs.providers.base_credentials import BaseProviderCredentials


@dataclass(frozen=True, slots=True)
class DropboxCredentials(BaseProviderCredentials):
    provider_key = "dropbox"

    token: str

    def validate(self) -> None:
        if not self.token.strip():
            raise ValueError("Dropbox token must not be empty.")
