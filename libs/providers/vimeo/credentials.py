from __future__ import annotations

from dataclasses import dataclass

from libs.providers.base_credentials import BaseProviderCredentials


@dataclass(frozen=True, slots=True)
class VimeoCredentials(BaseProviderCredentials):
    provider_key = "vimeo"

    token: str

    def validate(self) -> None:
        if not self.token.strip():
            raise ValueError("Vimeo token must not be empty.")
