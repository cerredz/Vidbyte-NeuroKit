from __future__ import annotations

from dataclasses import dataclass

from libs.providers.base_credentials import BaseProviderCredentials


@dataclass(frozen=True, slots=True)
class SlackCredentials(BaseProviderCredentials):
    provider_key = "slack"

    token: str

    def validate(self) -> None:
        if not self.token.strip():
            raise ValueError("Slack token must not be empty.")
