from __future__ import annotations

from dataclasses import dataclass

from libs.providers.base_credentials import BaseProviderCredentials


@dataclass(frozen=True, slots=True)
class MetaMarketingCredentials(BaseProviderCredentials):
    provider_key = "meta-ads"

    token: str
    account_id: str

    def validate(self) -> None:
        if not self.token.strip():
            raise ValueError("Meta Marketing token must not be empty.")
        if not self.account_id.strip():
            raise ValueError("Meta Marketing account_id must not be empty.")
        if not self.account_id.startswith("act_"):
            raise ValueError("Meta Marketing account_id must start with 'act_'.")
