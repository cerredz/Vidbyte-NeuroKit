from __future__ import annotations

from libs.providers.meta_marketing import MetaMarketingProvider

from .base import ProviderRunnerBase


class MetaMarketingRunner(ProviderRunnerBase):
    provider_name = "meta-ads"

    def __init__(self, provider: MetaMarketingProvider, tribe_runner=None) -> None:
        super().__init__(provider=provider, tribe_runner=tribe_runner)

    def analyze_creative(self, creative_id: str, *, metrics=None, save_to=None):
        return self.analyze_asset(self.provider.fetch_media(creative_id), metrics=metrics, save_to=save_to)

    def analyze_campaign(self, campaign_id: str, *, metrics=None, sort_by: str | None = None):
        return self.analyze_assets(self.provider.fetch_campaign_media(campaign_id), metrics=metrics, sort_by=sort_by)
