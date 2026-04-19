from __future__ import annotations

from libs.providers.dropbox import DropboxProvider

from .base import ProviderRunnerBase


class DropboxRunner(ProviderRunnerBase):
    provider_name = "dropbox"

    def __init__(self, provider: DropboxProvider, tribe_runner=None) -> None:
        super().__init__(provider=provider, tribe_runner=tribe_runner)

    def analyze_file(self, path: str, *, metrics=None, save_to=None):
        return self.analyze_asset(self.provider.fetch_media(path), metrics=metrics, save_to=save_to)

    def compare_files(self, left_path: str, right_path: str, *, metric: str):
        return self.compare_assets(self.provider.fetch_media(left_path), self.provider.fetch_media(right_path), metric=metric)
