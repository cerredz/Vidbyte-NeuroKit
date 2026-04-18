from __future__ import annotations

from libs.providers.google_drive import GoogleDriveProvider

from .base import ProviderRunnerBase


class GoogleDriveRunner(ProviderRunnerBase):
    provider_name = "google-drive"

    def __init__(self, provider: GoogleDriveProvider, tribe_runner=None) -> None:
        super().__init__(provider=provider, tribe_runner=tribe_runner)

    def analyze_file(self, file_id: str, *, metrics=None, save_to=None):
        return self.analyze_asset(self.provider.fetch_media(file_id), metrics=metrics, save_to=save_to)

    def compare_files(self, left_file_id: str, right_file_id: str, *, metric: str):
        return self.compare_assets(
            self.provider.fetch_media(left_file_id),
            self.provider.fetch_media(right_file_id),
            metric=metric,
        )
