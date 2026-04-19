from __future__ import annotations

from libs.providers.vimeo import VimeoProvider

from .base import ProviderRunnerBase


class VimeoRunner(ProviderRunnerBase):
    provider_name = "vimeo"

    def __init__(self, provider: VimeoProvider, tribe_runner=None) -> None:
        super().__init__(provider=provider, tribe_runner=tribe_runner)

    def analyze_video(self, video_id: str, *, metrics=None, save_to=None):
        return self.analyze_asset(self.provider.fetch_media(video_id), metrics=metrics, save_to=save_to)

    def compare_videos(self, left_video_id: str, right_video_id: str, *, metric: str):
        return self.compare_assets(
            self.provider.fetch_media(left_video_id),
            self.provider.fetch_media(right_video_id),
            metric=metric,
        )
