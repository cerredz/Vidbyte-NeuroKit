from __future__ import annotations

from typing import Protocol

import pandas as pd


class SupportsEventsDataFrame(Protocol):
    def get_events_dataframe(self, text_path: str | None = None, audio_path: str | None = None, video_path: str | None = None) -> pd.DataFrame:
        ...
