from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class SupportsTribeModel(Protocol):
    def get_events_dataframe(self, text_path: str | None = None, audio_path: str | None = None, video_path: str | None = None) -> pd.DataFrame:
        ...

    def predict(self, events: pd.DataFrame, verbose: bool = True) -> tuple[np.ndarray, list[Any]]:
        ...
