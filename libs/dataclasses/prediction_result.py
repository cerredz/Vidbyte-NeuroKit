from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from libs.enums import InputKind


@dataclass(frozen=True, slots=True)
class PredictionResult:
    input_path: Path
    model_input_path: Path
    input_kind: InputKind
    events: pd.DataFrame
    brain_stimulus: np.ndarray
    segments: list[Any]

    def brain_stimulus_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            self.brain_stimulus,
            columns=[f"vertex_{index}" for index in range(self.brain_stimulus.shape[1])],
        )
        frame.insert(0, "timepoint", range(len(frame)))
        return frame
