from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from libs.enums import ComparisonMetric, ComparisonWinner, ExportFormat


TRIBE_FSAVERAGE5_VERTEX_COUNT = 20484


@dataclass(frozen=True, slots=True)
class TribePredictions:
    values: np.ndarray

    def __post_init__(self) -> None:
        predictions = np.asarray(self.values, dtype=float)
        if predictions.ndim != 2:
            raise ValueError(f"Expected TRIBE predictions with shape (n_timesteps, n_vertices), got {predictions.shape}.")
        if predictions.shape[1] != TRIBE_FSAVERAGE5_VERTEX_COUNT:
            raise ValueError(
                f"Expected TRIBE predictions on fsaverage5 with {TRIBE_FSAVERAGE5_VERTEX_COUNT} vertices, got {predictions.shape[1]}."
            )
        object.__setattr__(self, "values", predictions)

    @property
    def n_timesteps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class NormalizedPredictions(TribePredictions):
    pass


@dataclass(frozen=True, slots=True)
class TribeSegments:
    raw_segments: tuple[Any, ...]
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        if not {"onset", "duration"}.issubset(self.frame.columns):
            raise ValueError("TribeSegments.frame must include 'onset' and 'duration' columns.")
        normalized = self.frame[["onset", "duration"]].astype(float).reset_index(drop=True)
        if (normalized["duration"] < 0).any():
            raise ValueError("TRIBE segments cannot contain negative durations.")
        if len(self.raw_segments) != len(normalized):
            raise ValueError(
                f"TRIBE segments mismatch: raw_segments has {len(self.raw_segments)} items but frame has {len(normalized)} rows."
            )
        object.__setattr__(self, "frame", normalized)

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.frame["onset"].tolist())

    @property
    def durations(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.frame["duration"].tolist())

    def to_records(self) -> tuple[dict[str, float], ...]:
        return tuple({"onset": onset, "duration": duration} for onset, duration in zip(self.timestamps, self.durations))

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True, slots=True)
class TemporalCurve:
    timestamps: tuple[float, ...]
    scores: tuple[float, ...]
    raw: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.timestamps) == len(self.scores) == len(self.raw)):
            raise ValueError("TemporalCurve fields must all have the same length.")


@dataclass(frozen=True, slots=True)
class PeakMoment:
    rank: int
    timestamp_s: float
    score: float

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("PeakMoment.rank must be greater than zero.")


@dataclass(frozen=True, slots=True)
class PeakMoments:
    items: tuple[PeakMoment, ...]


@dataclass(frozen=True, slots=True)
class RegionActivation:
    region: str
    timestamps: tuple[float, ...] = ()
    activation: tuple[float, ...] = ()
    n_vertices: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error is None and len(self.timestamps) != len(self.activation):
            raise ValueError("RegionActivation timestamps and activation lengths must match.")


@dataclass(frozen=True, slots=True)
class RegionActivations:
    items: dict[str, RegionActivation]


@dataclass(frozen=True, slots=True)
class CognitiveLoadScore:
    timestamps: tuple[float, ...]
    cognitive_load: tuple[float, ...]
    mean_score: float
    peak_timestamp_s: float

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.cognitive_load):
            raise ValueError("CognitiveLoadScore timestamps and cognitive_load lengths must match.")


@dataclass(frozen=True, slots=True)
class LanguageProcessingScore:
    timestamps: tuple[float, ...]
    language_score: tuple[float, ...]
    mean_score: float

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.language_score):
            raise ValueError("LanguageProcessingScore timestamps and language_score lengths must match.")


@dataclass(frozen=True, slots=True)
class ComparisonInput:
    mean: float
    data: TemporalCurve | CognitiveLoadScore | LanguageProcessingScore


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    metric: ComparisonMetric
    input_a: ComparisonInput
    input_b: ComparisonInput
    winner: ComparisonWinner
    delta: float


@dataclass(frozen=True, slots=True)
class PredictionDiff:
    delta: np.ndarray
    mean_diff_per_timestep: tuple[float, ...]
    mean_diff_per_vertex: tuple[float, ...]
    abs_mean: float
    max_diff_vertex: int

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta, dtype=float)
        if delta.ndim != 2:
            raise ValueError(f"PredictionDiff.delta must be 2D, got {delta.shape}.")
        object.__setattr__(self, "delta", delta)


@dataclass(frozen=True, slots=True)
class EngagementWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError("EngagementWindow end_s must be greater than or equal to start_s.")


@dataclass(frozen=True, slots=True)
class EngagementSegmentation:
    threshold: float
    high_engagement: tuple[EngagementWindow, ...]
    low_engagement: tuple[EngagementWindow, ...]
    pct_high: float


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    path: Path
    format: ExportFormat

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True, slots=True)
class ExportPayload:
    array: np.ndarray | None = None
    delta: np.ndarray | None = None
    segments: tuple[dict[str, float], ...] = ()
