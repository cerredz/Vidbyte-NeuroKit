from __future__ import annotations

from dataclasses import dataclass

from libs.dataclasses.tribe_analysis import (
    CognitiveLoadScore,
    ComparisonResult,
    EngagementSegmentation,
    ExportArtifact,
    LanguageProcessingScore,
    NormalizedPredictions,
    PeakMoments,
    PredictionDiff,
    RegionActivations,
    TemporalCurve,
)


@dataclass(frozen=True, slots=True)
class TranslationReport:
    temporal: TemporalCurve
    peak: PeakMoments
    regions: RegionActivations
    cognitive: CognitiveLoadScore
    language: LanguageProcessingScore
    compare: ComparisonResult
    diff: PredictionDiff
    normalize: NormalizedPredictions
    segment: EngagementSegmentation
    export: ExportArtifact
