from tribe_setup.models import (
    CognitiveLoadScore,
    ComparisonMetric,
    ComparisonResult,
    EngagementSegmentation,
    ExportArtifact,
    ExportFormat,
    ExportPayload,
    InputKind,
    LanguageProcessingScore,
    NormalizedPredictions,
    PeakMoments,
    PredictionDiff,
    PredictionResult,
    PreparedInput,
    RegionActivations,
    TemporalCurve,
    TranslationReport,
    TranslationOutputKey,
    TribeConfig,
    TribePredictions,
    TribeSegments,
)

__all__ = [
    "CognitiveLoadScore",
    "ComparisonMetric",
    "ComparisonResult",
    "DataInput",
    "EngagementSegmentation",
    "ExportArtifact",
    "ExportFormat",
    "ExportPayload",
    "InputKind",
    "LanguageProcessingScore",
    "NormalizedPredictions",
    "PeakMoments",
    "PredictionDiff",
    "PredictionResult",
    "PreparedInput",
    "RegionActivations",
    "TemporalCurve",
    "TranslationReport",
    "TranslationOutputKey",
    "TribeConfig",
    "TribePredictions",
    "TribeRunnerUtils",
    "TribeSegments",
    "TribeRunner",
]


def __getattr__(name: str):
    if name == "DataInput":
        from libs.utils import DataInput

        return DataInput
    if name == "TribeRunner":
        from services.inference import TribeRunner

        return TribeRunner
    if name == "TribeRunnerUtils":
        from libs.utils import TribeRunnerUtils

        return TribeRunnerUtils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
