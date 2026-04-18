from tribe_setup.models import InputKind, PredictionResult, PreparedInput, TranslationOutputKey, TribeConfig

__all__ = [
    "DataInput",
    "InputKind",
    "PredictionResult",
    "PreparedInput",
    "TranslationOutputKey",
    "TribeConfig",
    "TribeRunnerUtils",
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
