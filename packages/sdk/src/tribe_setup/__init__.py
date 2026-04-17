from tribe_setup.models import InputKind, PredictionResult, PreparedInput

__all__ = [
    "DataInput",
    "InputKind",
    "PredictionResult",
    "PreparedInput",
    "TribeRunner",
]


def __getattr__(name: str):
    if name == "DataInput":
        from libs.utils import DataInput

        return DataInput
    if name == "TribeRunner":
        from services.inference import TribeRunner

        return TribeRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
