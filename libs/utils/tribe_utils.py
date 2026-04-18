from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from libs.dataclasses import ExportPayload, PredictionResult, TribePredictions, TribeSegments
from libs.enums import DestrieuxRegion, TranslationOutputKey


def build_segments_frame(raw_segments: Sequence[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for segment in raw_segments:
        if isinstance(segment, Mapping):
            records.append(dict(segment))
            continue
        records.append(
            {
                field: getattr(segment, field)
                for field in ("onset", "offset", "start", "duration", "stop")
                if getattr(segment, field, None) is not None
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame({"onset": pd.Series(dtype=float), "duration": pd.Series(dtype=float)})
    if "onset" not in frame.columns:
        if "offset" in frame.columns:
            frame["onset"] = frame["offset"]
        elif "start" in frame.columns:
            frame["onset"] = frame["start"]
        else:
            raise ValueError("TRIBE segments must include an 'onset', 'offset', or 'start' field.")
    if "duration" not in frame.columns:
        if "stop" in frame.columns:
            frame["duration"] = frame["stop"] - frame["onset"]
        else:
            raise ValueError("TRIBE segments must include a 'duration' field or a 'stop' field.")
    return frame[["onset", "duration"]].astype(float).reset_index(drop=True)


def build_tribe_segments(raw_segments: Sequence[Any]) -> TribeSegments:
    return TribeSegments(raw_segments=tuple(raw_segments), frame=build_segments_frame(raw_segments))


def normalize_to_percentage(values: np.ndarray) -> np.ndarray:
    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0:
        return numeric_values
    spread = float(numeric_values.max() - numeric_values.min())
    if np.isclose(spread, 0.0):
        return np.zeros_like(numeric_values, dtype=float)
    return (numeric_values - numeric_values.min()) / spread * 100.0


def validate_timestep_alignment(predictions: TribePredictions, segments: TribeSegments) -> None:
    if predictions.n_timesteps != len(segments):
        raise ValueError(
            f"Timestep mismatch: predictions has {predictions.n_timesteps} rows but segments has {len(segments)} rows."
        )


def validate_vertex_indices(predictions: TribePredictions, vertex_indices: np.ndarray, region: str) -> None:
    if vertex_indices.size == 0:
        raise ValueError(f'Region "{region}" resolved to an empty vertex list.')
    if int(vertex_indices.max()) >= predictions.n_vertices:
        raise ValueError(
            f'Region "{region}" references vertex index {int(vertex_indices.max())}, but predictions only has {predictions.n_vertices} vertices.'
        )


def coerce_region_name(region: DestrieuxRegion | str) -> str:
    return region.value if isinstance(region, DestrieuxRegion) else str(region)


def coerce_translation_key(raw_key: TranslationOutputKey | str) -> TranslationOutputKey:
    if isinstance(raw_key, TranslationOutputKey):
        return raw_key
    try:
        return TranslationOutputKey(str(raw_key).lower())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported translation key '{raw_key}'. Expected one of {[member.value for member in TranslationOutputKey]}."
        ) from exc


def resolve_translation_keys(outputs: TranslationOutputKey | str | Sequence[TranslationOutputKey | str]) -> list[TranslationOutputKey]:
    raw_outputs = [outputs] if isinstance(outputs, (TranslationOutputKey, str)) else list(outputs)
    return [coerce_translation_key(output) for output in raw_outputs]


def normalize_translation_options(
    options: Mapping[TranslationOutputKey | str, Mapping[str, Any]],
) -> dict[TranslationOutputKey, Mapping[str, Any]]:
    return {coerce_translation_key(raw_key): value for raw_key, value in options.items()}


def require_translation_operand(output_key: TranslationOutputKey, params: dict[str, Any], operand_name: str) -> Any:
    if operand_name not in params:
        raise ValueError(
            f"translate(..., '{output_key.value}') requires options={{'{output_key.value}': "
            f"{{'{operand_name}': ...}}}}."
        )
    return params.pop(operand_name)


def require_segments(segments: TribeSegments | None) -> TribeSegments:
    if segments is None:
        raise ValueError("TRIBE segments are required for this translation output.")
    return segments


def resolve_prediction_artifacts(
    result_or_predictions: PredictionResult | TribePredictions,
    segments: TribeSegments | Sequence[Any] | None = None,
    *,
    require_segments: bool,
) -> tuple[TribePredictions, TribeSegments | None]:
    if isinstance(result_or_predictions, PredictionResult):
        if segments is not None:
            raise ValueError("Do not pass segments when supplying a PredictionResult.")
        return TribePredictions(result_or_predictions.brain_stimulus), build_tribe_segments(result_or_predictions.segments)

    resolved_segments: TribeSegments | None
    if isinstance(segments, TribeSegments):
        resolved_segments = segments
    elif segments is None:
        resolved_segments = None
    else:
        resolved_segments = build_tribe_segments(segments)

    if require_segments and resolved_segments is None:
        raise ValueError("TRIBE segments are required for this translation output.")
    return result_or_predictions, resolved_segments


def build_export_payload(predictions: TribePredictions, segments: TribeSegments | None = None) -> ExportPayload:
    serialized_segments = segments.to_records() if segments is not None else ()
    return ExportPayload(array=predictions.values, segments=serialized_segments)


def to_json_safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Mapping):
        return {str(key): to_json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [to_json_safe_value(item) for item in sorted(value)]
    if hasattr(value, "value") and value.__class__.__module__.startswith("libs.enums"):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_safe_value(getattr(value, field.name))
            for field in fields(value)
        }
    return value


def result_to_csv_frame(result: Any) -> pd.DataFrame:
    payload = to_json_safe_value(result)
    if not isinstance(payload, Mapping):
        raise ValueError("CSV export requires a mapping-like result payload.")

    list_fields = {key: value for key, value in payload.items() if isinstance(value, list)}
    if not list_fields:
        raise ValueError("CSV export requires at least one top-level list field in the result payload.")
    if len(list_fields) == 1:
        only_key, only_value = next(iter(list_fields.items()))
        if only_value and all(isinstance(item, Mapping) for item in only_value):
            return pd.DataFrame(only_value)
        return pd.DataFrame({only_key: only_value})

    lengths = {len(value) for value in list_fields.values()}
    if len(lengths) != 1:
        raise ValueError("CSV export requires top-level list fields to have the same length.")
    return pd.DataFrame(list_fields)
