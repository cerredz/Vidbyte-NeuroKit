from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


PFC_REGIONS: tuple[str, ...] = (
    "G_front_sup",
    "G_front_middle",
    "G_front_inf-Opercular",
    "G_front_inf-Triangul",
)

LANGUAGE_REGIONS: tuple[str, ...] = (
    "G_front_inf-Opercular",
    "G_front_inf-Triangul",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Plan_tempo",
    "S_temporal_sup",
)

ROIIndexResolver = Callable[[str], Sequence[int]]


class TribeRunnerUtils:
    def __init__(self, roi_index_resolver: ROIIndexResolver | None = None, atlas_data_dir: str | Path | None = None) -> None:
        self._roi_index_resolver = roi_index_resolver
        self._atlas_data_dir = Path(atlas_data_dir).expanduser().resolve() if atlas_data_dir is not None else None
        self._roi_lookup: dict[str, np.ndarray] | None = None

    def get_temporal_curve(self, preds: np.ndarray, segments: pd.DataFrame | Sequence[Any]) -> dict[str, list[float]]:
        predictions = self._ensure_predictions(preds)
        segment_frame = self._segments_to_frame(segments)
        self._validate_timestep_alignment(predictions, segment_frame)

        curve = predictions.mean(axis=1)
        curve_normalized = self._normalize_to_percentage(curve)
        timestamps = segment_frame["onset"].to_numpy(dtype=float)
        return {
            "timestamps": timestamps.tolist(),
            "scores": curve_normalized.tolist(),
            "raw": curve.astype(float).tolist(),
        }

    def get_peak_moments(self, preds: np.ndarray, segments: pd.DataFrame | Sequence[Any], top_n: int = 3) -> list[dict[str, float | int]]:
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero.")

        curve_data = self.get_temporal_curve(preds, segments)
        scores = np.asarray(curve_data["scores"], dtype=float)
        timestamps = np.asarray(curve_data["timestamps"], dtype=float)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [
            {
                "rank": rank + 1,
                "timestamp_s": float(timestamps[index]),
                "score": float(scores[index]),
            }
            for rank, index in enumerate(top_indices)
        ]

    def get_region_activations(
        self,
        preds: np.ndarray,
        segments: pd.DataFrame | Sequence[Any],
        regions: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        predictions = self._ensure_predictions(preds)
        segment_frame = self._segments_to_frame(segments)
        self._validate_timestep_alignment(predictions, segment_frame)
        timestamps = segment_frame["onset"].to_numpy(dtype=float).tolist()

        result: dict[str, dict[str, Any]] = {}
        for region in regions:
            try:
                vertex_indices = self.get_roi_indices(region)
            except ValueError:
                result[region] = {"error": f'Region "{region}" not found in Destrieux atlas.'}
                continue

            self._validate_vertex_indices(predictions, vertex_indices, region)
            region_curve = predictions[:, vertex_indices].mean(axis=1)
            result[region] = {
                "timestamps": timestamps,
                "activation": region_curve.astype(float).tolist(),
                "n_vertices": int(len(vertex_indices)),
            }
        return result

    def get_cognitive_load_score(self, preds: np.ndarray, segments: pd.DataFrame | Sequence[Any]) -> dict[str, Any]:
        return self._get_named_region_score(
            preds=preds,
            segments=segments,
            regions=PFC_REGIONS,
            score_key="cognitive_load",
            include_peak_timestamp=True,
        )

    def get_language_processing_score(self, preds: np.ndarray, segments: pd.DataFrame | Sequence[Any]) -> dict[str, Any]:
        return self._get_named_region_score(
            preds=preds,
            segments=segments,
            regions=LANGUAGE_REGIONS,
            score_key="language_score",
            include_peak_timestamp=False,
        )

    def compare(
        self,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        segments_a: pd.DataFrame | Sequence[Any],
        segments_b: pd.DataFrame | Sequence[Any],
        metric: str = "engagement",
    ) -> dict[str, Any]:
        fn_map = {
            "engagement": self.get_temporal_curve,
            "cognitive_load": self.get_cognitive_load_score,
            "language": self.get_language_processing_score,
        }
        score_key_map = {
            "engagement": "scores",
            "cognitive_load": "cognitive_load",
            "language": "language_score",
        }
        if metric not in fn_map:
            raise ValueError(f"Unsupported compare metric '{metric}'. Expected one of {sorted(fn_map)}.")

        result_a = fn_map[metric](preds_a, segments_a)
        result_b = fn_map[metric](preds_b, segments_b)
        score_key = score_key_map[metric]
        mean_a = float(np.mean(result_a[score_key]))
        mean_b = float(np.mean(result_b[score_key]))
        winner = "tie" if np.isclose(mean_a, mean_b) else ("a" if mean_a > mean_b else "b")
        return {
            "metric": metric,
            "input_a": {"mean": mean_a, "data": result_a},
            "input_b": {"mean": mean_b, "data": result_b},
            "winner": winner,
            "delta": float(abs(mean_a - mean_b)),
        }

    def diff(self, preds_a: np.ndarray, preds_b: np.ndarray) -> dict[str, Any]:
        left = self._ensure_predictions(preds_a)
        right = self._ensure_predictions(preds_b)
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch: {left.shape} vs {right.shape}. Inputs must have the same number of timesteps and vertices."
            )

        delta = left - right
        return {
            "delta": delta,
            "mean_diff_per_timestep": delta.mean(axis=1).astype(float).tolist(),
            "mean_diff_per_vertex": delta.mean(axis=0).astype(float).tolist(),
            "abs_mean": float(np.abs(delta).mean()),
            "max_diff_vertex": int(np.abs(delta).mean(axis=0).argmax()),
        }

    def normalize(self, preds: np.ndarray, baseline_preds: np.ndarray) -> np.ndarray:
        predictions = self._ensure_predictions(preds)
        baseline = self._ensure_predictions(baseline_preds)
        if predictions.shape[1] != baseline.shape[1]:
            raise ValueError(
                f"Vertex mismatch: {predictions.shape[1]} vs {baseline.shape[1]}. Baseline must have the same number of vertices."
            )

        baseline_mean = baseline.mean(axis=0)
        return predictions - baseline_mean

    def segment_by_engagement(
        self,
        preds: np.ndarray,
        segments: pd.DataFrame | Sequence[Any],
        threshold: float = 50.0,
    ) -> dict[str, Any]:
        curve = self.get_temporal_curve(preds, segments)
        segment_frame = self._segments_to_frame(segments)
        scores = np.asarray(curve["scores"], dtype=float)
        timestamps = np.asarray(curve["timestamps"], dtype=float)
        end_timestamps = (segment_frame["onset"] + segment_frame["duration"]).to_numpy(dtype=float)
        high_mask = scores >= threshold
        low_mask = ~high_mask

        def get_spans(mask: np.ndarray) -> list[dict[str, float]]:
            spans: list[dict[str, float]] = []
            in_span = False
            start = 0.0
            for index, is_active in enumerate(mask):
                if is_active and not in_span:
                    start = float(timestamps[index])
                    in_span = True
                    continue
                if not is_active and in_span:
                    spans.append({"start_s": start, "end_s": float(end_timestamps[index - 1])})
                    in_span = False
            if in_span:
                spans.append({"start_s": start, "end_s": float(end_timestamps[-1])})
            return spans

        return {
            "threshold": float(threshold),
            "high_engagement": get_spans(high_mask),
            "low_engagement": get_spans(low_mask),
            "pct_high": float(high_mask.mean() * 100),
        }

    def export(self, result: Mapping[str, Any], format: str = "json", path: str | Path = "output") -> str:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_format = format.lower()

        if normalized_format == "json":
            target = output_path.with_suffix(".json")
            target.write_text(json.dumps(self._make_json_safe(result), indent=2), encoding="utf-8")
            return str(target)

        if normalized_format == "csv":
            frame = self._result_to_csv_frame(result)
            target = output_path.with_suffix(".csv")
            frame.to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL)
            return str(target)

        if normalized_format == "nifti":
            if "array" not in result and "delta" not in result:
                raise ValueError("NIfTI export requires a result payload with an 'array' or 'delta' key.")
            raise NotImplementedError(
                "NIfTI export is not wired in this wrapper yet. Use tribev2 surface-to-volume projection utilities for MNI export."
            )

        raise ValueError(f"Unsupported export format '{format}'. Expected one of ['csv', 'json', 'nifti'].")

    def get_roi_indices(self, region: str) -> np.ndarray:
        if self._roi_index_resolver is not None:
            indices = np.asarray(self._roi_index_resolver(region), dtype=int)
            if indices.size == 0:
                raise ValueError(f'Region "{region}" resolved to an empty vertex list.')
            return indices

        lookup = self._get_roi_lookup()
        try:
            return lookup[region]
        except KeyError as exc:
            raise ValueError(f'Region "{region}" not found in Destrieux atlas.') from exc

    def _get_named_region_score(
        self,
        preds: np.ndarray,
        segments: pd.DataFrame | Sequence[Any],
        regions: Sequence[str],
        score_key: str,
        *,
        include_peak_timestamp: bool,
    ) -> dict[str, Any]:
        predictions = self._ensure_predictions(preds)
        segment_frame = self._segments_to_frame(segments)
        self._validate_timestep_alignment(predictions, segment_frame)

        vertex_groups = [self.get_roi_indices(region) for region in regions]
        all_vertices = np.unique(np.concatenate([np.asarray(group, dtype=int) for group in vertex_groups]))
        self._validate_vertex_indices(predictions, all_vertices, ",".join(regions))
        curve = predictions[:, all_vertices].mean(axis=1)
        score = self._normalize_to_percentage(curve)
        timestamps = segment_frame["onset"].to_numpy(dtype=float)
        result = {
            "timestamps": timestamps.tolist(),
            score_key: score.tolist(),
            "mean_score": float(score.mean()) if score.size else 0.0,
        }
        if include_peak_timestamp:
            result["peak_timestamp_s"] = float(timestamps[int(score.argmax())]) if score.size else 0.0
        return result

    def _get_roi_lookup(self) -> dict[str, np.ndarray]:
        if self._roi_lookup is not None:
            return self._roi_lookup

        try:
            from nilearn import datasets
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "Region-based analytics require nilearn. Install nilearn to use Destrieux atlas lookups."
            ) from exc

        atlas = datasets.fetch_atlas_surf_destrieux(
            data_dir=str(self._atlas_data_dir) if self._atlas_data_dir is not None else None,
            verbose=0,
        )
        labels = [label.decode("utf-8") if isinstance(label, bytes) else str(label) for label in atlas["labels"]]
        map_left = np.asarray(atlas["map_left"], dtype=int)
        map_right = np.asarray(atlas["map_right"], dtype=int)
        left_vertex_count = int(map_left.shape[0])

        lookup: dict[str, np.ndarray] = {}
        for label_index, label in enumerate(labels):
            left_indices = np.flatnonzero(map_left == label_index)
            right_indices = np.flatnonzero(map_right == label_index) + left_vertex_count
            combined = np.concatenate((left_indices, right_indices))
            if combined.size:
                lookup[label] = combined

        self._roi_lookup = lookup
        return lookup

    @staticmethod
    def _ensure_predictions(preds: np.ndarray) -> np.ndarray:
        predictions = np.asarray(preds, dtype=float)
        if predictions.ndim != 2:
            raise ValueError(f"Expected preds to have shape (n_timesteps, n_vertices), got {predictions.shape}.")
        return predictions

    @staticmethod
    def _normalize_to_percentage(values: np.ndarray) -> np.ndarray:
        numeric_values = np.asarray(values, dtype=float)
        if numeric_values.size == 0:
            return numeric_values
        spread = float(numeric_values.max() - numeric_values.min())
        if np.isclose(spread, 0.0):
            return np.zeros_like(numeric_values, dtype=float)
        return (numeric_values - numeric_values.min()) / spread * 100.0

    @staticmethod
    def _validate_timestep_alignment(preds: np.ndarray, segments: pd.DataFrame) -> None:
        if preds.shape[0] != len(segments):
            raise ValueError(
                f"Timestep mismatch: preds has {preds.shape[0]} rows but segments has {len(segments)} rows."
            )

    @staticmethod
    def _validate_vertex_indices(preds: np.ndarray, vertex_indices: np.ndarray, region: str) -> None:
        if vertex_indices.size == 0:
            raise ValueError(f'Region "{region}" resolved to an empty vertex list.')
        if int(vertex_indices.max()) >= preds.shape[1]:
            raise ValueError(
                f'Region "{region}" references vertex index {int(vertex_indices.max())}, but preds only has {preds.shape[1]} vertices.'
            )

    @staticmethod
    def _segments_to_frame(segments: pd.DataFrame | Sequence[Any]) -> pd.DataFrame:
        if isinstance(segments, pd.DataFrame):
            frame = segments.copy()
        else:
            records: list[dict[str, Any]] = []
            for segment in segments:
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
                raise ValueError("Segments must include an 'onset', 'offset', or 'start' field.")

        if "duration" not in frame.columns:
            if "stop" in frame.columns:
                frame["duration"] = frame["stop"] - frame["onset"]
            else:
                raise ValueError("Segments must include a 'duration' field or a 'stop' field.")

        return frame[["onset", "duration"]].astype(float).reset_index(drop=True)

    @classmethod
    def _result_to_csv_frame(cls, result: Mapping[str, Any]) -> pd.DataFrame:
        list_fields = {key: value for key, value in result.items() if isinstance(value, list)}
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

    @classmethod
    def _make_json_safe(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): cls._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._make_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._make_json_safe(item) for item in value]
        return value
