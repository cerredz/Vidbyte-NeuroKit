from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from libs.dataclasses import (
    CognitiveLoadScore,
    ComparisonInput,
    ComparisonResult,
    EngagementSegmentation,
    EngagementWindow,
    ExportArtifact,
    ExportPayload,
    LanguageProcessingScore,
    NormalizedPredictions,
    PeakMoment,
    PeakMoments,
    PredictionDiff,
    RegionActivation,
    RegionActivations,
    TemporalCurve,
    TribePredictions,
    TribeSegments,
)
from libs.enums import (
    LANGUAGE_REGIONS,
    PFC_REGIONS,
    ComparisonMetric,
    ComparisonWinner,
    DestrieuxRegion,
    ExportFormat,
)
from libs.utils.local_file_exporter import LocalFileExporter
from libs.utils.tribe_utils import (
    coerce_region_name,
    normalize_to_percentage,
    validate_timestep_alignment,
    validate_vertex_indices,
)


ROIIndexResolver = Callable[[str], Sequence[int]]


class TribeRunnerUtils:
    def __init__(
        self,
        roi_index_resolver: ROIIndexResolver | None = None,
        atlas_data_dir: str | Path | None = None,
        file_exporter: LocalFileExporter | None = None,
    ) -> None:
        self._roi_index_resolver = roi_index_resolver
        self._atlas_data_dir = Path(atlas_data_dir).expanduser().resolve() if atlas_data_dir is not None else None
        self._roi_lookup: dict[str, np.ndarray] | None = None
        self.file_exporter = file_exporter or LocalFileExporter()

    def get_temporal_curve(self, predictions: TribePredictions, segments: TribeSegments) -> TemporalCurve:
        validate_timestep_alignment(predictions, segments)
        curve = predictions.values.mean(axis=1)
        return TemporalCurve(
            timestamps=segments.timestamps,
            scores=tuple(float(value) for value in normalize_to_percentage(curve).tolist()),
            raw=tuple(float(value) for value in curve.astype(float).tolist()),
        )

    def get_peak_moments(self, predictions: TribePredictions, segments: TribeSegments, top_n: int = 3) -> PeakMoments:
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero.")

        curve = self.get_temporal_curve(predictions, segments)
        scores = np.asarray(curve.scores, dtype=float)
        timestamps = np.asarray(curve.timestamps, dtype=float)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return PeakMoments(
            items=tuple(
                PeakMoment(rank=rank + 1, timestamp_s=float(timestamps[index]), score=float(scores[index]))
                for rank, index in enumerate(top_indices)
            )
        )

    def get_region_activations(
        self,
        predictions: TribePredictions,
        segments: TribeSegments,
        regions: Sequence[DestrieuxRegion | str],
    ) -> RegionActivations:
        validate_timestep_alignment(predictions, segments)
        activations: dict[str, RegionActivation] = {}
        for region in regions:
            region_name = coerce_region_name(region)
            try:
                vertex_indices = self.get_roi_indices(region_name)
            except ValueError:
                activations[region_name] = RegionActivation(
                    region=region_name,
                    error=f'Region "{region_name}" not found in Destrieux atlas.',
                )
                continue
            validate_vertex_indices(predictions, vertex_indices, region_name)

            region_curve = predictions.values[:, vertex_indices].mean(axis=1)
            activations[region_name] = RegionActivation(
                region=region_name,
                timestamps=segments.timestamps,
                activation=tuple(float(value) for value in region_curve.astype(float).tolist()),
                n_vertices=int(len(vertex_indices)),
            )
        return RegionActivations(items=activations)

    def get_cognitive_load_score(self, predictions: TribePredictions, segments: TribeSegments) -> CognitiveLoadScore:
        curve = self._get_named_region_curve(predictions, segments, PFC_REGIONS)
        scores = normalize_to_percentage(curve)
        return CognitiveLoadScore(
            timestamps=segments.timestamps,
            cognitive_load=tuple(float(value) for value in scores.tolist()),
            mean_score=float(scores.mean()) if scores.size else 0.0,
            peak_timestamp_s=float(segments.timestamps[int(scores.argmax())]) if scores.size else 0.0,
        )

    def get_language_processing_score(self, predictions: TribePredictions, segments: TribeSegments) -> LanguageProcessingScore:
        curve = self._get_named_region_curve(predictions, segments, LANGUAGE_REGIONS)
        scores = normalize_to_percentage(curve)
        return LanguageProcessingScore(
            timestamps=segments.timestamps,
            language_score=tuple(float(value) for value in scores.tolist()),
            mean_score=float(scores.mean()) if scores.size else 0.0,
        )

    def compare(
        self,
        predictions_a: TribePredictions,
        predictions_b: TribePredictions,
        segments_a: TribeSegments,
        segments_b: TribeSegments,
        metric: ComparisonMetric = ComparisonMetric.ENGAGEMENT,
    ) -> ComparisonResult:
        fn_map = {
            ComparisonMetric.ENGAGEMENT: self.get_temporal_curve,
            ComparisonMetric.COGNITIVE_LOAD: self.get_cognitive_load_score,
            ComparisonMetric.LANGUAGE: self.get_language_processing_score,
        }
        score_getter = {
            ComparisonMetric.ENGAGEMENT: lambda result: result.scores,
            ComparisonMetric.COGNITIVE_LOAD: lambda result: result.cognitive_load,
            ComparisonMetric.LANGUAGE: lambda result: result.language_score,
        }

        result_a = fn_map[metric](predictions_a, segments_a)
        result_b = fn_map[metric](predictions_b, segments_b)
        mean_a = float(np.mean(score_getter[metric](result_a)))
        mean_b = float(np.mean(score_getter[metric](result_b)))
        winner = ComparisonWinner.TIE if np.isclose(mean_a, mean_b) else (ComparisonWinner.A if mean_a > mean_b else ComparisonWinner.B)
        return ComparisonResult(
            metric=metric,
            input_a=ComparisonInput(mean=mean_a, data=result_a),
            input_b=ComparisonInput(mean=mean_b, data=result_b),
            winner=winner,
            delta=float(abs(mean_a - mean_b)),
        )

    def diff(self, predictions_a: TribePredictions, predictions_b: TribePredictions) -> PredictionDiff:
        if predictions_a.values.shape != predictions_b.values.shape:
            raise ValueError(
                f"Shape mismatch: {predictions_a.values.shape} vs {predictions_b.values.shape}. Inputs must have the same number of timesteps and vertices."
            )

        delta = predictions_a.values - predictions_b.values
        return PredictionDiff(
            delta=delta,
            mean_diff_per_timestep=tuple(float(value) for value in delta.mean(axis=1).astype(float).tolist()),
            mean_diff_per_vertex=tuple(float(value) for value in delta.mean(axis=0).astype(float).tolist()),
            abs_mean=float(np.abs(delta).mean()),
            max_diff_vertex=int(np.abs(delta).mean(axis=0).argmax()),
        )

    def normalize(self, predictions: TribePredictions, baseline_predictions: TribePredictions) -> NormalizedPredictions:
        if predictions.n_vertices != baseline_predictions.n_vertices:
            raise ValueError(
                f"Vertex mismatch: {predictions.n_vertices} vs {baseline_predictions.n_vertices}. Baseline must have the same number of vertices."
            )
        baseline_mean = baseline_predictions.values.mean(axis=0)
        return NormalizedPredictions(predictions.values - baseline_mean)

    def segment_by_engagement(self, predictions: TribePredictions, segments: TribeSegments, threshold: float = 50.0) -> EngagementSegmentation:
        curve = self.get_temporal_curve(predictions, segments)
        scores = np.asarray(curve.scores, dtype=float)
        timestamps = np.asarray(curve.timestamps, dtype=float)
        end_timestamps = (segments.frame["onset"] + segments.frame["duration"]).to_numpy(dtype=float)
        high_mask = scores >= threshold
        low_mask = ~high_mask
        return EngagementSegmentation(
            threshold=float(threshold),
            high_engagement=self._build_windows(high_mask, timestamps, end_timestamps),
            low_engagement=self._build_windows(low_mask, timestamps, end_timestamps),
            pct_high=float(high_mask.mean() * 100),
        )

    def export(self, result: ExportPayload | TemporalCurve | PeakMoments | RegionActivations | CognitiveLoadScore | LanguageProcessingScore | ComparisonResult | PredictionDiff | NormalizedPredictions | EngagementSegmentation, format: ExportFormat = ExportFormat.JSON, path: str | Path = "output") -> ExportArtifact:
        if format is ExportFormat.NIFTI:
            payload = result if isinstance(result, ExportPayload) else None
            if payload is None or (payload.array is None and payload.delta is None):
                raise ValueError("NIfTI export requires an ExportPayload with an 'array' or 'delta' value.")
        return self.file_exporter.export(result=result, format=format, path=path)

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

    def _get_named_region_curve(
        self,
        predictions: TribePredictions,
        segments: TribeSegments,
        regions: Sequence[DestrieuxRegion],
    ) -> np.ndarray:
        validate_timestep_alignment(predictions, segments)
        vertex_groups = [self.get_roi_indices(region.value) for region in regions]
        all_vertices = np.unique(np.concatenate([np.asarray(group, dtype=int) for group in vertex_groups]))
        validate_vertex_indices(predictions, all_vertices, ",".join(region.value for region in regions))
        return predictions.values[:, all_vertices].mean(axis=1)

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
    def _build_windows(mask: np.ndarray, timestamps: np.ndarray, end_timestamps: np.ndarray) -> tuple[EngagementWindow, ...]:
        windows: list[EngagementWindow] = []
        in_window = False
        start = 0.0
        for index, is_active in enumerate(mask):
            if is_active and not in_window:
                start = float(timestamps[index])
                in_window = True
                continue
            if not is_active and in_window:
                windows.append(EngagementWindow(start_s=start, end_s=float(end_timestamps[index - 1])))
                in_window = False
        if in_window:
            windows.append(EngagementWindow(start_s=start, end_s=float(end_timestamps[-1])))
        return tuple(windows)
