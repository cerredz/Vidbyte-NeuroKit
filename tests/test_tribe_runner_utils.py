from __future__ import annotations

from pathlib import Path

import numpy as np

from libs.dataclasses import TRIBE_FSAVERAGE5_VERTEX_COUNT, TribePredictions
from libs.enums import ComparisonMetric, ExportFormat
from libs.utils import TribeRunnerUtils, build_export_payload, build_tribe_segments


def build_prediction_matrix() -> np.ndarray:
    predictions = np.zeros((2, TRIBE_FSAVERAGE5_VERTEX_COUNT), dtype=float)
    predictions[:, 0] = [1.0, 3.0]
    predictions[:, 1] = [2.0, 4.0]
    return predictions


def test_tribe_predictions_reject_non_fsaverage_shape() -> None:
    try:
        TribePredictions(np.ones((2, 2)))
    except ValueError as exc:
        assert "20484" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")


def test_utils_return_validated_dataclasses(tmp_path: Path) -> None:
    predictions = TribePredictions(build_prediction_matrix())
    baseline = TribePredictions(build_prediction_matrix() - 1.0)
    segments = build_tribe_segments([{"start": 0.0, "duration": 1.0}, {"start": 1.0, "duration": 1.0}])
    utils = TribeRunnerUtils(roi_index_resolver=lambda region: np.array([0]) if region == "G_front_sup" else np.array([1]))

    curve = utils.get_temporal_curve(predictions, segments)
    comparison = utils.compare(predictions, baseline, segments, segments, metric=ComparisonMetric.ENGAGEMENT)
    normalized = utils.normalize(predictions, baseline)
    exported = utils.export(build_export_payload(predictions, segments), format=ExportFormat.JSON, path=tmp_path / "exported")

    assert curve.scores == (0.0, 100.0)
    assert comparison.metric is ComparisonMetric.ENGAGEMENT
    assert normalized.values.shape == predictions.values.shape
    assert Path(exported.path).exists()
