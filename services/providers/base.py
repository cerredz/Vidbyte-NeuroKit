from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from libs.dataclasses.provider_models import (
    DownloadedProviderAsset,
    ProviderAnalysisResult,
    ProviderBatchAnalysisResult,
    ProviderComparisonResult,
    ProviderMetricResult,
)
from libs.enums import ComparisonMetric, TranslationOutputKey
from services.inference import TribeRunner


METRIC_TRANSLATIONS: dict[str, TranslationOutputKey] = {
    "engagement": TranslationOutputKey.TEMPORAL,
    "cognitive-load": TranslationOutputKey.COGNITIVE,
    "language": TranslationOutputKey.LANGUAGE,
    "peak": TranslationOutputKey.PEAK,
}

COMPARISON_METRICS: dict[str, ComparisonMetric] = {
    "engagement": ComparisonMetric.ENGAGEMENT,
    "cognitive-load": ComparisonMetric.COGNITIVE_LOAD,
    "language": ComparisonMetric.LANGUAGE,
}


class ProviderRunnerBase:
    provider_name: str

    def __init__(self, provider: Any, tribe_runner: TribeRunner | None = None) -> None:
        self.provider = provider
        self.tribe_runner = tribe_runner or TribeRunner()
        self._align_provider_cache()

    def analyze_asset(
        self,
        asset: DownloadedProviderAsset,
        *,
        metrics: Sequence[str] | None = None,
        save_to: str | Path | None = None,
    ) -> ProviderAnalysisResult:
        prediction = self.tribe_runner.run(asset.local_path, save_to=save_to)
        metric_results = tuple(self._compute_metric(prediction, metric) for metric in self._normalize_metrics(metrics))
        return ProviderAnalysisResult(provider=self.provider_name, asset=asset, prediction=prediction, metrics=metric_results)

    def analyze_assets(
        self,
        assets: Iterable[DownloadedProviderAsset],
        *,
        metrics: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> ProviderBatchAnalysisResult:
        items = tuple(self.analyze_asset(asset, metrics=metrics) for asset in assets)
        if sort_by is None:
            return ProviderBatchAnalysisResult(provider=self.provider_name, items=items, sorted_by=None)
        normalized_sort = self._normalize_metric_name(sort_by)
        sorted_items = tuple(
            sorted(
                items,
                key=lambda item: self._extract_metric_score(item, normalized_sort) or float("-inf"),
                reverse=True,
            )
        )
        return ProviderBatchAnalysisResult(provider=self.provider_name, items=sorted_items, sorted_by=normalized_sort)

    def compare_assets(
        self,
        left_asset: DownloadedProviderAsset,
        right_asset: DownloadedProviderAsset,
        *,
        metric: str,
    ) -> ProviderComparisonResult:
        normalized_metric = self._normalize_metric_name(metric)
        if normalized_metric not in COMPARISON_METRICS:
            raise ValueError(f"Metric '{metric}' does not support compare().")
        left = self.analyze_asset(left_asset, metrics=[normalized_metric])
        right = self.analyze_asset(right_asset, metrics=[normalized_metric])
        comparison = self.tribe_runner.translate(
            left.prediction,
            [TranslationOutputKey.COMPARE],
            options={
                TranslationOutputKey.COMPARE.value: {
                    "other": right.prediction,
                    "metric": COMPARISON_METRICS[normalized_metric].value,
                }
            },
        )["compare"]
        return ProviderComparisonResult(
            provider=self.provider_name,
            metric=normalized_metric,
            left=left,
            right=right,
            comparison=comparison,
        )

    def _compute_metric(self, prediction: Any, metric: str) -> ProviderMetricResult:
        normalized_metric = self._normalize_metric_name(metric)
        output_key = METRIC_TRANSLATIONS.get(normalized_metric)
        if output_key is None:
            raise ValueError(f"Unsupported metric '{metric}'. Expected one of {sorted(METRIC_TRANSLATIONS)}.")
        translated = self.tribe_runner.translate(prediction, [output_key])[output_key.value]
        return ProviderMetricResult(metric=normalized_metric, score=self._score_metric(translated, normalized_metric), result=translated)

    @staticmethod
    def _normalize_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
        if not metrics:
            return ("engagement",)
        return tuple(ProviderRunnerBase._normalize_metric_name(metric) for metric in metrics)

    @staticmethod
    def _normalize_metric_name(metric: str) -> str:
        normalized = metric.strip().lower().replace("_", "-")
        if normalized == "cognitive_load":
            return "cognitive-load"
        return normalized

    @staticmethod
    def _score_metric(result: Any, metric: str) -> float | None:
        if metric == "engagement":
            scores = getattr(result, "scores", ())
            return float(mean(scores)) if scores else None
        if metric in {"cognitive-load", "language"}:
            mean_score = getattr(result, "mean_score", None)
            return None if mean_score is None else float(mean_score)
        if metric == "peak":
            items = getattr(result, "items", ())
            if not items:
                return None
            return float(getattr(items[0], "score", 0.0))
        return None

    @staticmethod
    def _extract_metric_score(result: ProviderAnalysisResult, metric: str) -> float | None:
        for item in result.metrics:
            if item.metric == metric:
                return item.score
        return None

    def _align_provider_cache(self) -> None:
        provider_dir = Path(self.tribe_runner.cache_dir) / "providers" / self.provider_name
        if hasattr(self.provider, "download_dir"):
            self.provider.download_dir = provider_dir.resolve()
