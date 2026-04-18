from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from libs.dataclasses import PredictionResult, PreparedInput, TribeConfig
from libs.protocols import SupportsTribeModel


class InferenceWorkflowCoordinator:
    def __init__(self, config: TribeConfig, backend: SupportsTribeModel | None = None) -> None:
        self.config = config
        self.backend = backend

    def get_backend(self) -> SupportsTribeModel:
        if self.backend is None:
            self.backend = self.load_backend()
        return self.backend

    def load_backend(self) -> SupportsTribeModel:
        try:
            from tribev2 import TribeModel
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("Failed to import tribev2. Install the project dependencies first.") from exc

        return TribeModel.from_pretrained(
            self.config.model_name,
            checkpoint_name=self.config.checkpoint_name,
            cache_folder=str(self.config.cache_dir),
            cluster=self.config.cluster,
            device=self.config.device,
            config_update=dict(self.config.config_update or {}) or None,
        )

    def build_prediction_result(self, prepared_input: PreparedInput, events, brain_stimulus, segments) -> PredictionResult:
        return PredictionResult(
            input_path=prepared_input.original_path,
            model_input_path=prepared_input.model_path,
            input_kind=prepared_input.kind,
            events=events,
            brain_stimulus=brain_stimulus,
            segments=segments,
        )

    def resolve_prediction_result(self, result_or_input: PredictionResult | str | Path, run_prediction: Callable[..., PredictionResult], verbose: bool) -> PredictionResult:
        if isinstance(result_or_input, PredictionResult):
            return result_or_input
        return run_prediction(result_or_input, verbose=verbose)

    def build_output_metadata(self, result: PredictionResult) -> dict[str, Any]:
        return {
            "input_path": str(result.input_path),
            "model_input_path": str(result.model_input_path),
            "input_kind": result.input_kind.value,
            "brain_stimulus_shape": list(result.brain_stimulus.shape),
            "event_count": int(len(result.events)),
            "segment_count": int(len(result.segments)),
        }

    def serialize_segments(self, segments: list[Any]) -> list[dict[str, Any]]:
        serialized_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            payload: dict[str, Any] = {"index": index}
            if isinstance(segment, Mapping):
                payload.update({str(key): value for key, value in segment.items()})
                serialized_segments.append(payload)
                continue

            for field in ("start", "stop", "offset", "duration"):
                value = getattr(segment, field, None)
                if value is not None:
                    payload[field] = value

            events = getattr(segment, "ns_events", None)
            if events is not None:
                payload["event_count"] = len(events)

            if len(payload) == 1:
                payload["repr"] = repr(segment)

            serialized_segments.append(payload)
        return serialized_segments
