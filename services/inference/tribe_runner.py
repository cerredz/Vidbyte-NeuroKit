from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from libs.utils.data_input import DataInput
from tribe_setup.models import PredictionResult


class SupportsTribeModel(Protocol):
    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> pd.DataFrame:
        ...

    def predict(
        self,
        events: pd.DataFrame,
        verbose: bool = True,
    ) -> tuple[np.ndarray, list[Any]]:
        ...


class TribeRunner:
    def __init__(
        self,
        model_name: str = "facebook/tribev2",
        cache_dir: str | Path = "cache",
        output_dir: str | Path = "outputs",
        checkpoint_name: str = "best.ckpt",
        device: str = "auto",
        cluster: str | None = None,
        config_update: Mapping[str, Any] | None = None,
        backend: SupportsTribeModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.cluster = cluster
        self.config_update = dict(config_update or {})
        self._backend = backend

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_event_dataframe(self, input_path: str | Path) -> pd.DataFrame:
        data_input = DataInput.from_path(input_path)
        backend = self._get_backend()
        _, events = data_input.build_events_dataframe(
            model=backend,
            working_dir=self.cache_dir,
        )
        return events

    def run(
        self,
        input_path: str | Path,
        *,
        verbose: bool = True,
        save_to: str | Path | None = None,
    ) -> PredictionResult:
        data_input = DataInput.from_path(input_path)
        backend = self._get_backend()
        prepared_input, events = data_input.build_events_dataframe(
            model=backend,
            working_dir=self.cache_dir,
        )
        brain_stimulus, segments = backend.predict(events=events, verbose=verbose)
        result = PredictionResult(
            input_path=prepared_input.original_path,
            model_input_path=prepared_input.model_path,
            input_kind=prepared_input.kind,
            events=events,
            brain_stimulus=brain_stimulus,
            segments=segments,
        )
        if save_to is not None:
            self.save_output(result=result, output_path=save_to)
        return result

    def get_brain_stimulus(
        self,
        result_or_input: PredictionResult | str | Path,
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        result = self._coerce_result(result_or_input=result_or_input, verbose=verbose)
        return result.brain_stimulus

    def get_brain_stimulus_dataframe(
        self,
        result_or_input: PredictionResult | str | Path,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        result = self._coerce_result(result_or_input=result_or_input, verbose=verbose)
        return result.brain_stimulus_frame()

    def save_output(
        self,
        result: PredictionResult,
        output_path: str | Path | None = None,
        *,
        include_brain_stimulus_csv: bool = False,
    ) -> Path:
        destination = self._resolve_output_directory(result=result, output_path=output_path)
        destination.mkdir(parents=True, exist_ok=True)

        result.events.to_csv(destination / "events.csv", index=False)
        np.save(destination / "brain_stimulus.npy", result.brain_stimulus)

        if include_brain_stimulus_csv:
            result.brain_stimulus_frame().to_csv(
                destination / "brain_stimulus.csv",
                index=False,
            )

        (destination / "metadata.json").write_text(
            json.dumps(self._build_metadata(result=result), indent=2),
            encoding="utf-8",
        )
        (destination / "segments.json").write_text(
            json.dumps(self._serialize_segments(result.segments), indent=2),
            encoding="utf-8",
        )
        return destination

    def _get_backend(self) -> SupportsTribeModel:
        if self._backend is None:
            self._backend = self._load_backend()
        return self._backend

    def _load_backend(self) -> SupportsTribeModel:
        try:
            from tribev2 import TribeModel
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "Failed to import tribev2. Install the project dependencies first."
            ) from exc

        return TribeModel.from_pretrained(
            self.model_name,
            checkpoint_name=self.checkpoint_name,
            cache_folder=str(self.cache_dir),
            cluster=self.cluster,
            device=self.device,
            config_update=self.config_update or None,
        )

    def _coerce_result(
        self,
        result_or_input: PredictionResult | str | Path,
        verbose: bool,
    ) -> PredictionResult:
        if isinstance(result_or_input, PredictionResult):
            return result_or_input
        return self.run(result_or_input, verbose=verbose)

    def _resolve_output_directory(
        self,
        result: PredictionResult,
        output_path: str | Path | None,
    ) -> Path:
        if output_path is not None:
            path = Path(output_path).expanduser().resolve()
            if path.exists() and path.is_file():
                raise ValueError(f"Output path must be a directory, got file: {path}")
            return path

        base_path = self.output_dir / result.input_path.stem
        if not base_path.exists():
            return base_path

        suffix = 1
        while True:
            candidate = self.output_dir / f"{result.input_path.stem}-{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def _build_metadata(self, result: PredictionResult) -> dict[str, Any]:
        return {
            "input_path": str(result.input_path),
            "model_input_path": str(result.model_input_path),
            "input_kind": result.input_kind.value,
            "brain_stimulus_shape": list(result.brain_stimulus.shape),
            "event_count": int(len(result.events)),
            "segment_count": int(len(result.segments)),
        }

    def _serialize_segments(self, segments: list[Any]) -> list[dict[str, Any]]:
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
