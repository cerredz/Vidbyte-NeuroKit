from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from libs.config import ConfigLoader
from libs.dataclasses import PredictionResult, TribeConfig
from libs.enums import TranslationOutputKey
from libs.protocols import SupportsTribeModel
from libs.utils import DataInput, LocalFileManager, TribeRunnerUtils
from services.inference.inference_workflow_coordinator import InferenceWorkflowCoordinator


class TribeRunner:
    def __init__(
        self,
        config: TribeConfig | None = None,
        backend: SupportsTribeModel | None = None,
        config_loader: ConfigLoader | None = None,
        file_manager: LocalFileManager | None = None,
        workflow: InferenceWorkflowCoordinator | None = None,
        utils: TribeRunnerUtils | None = None,
    ) -> None:
        # Load the runtime config and wire the collaborators this runner depends on.
        self.config_loader = config_loader or ConfigLoader(config or TribeConfig())
        self.file_manager = file_manager or LocalFileManager()
        loaded_config = self.config_loader.load()
        self.cache_dir, self.output_dir = self.file_manager.prepare_runner_directories(
            loaded_config.cache_dir,
            loaded_config.output_dir,
        )
        self.config = TribeConfig(
            model_name=loaded_config.model_name,
            cache_dir=self.cache_dir,
            output_dir=self.output_dir,
            checkpoint_name=loaded_config.checkpoint_name,
            device=loaded_config.device,
            cluster=loaded_config.cluster,
            config_update=dict(loaded_config.config_update or {}),
            input_path=loaded_config.input_path,
            save_to=loaded_config.save_to,
            verbose=loaded_config.verbose,
            include_brain_stimulus_csv=loaded_config.include_brain_stimulus_csv,
        )
        self.workflow = workflow or InferenceWorkflowCoordinator(config=self.config, backend=backend)
        self.utils = utils or TribeRunnerUtils(atlas_data_dir=self.cache_dir)

    def get_event_dataframe(self, input_path: str | Path | None = None) -> pd.DataFrame:
        # Build the standardized events dataframe for a single input file.
        resolved_input_path = self.workflow.resolve_input_path(input_path)
        data_input = DataInput.from_path(resolved_input_path)
        backend = self.workflow.get_backend()
        _, events = data_input.build_events_dataframe(model=backend, working_dir=self.cache_dir)
        return events

    def run(
        self,
        input_path: str | Path | None = None,
        *,
        verbose: bool | None = None,
        save_to: str | Path | None = None,
    ) -> PredictionResult:
        # Create a normalized input and build the events the backend expects.
        resolved_input_path = self.workflow.resolve_input_path(input_path)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        resolved_save_to = self.workflow.resolve_save_to(save_to)
        data_input = DataInput.from_path(resolved_input_path)
        backend = self.workflow.get_backend()
        prepared_input, events = data_input.build_events_dataframe(model=backend, working_dir=self.cache_dir)

        # Execute the model against the prepared event dataframe.
        brain_stimulus, segments = backend.predict(events=events, verbose=resolved_verbose)

        # Build the structured result object returned to callers.
        result = self.workflow.build_prediction_result(
            prepared_input=prepared_input,
            events=events,
            brain_stimulus=brain_stimulus,
            segments=segments,
        )

        # Persist artifacts immediately when the caller supplies an output target.
        if resolved_save_to is not None:
            self.save_output(result=result, output_path=resolved_save_to)
        return result

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        verbose: bool | None = None,
        save_to: str | Path | None = None,
        max_workers: int | None = None,
    ) -> list[PredictionResult]:
        if isinstance(inputs, (str, Path)):
            raise TypeError("run_batch expects a sequence of input paths, not a single path.")

        input_list = list(inputs)
        if not input_list:
            return []

        resolved_verbose = self.workflow.resolve_verbose(verbose)
        resolved_save_to = self.workflow.resolve_save_to(save_to)
        worker_count = max_workers or min(len(input_list), 32)
        if worker_count <= 0:
            raise ValueError("max_workers must be greater than zero.")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda input_path: self.run(input_path, verbose=resolved_verbose, save_to=None),
                    input_list,
                )
            )

        if resolved_save_to is not None:
            base_output_dir = self.file_manager.ensure_directory(resolved_save_to)
            for result in results:
                self.save_output(result=result, output_path=base_output_dir / result.input_path.stem)
        return results

    def get_brain_stimulus(
        self,
        result_or_input: PredictionResult | str | Path | None = None,
        *,
        verbose: bool | None = None,
    ):
        # Accept either a ready result or a raw input path and return the array payload.
        resolved_input = result_or_input if result_or_input is not None else self.workflow.resolve_input_path(None)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        result = self.workflow.resolve_prediction_result(resolved_input, self.run, resolved_verbose)
        return result.brain_stimulus

    def get_brain_stimulus_dataframe(
        self,
        result_or_input: PredictionResult | str | Path | None = None,
        *,
        verbose: bool | None = None,
    ) -> pd.DataFrame:
        # Accept either a ready result or a raw input path and return a tabular view.
        resolved_input = result_or_input if result_or_input is not None else self.workflow.resolve_input_path(None)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        result = self.workflow.resolve_prediction_result(resolved_input, self.run, resolved_verbose)
        return result.brain_stimulus_frame()

    def save_output(
        self,
        result: PredictionResult,
        output_path: str | Path | None = None,
        *,
        include_brain_stimulus_csv: bool | None = None,
    ) -> Path:
        # Resolve the destination directory for this result bundle.
        resolved_include_csv = self.workflow.resolve_include_brain_stimulus_csv(include_brain_stimulus_csv)
        destination = self.file_manager.create_output_directory(
            base_output_dir=self.output_dir,
            input_path=result.input_path,
            output_path=output_path,
        )

        # Persist the event table and the primary brain-stimulus array.
        self.file_manager.write_events(destination, result.events)
        self.file_manager.write_brain_stimulus_array(destination, result.brain_stimulus)

        # Persist the optional CSV form when the caller requests it.
        if resolved_include_csv:
            self.file_manager.write_brain_stimulus_frame(destination, result.brain_stimulus_frame())

        # Persist metadata and serialized segment summaries for downstream use.
        self.file_manager.write_json(destination, "metadata.json", self.workflow.build_output_metadata(result))
        self.file_manager.write_json(destination, "segments.json", self.workflow.serialize_segments(result.segments))
        return destination

    def get_temporal_curve(self, result_or_preds: PredictionResult | np.ndarray, segments=None) -> dict[str, list[float]]:
        preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
        return self.utils.get_temporal_curve(preds, resolved_segments)

    def get_peak_moments(
        self,
        result_or_preds: PredictionResult | np.ndarray,
        segments=None,
        *,
        top_n: int = 3,
    ) -> list[dict[str, float | int]]:
        preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
        return self.utils.get_peak_moments(preds, resolved_segments, top_n=top_n)

    def get_region_activations(
        self,
        result_or_preds: PredictionResult | np.ndarray,
        regions: Sequence[str],
        segments=None,
    ) -> dict[str, dict[str, Any]]:
        preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
        return self.utils.get_region_activations(preds, resolved_segments, regions)

    def get_cognitive_load_score(self, result_or_preds: PredictionResult | np.ndarray, segments=None) -> dict[str, Any]:
        preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
        return self.utils.get_cognitive_load_score(preds, resolved_segments)

    def get_language_processing_score(self, result_or_preds: PredictionResult | np.ndarray, segments=None) -> dict[str, Any]:
        preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
        return self.utils.get_language_processing_score(preds, resolved_segments)

    def compare(
        self,
        result_or_preds_a: PredictionResult | np.ndarray,
        result_or_preds_b: PredictionResult | np.ndarray,
        *,
        segments_a=None,
        segments_b=None,
        metric: str = "engagement",
    ) -> dict[str, Any]:
        preds_a, resolved_segments_a = self._resolve_prediction_payload(result_or_preds_a, segments_a)
        preds_b, resolved_segments_b = self._resolve_prediction_payload(result_or_preds_b, segments_b)
        return self.utils.compare(preds_a, preds_b, resolved_segments_a, resolved_segments_b, metric=metric)

    def diff(self, result_or_preds_a: PredictionResult | np.ndarray, result_or_preds_b: PredictionResult | np.ndarray) -> dict[str, Any]:
        preds_a, _ = self._resolve_prediction_payload(result_or_preds_a, None, allow_missing_segments=True)
        preds_b, _ = self._resolve_prediction_payload(result_or_preds_b, None, allow_missing_segments=True)
        return self.utils.diff(preds_a, preds_b)

    def normalize(
        self,
        result_or_preds: PredictionResult | np.ndarray,
        baseline_result_or_preds: PredictionResult | np.ndarray,
    ) -> np.ndarray:
        preds, _ = self._resolve_prediction_payload(result_or_preds, None, allow_missing_segments=True)
        baseline_preds, _ = self._resolve_prediction_payload(
            baseline_result_or_preds,
            None,
            allow_missing_segments=True,
        )
        return self.utils.normalize(preds, baseline_preds)

    def segment_by_engagement(
        self,
        result_or_preds: PredictionResult | np.ndarray,
        segments=None,
        *,
        threshold: float = 50.0,
    ) -> dict[str, Any]:
        preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
        return self.utils.segment_by_engagement(preds, resolved_segments, threshold=threshold)

    def export(self, result: Mapping[str, Any], *, format: str = "json", path: str | Path = "output") -> str:
        return self.utils.export(result=result, format=format, path=path)

    def translate(
        self,
        result_or_preds: PredictionResult | np.ndarray,
        outputs: TranslationOutputKey | str | Sequence[TranslationOutputKey | str],
        *,
        segments=None,
        options: Mapping[TranslationOutputKey | str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved_keys = self._resolve_translation_keys(outputs)
        normalized_options = self._normalize_translation_options(options or {})
        translated: dict[str, Any] = {}

        for output_key in resolved_keys:
            params = dict(normalized_options.get(output_key, {}))
            if output_key is TranslationOutputKey.TEMPORAL:
                translated[output_key.value] = self.get_temporal_curve(result_or_preds, segments=segments)
                continue
            if output_key is TranslationOutputKey.PEAK:
                translated[output_key.value] = self.get_peak_moments(
                    result_or_preds,
                    segments=segments,
                    top_n=int(params.pop("top_n", 3)),
                )
                continue
            if output_key is TranslationOutputKey.REGIONS:
                if "regions" not in params:
                    raise ValueError("translate(..., 'regions') requires options={'regions': {'regions': [...]}}.")
                translated[output_key.value] = self.get_region_activations(
                    result_or_preds,
                    regions=params.pop("regions"),
                    segments=segments,
                )
                continue
            if output_key is TranslationOutputKey.COGNITIVE:
                translated[output_key.value] = self.get_cognitive_load_score(result_or_preds, segments=segments)
                continue
            if output_key is TranslationOutputKey.LANGUAGE:
                translated[output_key.value] = self.get_language_processing_score(result_or_preds, segments=segments)
                continue
            if output_key is TranslationOutputKey.COMPARE:
                other = self._require_translation_operand(output_key, params, "other")
                translated[output_key.value] = self.compare(
                    result_or_preds,
                    other,
                    segments_a=segments,
                    segments_b=params.pop("segments_b", None),
                    metric=str(params.pop("metric", "engagement")),
                )
                continue
            if output_key is TranslationOutputKey.DIFF:
                other = self._require_translation_operand(output_key, params, "other")
                translated[output_key.value] = self.diff(result_or_preds, other)
                continue
            if output_key is TranslationOutputKey.NORMALIZE:
                baseline = self._require_translation_operand(output_key, params, "baseline")
                translated[output_key.value] = self.normalize(result_or_preds, baseline)
                continue
            if output_key is TranslationOutputKey.SEGMENT:
                translated[output_key.value] = self.segment_by_engagement(
                    result_or_preds,
                    segments=segments,
                    threshold=float(params.pop("threshold", 50.0)),
                )
                continue
            if output_key is TranslationOutputKey.EXPORT:
                export_payload = params.pop("result", None)
                if export_payload is None:
                    preds, resolved_segments = self._resolve_prediction_payload(result_or_preds, segments)
                    if isinstance(resolved_segments, pd.DataFrame):
                        serialized_segments = resolved_segments.to_dict(orient="records")
                    else:
                        serialized_segments = self.workflow.serialize_segments(list(resolved_segments))
                    export_payload = {
                        "array": preds,
                        "segments": serialized_segments,
                    }
                translated[output_key.value] = self.export(
                    export_payload,
                    format=str(params.pop("format", "json")),
                    path=params.pop("path", "output"),
                )
                continue
            raise ValueError(f"Unsupported translation output '{output_key.value}'.")

        return translated

    @staticmethod
    def _resolve_prediction_payload(
        result_or_preds: PredictionResult | np.ndarray,
        segments,
        *,
        allow_missing_segments: bool = False,
    ):
        if isinstance(result_or_preds, PredictionResult):
            if segments is not None:
                raise ValueError("Do not pass segments when supplying a PredictionResult.")
            return result_or_preds.brain_stimulus, result_or_preds.segments

        if allow_missing_segments:
            return np.asarray(result_or_preds), segments
        if segments is None:
            raise ValueError("Segments are required when supplying a raw prediction array.")
        return np.asarray(result_or_preds), segments

    @staticmethod
    def _normalize_translation_options(
        options: Mapping[TranslationOutputKey | str, Mapping[str, Any]],
    ) -> dict[TranslationOutputKey, Mapping[str, Any]]:
        normalized: dict[TranslationOutputKey, Mapping[str, Any]] = {}
        for raw_key, value in options.items():
            normalized[TribeRunner._coerce_translation_key(raw_key)] = value
        return normalized

    @staticmethod
    def _resolve_translation_keys(
        outputs: TranslationOutputKey | str | Sequence[TranslationOutputKey | str],
    ) -> list[TranslationOutputKey]:
        if isinstance(outputs, (TranslationOutputKey, str)):
            raw_outputs: Sequence[TranslationOutputKey | str] = [outputs]
        else:
            raw_outputs = list(outputs)
        return [TribeRunner._coerce_translation_key(output) for output in raw_outputs]

    @staticmethod
    def _coerce_translation_key(raw_key: TranslationOutputKey | str) -> TranslationOutputKey:
        if isinstance(raw_key, TranslationOutputKey):
            return raw_key
        try:
            return TranslationOutputKey(raw_key.lower())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported translation key '{raw_key}'. Expected one of {[member.value for member in TranslationOutputKey]}."
            ) from exc

    @staticmethod
    def _require_translation_operand(
        output_key: TranslationOutputKey,
        params: dict[str, Any],
        operand_name: str,
    ) -> Any:
        if operand_name not in params:
            raise ValueError(
                f"translate(..., '{output_key.value}') requires options={{'{output_key.value}': "
                f"{{'{operand_name}': ...}}}}."
            )
        return params.pop(operand_name)
