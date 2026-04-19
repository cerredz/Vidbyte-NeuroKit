from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from libs.config import ConfigLoader
from libs.dataclasses import FormatRequest, PredictionResult, TranslationReport, TribeConfig, TribePredictions, TribeSegments
from libs.enums import ComparisonMetric, DataFormat, ExportFormat, TranslationOutputKey
from libs.protocols import SupportsTribeModel
from libs.utils import DataInput, Formatter, LocalFileManager, TribeRunnerUtils
from libs.utils.tribe_utils import (
    build_export_payload,
    normalize_translation_options,
    require_segments,
    require_translation_operand,
    resolve_prediction_artifacts,
    resolve_translation_keys,
)
from services.inference.inference_workflow_coordinator import InferenceWorkflowCoordinator


class TribeRunner:
    def __init__(self, config: TribeConfig | None = None, backend: SupportsTribeModel | None = None, config_loader: ConfigLoader | None = None, file_manager: LocalFileManager | None = None, workflow: InferenceWorkflowCoordinator | None = None, utils: TribeRunnerUtils | None = None, formatter: Formatter | None = None) -> None:
        self.config_loader = config_loader or ConfigLoader(config or TribeConfig())
        self.file_manager = file_manager or LocalFileManager()
        loaded_config = self.config_loader.load()
        self.cache_dir, self.output_dir = self.file_manager.prepare_runner_directories(loaded_config.cache_dir, loaded_config.output_dir)
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
        self.formatter = formatter or Formatter()

    def get_event_dataframe(self, input_path: str | Path | None = None) -> pd.DataFrame:
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
        format: FormatRequest | DataFormat | str | Mapping[str, Any] | None = None,
    ) -> PredictionResult:
        resolved_input_path = self.workflow.resolve_input_path(input_path)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        resolved_save_to = self.workflow.resolve_save_to(save_to)
        resolved_format = self.formatter.validate_request(format=format)
        data_input = DataInput.from_path(resolved_input_path)
        backend = self.workflow.get_backend()
        prepared_input, events = data_input.build_events_dataframe(model=backend, working_dir=self.cache_dir)
        brain_stimulus, segments = backend.predict(events=events, verbose=resolved_verbose)
        result = self.workflow.build_prediction_result(prepared_input=prepared_input, events=events, brain_stimulus=brain_stimulus, segments=segments)
        if resolved_save_to is not None:
            self.save_output(result=result, output_path=resolved_save_to, format=resolved_format)
        return result

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        verbose: bool | None = None,
        save_to: str | Path | None = None,
        format: FormatRequest | DataFormat | str | Mapping[str, Any] | None = None,
        max_workers: int | None = None,
        parallel: bool = False,
    ) -> list[PredictionResult]:
        if isinstance(inputs, (str, Path)):
            raise TypeError("run_batch expects a sequence of input paths, not a single path.")

        input_list = list(inputs)
        if not input_list:
            return []

        resolved_verbose = self.workflow.resolve_verbose(verbose)
        resolved_save_to = self.workflow.resolve_save_to(save_to)
        resolved_format = self.formatter.validate_request(format=format)
        use_parallel = parallel or max_workers is not None

        if use_parallel:
            worker_count = max_workers or min(len(input_list), 32)
            if worker_count <= 0:
                raise ValueError("max_workers must be greater than zero.")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        lambda input_path: self.run(
                            input_path,
                            verbose=resolved_verbose,
                            save_to=None,
                            format=resolved_format,
                        ),
                        input_list,
                    )
                )
        else:
            results = [
                self.run(
                    input_path,
                    verbose=resolved_verbose,
                    save_to=None,
                    format=resolved_format,
                )
                for input_path in input_list
            ]

        if resolved_save_to is not None:
            base_output_dir = self.file_manager.ensure_directory(resolved_save_to)
            for result in results:
                self.save_output(
                    result=result,
                    output_path=base_output_dir / result.input_path.stem,
                    format=resolved_format,
                )
        return results

    def get_brain_stimulus(self, result_or_input: PredictionResult | str | Path | None = None, *, verbose: bool | None = None):
        resolved_input = result_or_input if result_or_input is not None else self.workflow.resolve_input_path(None)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        result = self.workflow.resolve_prediction_result(resolved_input, self.run, resolved_verbose)
        return result.brain_stimulus

    def get_brain_stimulus_dataframe(self, result_or_input: PredictionResult | str | Path | None = None, *, verbose: bool | None = None) -> pd.DataFrame:
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
        format: FormatRequest | DataFormat | str | Mapping[str, Any] | None = None,
    ) -> Path:
        resolved_include_csv = self.workflow.resolve_include_brain_stimulus_csv(include_brain_stimulus_csv)
        resolved_format = self.formatter.validate_request(format=format)
        destination = self.file_manager.create_output_directory(base_output_dir=self.output_dir, input_path=result.input_path, output_path=output_path)
        self.file_manager.write_events(destination, result.events)
        self.file_manager.write_brain_stimulus_array(destination, result.brain_stimulus)
        if resolved_include_csv:
            self.file_manager.write_brain_stimulus_frame(destination, result.brain_stimulus_frame())
        self.file_manager.write_json(destination, "metadata.json", self.workflow.build_output_metadata(result))
        self.file_manager.write_json(destination, "segments.json", self.workflow.serialize_segments(result.segments))
        if resolved_format is not None:
            self._write_formatted_output(destination=destination, result=result, format_request=resolved_format)
        return destination

    def _write_formatted_output(self, destination: Path, result: PredictionResult, format_request: FormatRequest) -> Path:
        target_format = format_request.target or format_request.source
        if target_format is None:
            raise ValueError("A target format is required when writing formatted output.")

        payload = self._build_formatter_payload(result, target_format)
        formatted_dir = self.file_manager.ensure_directory(destination / "formatted")
        if target_format is DataFormat.BIDS:
            output_path = formatted_dir / "bids"
        elif target_format is DataFormat.BIDS_EVENTS:
            output_path = formatted_dir / "sub-01_task-tribe_events"
        else:
            output_path = formatted_dir / "brain_stimulus_formatted"
        formatted_output = self.formatter.to(payload, format_request, output_path=output_path)
        if not isinstance(formatted_output, Path):
            raise TypeError("Formatter exports in TribeRunner must resolve to a filesystem path.")
        return formatted_output

    @staticmethod
    def _build_formatter_payload(result: PredictionResult, target_format: DataFormat):
        if target_format in {DataFormat.BIDS, DataFormat.BIDS_EVENTS}:
            return result.events
        if target_format in {DataFormat.CSV, DataFormat.TSV, DataFormat.AFNI_1D, DataFormat.HDF5}:
            return result.brain_stimulus_frame()
        raise NotImplementedError(
            f"TribeRunner export to '{target_format.value}' is not implemented for current prediction artifacts."
        )

    def translate(self, result_or_predictions: PredictionResult | TribePredictions, outputs: TranslationOutputKey | str | Sequence[TranslationOutputKey | str], *, segments: TribeSegments | Sequence[Any] | None = None, options: Mapping[TranslationOutputKey | str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        primary_predictions, primary_segments = resolve_prediction_artifacts(result_or_predictions, segments, require_segments=False)
        resolved_keys = resolve_translation_keys(outputs)
        normalized_options = normalize_translation_options(options or {})
        translated: dict[str, Any] = {}

        for output_key in resolved_keys:
            params = dict(normalized_options.get(output_key, {}))
            match output_key:
                case TranslationOutputKey.TEMPORAL:
                    translated[output_key.value] = self.utils.get_temporal_curve(primary_predictions, require_segments(primary_segments))
                case TranslationOutputKey.PEAK:
                    translated[output_key.value] = self.utils.get_peak_moments(
                        primary_predictions,
                        require_segments(primary_segments),
                        top_n=int(params.pop("top_n", 3)),
                    )
                case TranslationOutputKey.REGIONS:
                    if "regions" not in params:
                        raise ValueError("translate(..., 'regions') requires options={'regions': {'regions': [...]}}.")
                    translated[output_key.value] = self.utils.get_region_activations(
                        primary_predictions,
                        require_segments(primary_segments),
                        regions=params.pop("regions"),
                    )
                case TranslationOutputKey.COGNITIVE:
                    translated[output_key.value] = self.utils.get_cognitive_load_score(primary_predictions, require_segments(primary_segments))
                case TranslationOutputKey.LANGUAGE:
                    translated[output_key.value] = self.utils.get_language_processing_score(primary_predictions, require_segments(primary_segments))
                case TranslationOutputKey.COMPARE:
                    other = require_translation_operand(output_key, params, "other")
                    other_predictions, other_segments = resolve_prediction_artifacts(other, params.pop("segments_b", None), require_segments=True)
                    translated[output_key.value] = self.utils.compare(
                        primary_predictions,
                        other_predictions,
                        require_segments(primary_segments),
                        require_segments(other_segments),
                        metric=ComparisonMetric(str(params.pop("metric", ComparisonMetric.ENGAGEMENT.value)).lower()),
                    )
                case TranslationOutputKey.DIFF:
                    other = require_translation_operand(output_key, params, "other")
                    other_predictions, _ = resolve_prediction_artifacts(other, None, require_segments=False)
                    translated[output_key.value] = self.utils.diff(primary_predictions, other_predictions)
                case TranslationOutputKey.NORMALIZE:
                    baseline = require_translation_operand(output_key, params, "baseline")
                    baseline_predictions, _ = resolve_prediction_artifacts(baseline, None, require_segments=False)
                    translated[output_key.value] = self.utils.normalize(primary_predictions, baseline_predictions)
                case TranslationOutputKey.SEGMENT:
                    translated[output_key.value] = self.utils.segment_by_engagement(
                        primary_predictions,
                        require_segments(primary_segments),
                        threshold=float(params.pop("threshold", 50.0)),
                    )
                case TranslationOutputKey.EXPORT:
                    payload = params.pop("result", None)
                    export_payload = payload if payload is not None else build_export_payload(primary_predictions, primary_segments)
                    translated[output_key.value] = self.utils.export(
                        export_payload,
                        format=ExportFormat(str(params.pop("format", ExportFormat.JSON.value)).lower()),
                        path=params.pop("path", "output"),
                    )
                case _:
                    raise ValueError(f"Unsupported translation output '{output_key.value}'.")

        return translated

    def report(
        self,
        result_or_predictions: PredictionResult | TribePredictions,
        *,
        segments: TribeSegments | Sequence[Any] | None = None,
        options: Mapping[TranslationOutputKey | str, Mapping[str, Any]] | None = None,
    ) -> TranslationReport:
        translated = self.translate(
            result_or_predictions,
            list(TranslationOutputKey),
            segments=segments,
            options=options,
        )
        return TranslationReport(
            temporal=translated[TranslationOutputKey.TEMPORAL.value],
            peak=translated[TranslationOutputKey.PEAK.value],
            regions=translated[TranslationOutputKey.REGIONS.value],
            cognitive=translated[TranslationOutputKey.COGNITIVE.value],
            language=translated[TranslationOutputKey.LANGUAGE.value],
            compare=translated[TranslationOutputKey.COMPARE.value],
            diff=translated[TranslationOutputKey.DIFF.value],
            normalize=translated[TranslationOutputKey.NORMALIZE.value],
            segment=translated[TranslationOutputKey.SEGMENT.value],
            export=translated[TranslationOutputKey.EXPORT.value],
        )
