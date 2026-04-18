from __future__ import annotations

from pathlib import Path

import pandas as pd

from libs.config import ConfigLoader
from libs.dataclasses import PredictionResult, TribeConfig
from libs.protocols import SupportsTribeModel
from libs.utils import DataInput, LocalFileManager
from services.inference.inference_workflow_coordinator import InferenceWorkflowCoordinator


class TribeRunner:
    def __init__(self, config: TribeConfig | None = None, backend: SupportsTribeModel | None = None, config_loader: ConfigLoader | None = None, file_manager: LocalFileManager | None = None, workflow: InferenceWorkflowCoordinator | None = None) -> None:
        # Load the runtime config and wire the collaborators this runner depends on.
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

    def get_event_dataframe(self, input_path: str | Path | None = None) -> pd.DataFrame:
        # Build the standardized events dataframe for a single input file.
        resolved_input_path = self.workflow.resolve_input_path(input_path)
        data_input = DataInput.from_path(resolved_input_path)
        backend = self.workflow.get_backend()
        _, events = data_input.build_events_dataframe(model=backend, working_dir=self.cache_dir)
        return events

    def run(self, input_path: str | Path | None = None, *, verbose: bool | None = None, save_to: str | Path | None = None) -> PredictionResult:
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
        result = self.workflow.build_prediction_result(prepared_input=prepared_input, events=events, brain_stimulus=brain_stimulus, segments=segments)

        # Persist artifacts immediately when the caller supplies an output target.
        if resolved_save_to is not None:
            self.save_output(result=result, output_path=resolved_save_to)
        return result

    def get_brain_stimulus(self, result_or_input: PredictionResult | str | Path | None = None, *, verbose: bool | None = None):
        # Accept either a ready result or a raw input path and return the array payload.
        resolved_input = result_or_input if result_or_input is not None else self.workflow.resolve_input_path(None)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        result = self.workflow.resolve_prediction_result(resolved_input, self.run, resolved_verbose)
        return result.brain_stimulus

    def get_brain_stimulus_dataframe(self, result_or_input: PredictionResult | str | Path | None = None, *, verbose: bool | None = None) -> pd.DataFrame:
        # Accept either a ready result or a raw input path and return a tabular view.
        resolved_input = result_or_input if result_or_input is not None else self.workflow.resolve_input_path(None)
        resolved_verbose = self.workflow.resolve_verbose(verbose)
        result = self.workflow.resolve_prediction_result(resolved_input, self.run, resolved_verbose)
        return result.brain_stimulus_frame()

    def save_output(self, result: PredictionResult, output_path: str | Path | None = None, *, include_brain_stimulus_csv: bool | None = None) -> Path:
        # Resolve the destination directory for this result bundle.
        resolved_include_csv = self.workflow.resolve_include_brain_stimulus_csv(include_brain_stimulus_csv)
        destination = self.file_manager.create_output_directory(base_output_dir=self.output_dir, input_path=result.input_path, output_path=output_path)

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
