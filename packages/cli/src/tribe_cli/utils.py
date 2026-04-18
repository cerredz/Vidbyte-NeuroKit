from __future__ import annotations

import argparse
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from libs.dataclasses import PredictionResult, TribeConfig
from services.inference import TribeRunner


class CliCommand(StrEnum):
    INSPECT_BRAIN_RESPONSE = "inspect-brain-response"
    INSPECT_BRAIN_RESPONSE_TABLE = "inspect-brain-response-table"
    INSPECT_EVENTS = "inspect-events"
    PREDICT_RESPONSE = "predict-response"
    SAVE_BUNDLE = "save-bundle"


class CliCommandGroup(StrEnum):
    INSPECT = "inspect"
    PREDICT = "predict"
    SAVE = "save"


class CliCommandAction(StrEnum):
    BRAIN_RESPONSE = "brain-response"
    BRAIN_RESPONSE_TABLE = "brain-response-table"
    BUNDLE = "bundle"
    EVENTS = "events"
    RESPONSE = "response"


class CliPayloadKey(StrEnum):
    BRAIN_STIMULUS = "brain_stimulus"
    BRAIN_STIMULUS_SHAPE = "brain_stimulus_shape"
    COMMAND = "command"
    CONFIG = "config"
    COUNT = "count"
    ERROR = "error"
    EVENTS = "events"
    INCLUDE_BRAIN_STIMULUS_CSV = "include_brain_stimulus_csv"
    INPUT_KIND = "input_kind"
    INPUT_PATH = "input_path"
    MODEL_INPUT_PATH = "model_input_path"
    OK = "ok"
    RESULT = "result"
    SAVE_TO = "save_to"
    SAVED_TO = "saved_to"
    SEGMENTS = "segments"
    SHAPE = "shape"
    VERBOSE = "verbose"


class CliParserValue(StrEnum):
    COMMAND_ACTION_ARG = "action"
    COMMAND_GROUP_ARG = "task"
    INLINE_JSON_ARG = "--json"
    JSON_PAYLOAD_DEST = "json_payload"
    PROGRAM_NAME = "tribe-cli"


COMMANDS = tuple(sorted(command.value for command in CliCommand))
COMMAND_LOOKUP = {
    (CliCommandGroup.PREDICT.value, CliCommandAction.RESPONSE.value): CliCommand.PREDICT_RESPONSE.value,
    (CliCommandGroup.INSPECT.value, CliCommandAction.EVENTS.value): CliCommand.INSPECT_EVENTS.value,
    (CliCommandGroup.INSPECT.value, CliCommandAction.BRAIN_RESPONSE.value): CliCommand.INSPECT_BRAIN_RESPONSE.value,
    (CliCommandGroup.INSPECT.value, CliCommandAction.BRAIN_RESPONSE_TABLE.value): CliCommand.INSPECT_BRAIN_RESPONSE_TABLE.value,
    (CliCommandGroup.SAVE.value, CliCommandAction.BUNDLE.value): CliCommand.SAVE_BUNDLE.value,
}


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = load_payload(args.json_payload)
    command = resolve_command(args.task, args.action)

    try:
        response = execute_request(command, payload)
    except Exception as exc:
        print(json.dumps({CliPayloadKey.OK.value: False, CliPayloadKey.COMMAND.value: command, CliPayloadKey.ERROR.value: str(exc)}))
        return 1

    print(json.dumps({CliPayloadKey.OK.value: True, CliPayloadKey.COMMAND.value: command, CliPayloadKey.RESULT.value: response}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=CliParserValue.PROGRAM_NAME.value)
    parser.add_argument(CliParserValue.COMMAND_GROUP_ARG.value, choices=sorted(group.value for group in CliCommandGroup))
    parser.add_argument(CliParserValue.COMMAND_ACTION_ARG.value, choices=sorted(action.value for action in CliCommandAction))
    parser.add_argument(CliParserValue.INLINE_JSON_ARG.value, dest=CliParserValue.JSON_PAYLOAD_DEST.value, help="Inline JSON payload. If omitted, stdin is used when piped.")
    return parser


def load_payload(raw_payload: str | None) -> dict[str, Any]:
    payload_text = raw_payload
    if payload_text is None and not sys.stdin.isatty():
        payload_text = sys.stdin.read()
    if payload_text is None or not payload_text.strip():
        return {}

    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("CLI payload must decode to a JSON object.")
    return payload


def resolve_command(command_group: str, command_action: str) -> str:
    try:
        return COMMAND_LOOKUP[(command_group, command_action)]
    except KeyError as exc:
        raise ValueError(f"Unsupported CLI command path: {command_group} {command_action}") from exc


def execute_request(command: str, payload: dict[str, Any], runner_factory: Callable[..., Any] = TribeRunner) -> dict[str, Any]:
    if command not in COMMANDS:
        raise ValueError(f"Unsupported command: {command}")

    runner = build_runner(payload, runner_factory)
    if command == CliCommand.PREDICT_RESPONSE.value:
        return serialize_prediction_result(
            runner.run(
                payload.get(CliPayloadKey.INPUT_PATH.value),
                verbose=payload.get(CliPayloadKey.VERBOSE.value),
                save_to=payload.get(CliPayloadKey.SAVE_TO.value),
            ),
            runner,
        )
    if command == CliCommand.INSPECT_EVENTS.value:
        events = runner.get_event_dataframe(payload.get(CliPayloadKey.INPUT_PATH.value))
        return {CliPayloadKey.EVENTS.value: events.to_dict(orient="records"), CliPayloadKey.COUNT.value: len(events)}
    if command == CliCommand.INSPECT_BRAIN_RESPONSE.value:
        brain_stimulus = runner.get_brain_stimulus(payload.get(CliPayloadKey.INPUT_PATH.value), verbose=payload.get(CliPayloadKey.VERBOSE.value))
        return {CliPayloadKey.BRAIN_STIMULUS.value: brain_stimulus.tolist(), CliPayloadKey.SHAPE.value: list(brain_stimulus.shape)}
    if command == CliCommand.INSPECT_BRAIN_RESPONSE_TABLE.value:
        brain_stimulus_frame = runner.get_brain_stimulus_dataframe(payload.get(CliPayloadKey.INPUT_PATH.value), verbose=payload.get(CliPayloadKey.VERBOSE.value))
        return {CliPayloadKey.BRAIN_STIMULUS.value: brain_stimulus_frame.to_dict(orient="records"), CliPayloadKey.COUNT.value: len(brain_stimulus_frame)}
    if command == CliCommand.SAVE_BUNDLE.value:
        result = runner.run(payload.get(CliPayloadKey.INPUT_PATH.value), verbose=payload.get(CliPayloadKey.VERBOSE.value))
        saved_path = runner.save_output(
            result,
            output_path=payload.get(CliPayloadKey.SAVE_TO.value),
            include_brain_stimulus_csv=payload.get(CliPayloadKey.INCLUDE_BRAIN_STIMULUS_CSV.value),
        )
        return {CliPayloadKey.SAVED_TO.value: str(saved_path)}
    raise ValueError(f"Unsupported command: {command}")


def build_runner(payload: dict[str, Any], runner_factory: Callable[..., Any]) -> Any:
    config_payload = payload.get(CliPayloadKey.CONFIG.value, {})
    if not isinstance(config_payload, dict):
        raise ValueError("config must be a JSON object when provided.")
    config = TribeConfig(**config_payload)
    return runner_factory(config=config)


def serialize_prediction_result(result: PredictionResult, runner: Any) -> dict[str, Any]:
    serialized_segments = result.segments
    workflow = getattr(runner, "workflow", None)
    if workflow is not None and hasattr(workflow, "serialize_segments"):
        serialized_segments = workflow.serialize_segments(result.segments)
    return {
        CliPayloadKey.INPUT_PATH.value: str(result.input_path),
        CliPayloadKey.MODEL_INPUT_PATH.value: str(result.model_input_path),
        CliPayloadKey.INPUT_KIND.value: result.input_kind.value,
        CliPayloadKey.EVENTS.value: result.events.to_dict(orient="records"),
        CliPayloadKey.BRAIN_STIMULUS.value: result.brain_stimulus.tolist(),
        CliPayloadKey.BRAIN_STIMULUS_SHAPE.value: list(result.brain_stimulus.shape),
        CliPayloadKey.SEGMENTS.value: serialized_segments,
    }
