from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from libs.dataclasses import PredictionResult, TribeConfig
from services.inference import TribeRunner


COMMANDS = {
    "run",
    "get-event-dataframe",
    "get-brain-stimulus",
    "get-brain-stimulus-dataframe",
    "save-output",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = load_payload(args.json_payload)

    try:
        response = execute_request(args.command, payload)
    except Exception as exc:
        print(json.dumps({"ok": False, "command": args.command, "error": str(exc)}))
        return 1

    print(json.dumps({"ok": True, "command": args.command, "result": response}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tribe-cli")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--json", dest="json_payload", help="Inline JSON payload. If omitted, stdin is used when piped.")
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


def execute_request(command: str, payload: dict[str, Any], runner_factory: Callable[..., Any] = TribeRunner) -> dict[str, Any]:
    if command not in COMMANDS:
        raise ValueError(f"Unsupported command: {command}")

    runner = build_runner(payload, runner_factory)
    if command == "run":
        return serialize_prediction_result(
            runner.run(
                payload.get("input_path"),
                verbose=payload.get("verbose"),
                save_to=payload.get("save_to"),
            ),
            runner,
        )
    if command == "get-event-dataframe":
        events = runner.get_event_dataframe(payload.get("input_path"))
        return {"events": events.to_dict(orient="records"), "count": len(events)}
    if command == "get-brain-stimulus":
        brain_stimulus = runner.get_brain_stimulus(payload.get("input_path"), verbose=payload.get("verbose"))
        return {"brain_stimulus": brain_stimulus.tolist(), "shape": list(brain_stimulus.shape)}
    if command == "get-brain-stimulus-dataframe":
        brain_stimulus_frame = runner.get_brain_stimulus_dataframe(payload.get("input_path"), verbose=payload.get("verbose"))
        return {"brain_stimulus": brain_stimulus_frame.to_dict(orient="records"), "count": len(brain_stimulus_frame)}
    if command == "save-output":
        result = runner.run(payload.get("input_path"), verbose=payload.get("verbose"))
        saved_path = runner.save_output(
            result,
            output_path=payload.get("save_to"),
            include_brain_stimulus_csv=payload.get("include_brain_stimulus_csv"),
        )
        return {"saved_to": str(saved_path)}
    raise ValueError(f"Unsupported command: {command}")


def build_runner(payload: dict[str, Any], runner_factory: Callable[..., Any]) -> Any:
    config_payload = payload.get("config", {})
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
        "input_path": str(result.input_path),
        "model_input_path": str(result.model_input_path),
        "input_kind": result.input_kind.value,
        "events": result.events.to_dict(orient="records"),
        "brain_stimulus": result.brain_stimulus.tolist(),
        "brain_stimulus_shape": list(result.brain_stimulus.shape),
        "segments": serialized_segments,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
