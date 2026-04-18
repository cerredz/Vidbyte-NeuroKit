from __future__ import annotations

import json
from pathlib import Path

from tribe_cli.main import execute_request, load_payload


class FakeCliRunner:
    def __init__(self, config=None) -> None:
        self.config = config
        self.run_calls: list[dict[str, object]] = []

    def run(self, input_path=None, *, verbose=None, save_to=None):
        self.run_calls.append({"input_path": input_path, "verbose": verbose, "save_to": save_to})
        return FakePredictionResult(input_path or "env.wav")

    def get_event_dataframe(self, input_path):
        return FakePredictionResult(input_path).events

    def get_brain_stimulus(self, input_path=None, *, verbose=None):
        return FakePredictionResult(input_path or "env.wav").brain_stimulus

    def get_brain_stimulus_dataframe(self, input_path=None, *, verbose=None):
        return FakePredictionResult(input_path or "env.wav").brain_stimulus_frame()

    def save_output(self, result, output_path=None, *, include_brain_stimulus_csv=None):
        return Path(output_path or "outputs")


class FakePredictionResult:
    def __init__(self, input_path: str) -> None:
        import numpy as np
        import pandas as pd

        self.input_path = Path(input_path)
        self.model_input_path = Path(input_path)
        self.input_kind = type("InputKindValue", (), {"value": "audio"})()
        self.events = pd.DataFrame([{"filepath": input_path, "start": 0.0}])
        self.brain_stimulus = np.array([[1.0, 2.0]])
        self.segments = [{"offset": 0.0, "duration": 1.0}]

    def brain_stimulus_frame(self):
        import pandas as pd

        return pd.DataFrame([{"timepoint": 0, "vertex_0": 1.0, "vertex_1": 2.0}])


def fake_runner_factory(config=None):
    return FakeCliRunner(config=config)


def test_execute_request_run_returns_json_safe_payload() -> None:
    response = execute_request(
        "run",
        {"input_path": "sample.wav", "config": {"device": "cpu"}},
        runner_factory=fake_runner_factory,
    )

    assert response["input_path"] == "sample.wav"
    assert response["brain_stimulus_shape"] == [1, 2]


def test_execute_request_save_output_returns_path() -> None:
    response = execute_request(
        "save-output",
        {"input_path": "sample.wav", "save_to": "bundle"},
        runner_factory=fake_runner_factory,
    )

    assert response == {"saved_to": str(Path("bundle"))}


def test_load_payload_requires_json_object() -> None:
    try:
        load_payload('["not", "an", "object"]')
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")


def test_execute_request_rejects_non_object_config() -> None:
    try:
        execute_request("run", {"config": "bad"}, runner_factory=fake_runner_factory)
    except ValueError as exc:
        assert "config must be a JSON object" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")
