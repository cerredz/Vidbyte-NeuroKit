from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from libs.dataclasses import TribeConfig
from services.inference import TribeRunner
from tribe_setup.models import PredictionResult


class FakeTribeBackend:
    def __init__(self) -> None:
        self.events_calls: list[dict[str, str]] = []
        self.predict_calls: list[pd.DataFrame] = []

    def get_events_dataframe(self, **kwargs: str) -> pd.DataFrame:
        self.events_calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "type": "Audio" if "audio_path" in kwargs else "Video",
                    "filepath": next(iter(kwargs.values())),
                    "start": 0.0,
                    "duration": 1.0,
                }
            ]
        )

    def predict(self, events: pd.DataFrame, verbose: bool = True) -> tuple[np.ndarray, list[dict[str, float]]]:
        self.predict_calls.append(events)
        brain_stimulus = np.array([[1.0, 2.0], [3.0, 4.0]])
        segments = [{"offset": 0.0, "duration": 1.0}, {"offset": 1.0, "duration": 1.0}]
        return brain_stimulus, segments


def build_runner_config(tmp_path: Path) -> TribeConfig:
    return TribeConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "outputs")


def test_runner_run_returns_prediction_result(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    backend = FakeTribeBackend()
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=backend)

    result = runner.run(audio_path, verbose=False)

    assert isinstance(result, PredictionResult)
    assert result.input_path == audio_path.resolve()
    assert result.brain_stimulus.shape == (2, 2)
    assert len(result.events) == 1
    assert len(backend.predict_calls) == 1


def test_runner_get_event_dataframe_uses_backend(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    backend = FakeTribeBackend()
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=backend)

    events = runner.get_event_dataframe(video_path)

    assert len(events) == 1
    assert backend.events_calls == [{"video_path": str(video_path.resolve())}]


def test_runner_get_brain_stimulus_dataframe_from_result(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    backend = FakeTribeBackend()
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=backend)
    result = runner.run(audio_path, verbose=False)

    frame = runner.get_brain_stimulus_dataframe(result)

    assert list(frame.columns) == ["timepoint", "vertex_0", "vertex_1"]
    assert frame.shape == (2, 3)


def test_runner_save_output_writes_bundle(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    backend = FakeTribeBackend()
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=backend)
    result = runner.run(audio_path, verbose=False)

    output_path = runner.save_output(result, include_brain_stimulus_csv=True)

    assert (output_path / "events.csv").exists()
    assert (output_path / "brain_stimulus.npy").exists()
    assert (output_path / "brain_stimulus.csv").exists()
    assert (output_path / "metadata.json").exists()
    assert (output_path / "segments.json").exists()

    metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["brain_stimulus_shape"] == [2, 2]

    saved_brain_stimulus = np.load(output_path / "brain_stimulus.npy")
    assert saved_brain_stimulus.shape == (2, 2)


def test_runner_fails_fast_on_invalid_input(tmp_path: Path) -> None:
    backend = FakeTribeBackend()
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=backend)

    try:
        runner.run(tmp_path / "missing.wav", verbose=False)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected FileNotFoundError")

    assert backend.events_calls == []


def test_runner_uses_yaml_defaults_when_config_is_not_provided(tmp_path: Path, monkeypatch) -> None:
    backend = FakeTribeBackend()
    monkeypatch.chdir(tmp_path)

    runner = TribeRunner(backend=backend)

    assert runner.cache_dir == (tmp_path / "cache").resolve()
    assert runner.output_dir == (tmp_path / "outputs").resolve()


def test_runner_uses_root_env_for_no_arg_run(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    (tmp_path / ".env").write_text(f"TRIBE_INPUT_PATH={audio_path}\n", encoding="utf-8")
    backend = FakeTribeBackend()
    monkeypatch.chdir(tmp_path)

    runner = TribeRunner(backend=backend)
    result = runner.run()

    assert result.input_path == audio_path.resolve()
    assert len(backend.predict_calls) == 1
