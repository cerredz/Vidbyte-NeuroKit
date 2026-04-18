from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from libs.dataclasses import TRIBE_FSAVERAGE5_VERTEX_COUNT, TribeConfig, TribePredictions
from libs.enums import TranslationOutputKey
from libs.utils import TribeRunnerUtils, build_tribe_segments
from services.inference import TribeRunner
from tribe_setup.models import PredictionResult


def build_prediction_matrix(first_column: tuple[float, float] = (1.0, 3.0), second_column: tuple[float, float] = (2.0, 4.0)) -> np.ndarray:
    predictions = np.zeros((2, TRIBE_FSAVERAGE5_VERTEX_COUNT), dtype=float)
    predictions[:, 0] = first_column
    predictions[:, 1] = second_column
    return predictions


class FakeTribeBackend:
    def __init__(self) -> None:
        self.events_calls: list[dict[str, str]] = []
        self.predict_calls: list[pd.DataFrame] = []

    def get_events_dataframe(self, **kwargs: str) -> pd.DataFrame:
        self.events_calls.append(kwargs)
        input_type = "Audio"
        if "video_path" in kwargs:
            input_type = "Video"
        if "text_path" in kwargs:
            input_type = "Text"
        return pd.DataFrame([{"type": input_type, "filepath": next(iter(kwargs.values())), "start": 0.0, "duration": 1.0}])

    def predict(self, events: pd.DataFrame, verbose: bool = True) -> tuple[np.ndarray, list[dict[str, float]]]:
        self.predict_calls.append(events)
        return build_prediction_matrix(), [{"offset": 0.0, "duration": 1.0}, {"offset": 1.0, "duration": 1.0}]


def build_runner_config(tmp_path: Path) -> TribeConfig:
    return TribeConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "outputs")


def test_runner_run_returns_prediction_result(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend())

    result = runner.run(audio_path, verbose=False)

    assert isinstance(result, PredictionResult)
    assert result.input_path == audio_path.resolve()
    assert result.brain_stimulus.shape == (2, TRIBE_FSAVERAGE5_VERTEX_COUNT)
    assert len(result.events) == 1


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
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend())
    result = runner.run(audio_path, verbose=False)

    frame = runner.get_brain_stimulus_dataframe(result)

    assert list(frame.columns[:3]) == ["timepoint", "vertex_0", "vertex_1"]
    assert frame.shape == (2, TRIBE_FSAVERAGE5_VERTEX_COUNT + 1)


def test_runner_save_output_writes_bundle(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend())
    result = runner.run(audio_path, verbose=False)

    output_path = runner.save_output(result, include_brain_stimulus_csv=True)

    assert (output_path / "events.csv").exists()
    assert (output_path / "brain_stimulus.npy").exists()
    assert (output_path / "brain_stimulus.csv").exists()
    assert (output_path / "metadata.json").exists()
    assert (output_path / "segments.json").exists()
    assert json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))["brain_stimulus_shape"] == [2, TRIBE_FSAVERAGE5_VERTEX_COUNT]


def test_runner_fails_fast_on_invalid_input(tmp_path: Path) -> None:
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend())

    try:
        runner.run(tmp_path / "missing.wav", verbose=False)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected FileNotFoundError")


def test_runner_uses_yaml_defaults_when_config_is_not_provided(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    runner = TribeRunner(backend=FakeTribeBackend())

    assert runner.cache_dir == (tmp_path / "cache").resolve()
    assert runner.output_dir == (tmp_path / "outputs").resolve()


def test_runner_uses_root_env_for_no_arg_run(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    (tmp_path / ".env").write_text(f"TRIBE_INPUT_PATH={audio_path}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = TribeRunner(backend=FakeTribeBackend()).run()

    assert result.input_path == audio_path.resolve()


def test_runner_run_batch_preserves_input_order_and_supports_mixed_types(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    video_path = tmp_path / "sample.mp4"
    text_path = tmp_path / "sample.txt"
    audio_path.write_bytes(b"audio")
    video_path.write_bytes(b"video")
    text_path.write_text("hello", encoding="utf-8")
    backend = FakeTribeBackend()
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=backend)

    results = runner.run_batch([video_path, audio_path, text_path], verbose=False, max_workers=2)

    assert [result.input_path for result in results] == [video_path.resolve(), audio_path.resolve(), text_path.resolve()]
    assert [result.input_kind.value for result in results] == ["video", "audio", "text"]
    assert len(backend.predict_calls) == 3


def test_runner_translate_returns_requested_outputs(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    utils = TribeRunnerUtils(roi_index_resolver=lambda region: np.array([0]) if region == "G_front_sup" else np.array([1]))
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend(), utils=utils)
    result = runner.run(audio_path, verbose=False)

    translated = runner.translate(
        result,
        [TranslationOutputKey.PEAK, "cognitive", "regions"],
        options={"peak": {"top_n": 1}, "regions": {"regions": ["G_front_sup"]}},
    )

    assert set(translated) == {"peak", "cognitive", "regions"}
    assert translated["peak"].items[0].timestamp_s == 1.0
    assert translated["cognitive"].mean_score == 50.0
    assert translated["regions"].items["G_front_sup"].n_vertices == 1


def test_runner_translate_handles_compare_diff_normalize_segment_and_export(tmp_path: Path) -> None:
    runner = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend())
    predictions_a = TribePredictions(build_prediction_matrix())
    predictions_b = TribePredictions(build_prediction_matrix((0.5, 1.0), (1.0, 1.0)))
    segments = build_tribe_segments([{"onset": 0.0, "duration": 1.0}, {"onset": 1.0, "duration": 1.0}])

    translated = runner.translate(
        predictions_a,
        [
            TranslationOutputKey.COMPARE,
            TranslationOutputKey.DIFF,
            TranslationOutputKey.NORMALIZE,
            TranslationOutputKey.SEGMENT,
            TranslationOutputKey.EXPORT,
        ],
        segments=segments,
        options={
            "compare": {"other": predictions_b, "segments_b": segments, "metric": "engagement"},
            "diff": {"other": predictions_b},
            "normalize": {"baseline": predictions_b},
            "segment": {"threshold": 50.0},
            "export": {"format": "json", "path": tmp_path / "translated-output"},
        },
    )

    assert translated["compare"].winner.value == "tie"
    assert translated["diff"].max_diff_vertex == 1
    assert translated["normalize"].values.shape == predictions_a.values.shape
    assert translated["segment"].pct_high == 50.0
    assert translated["export"].path.exists()


def test_runner_translate_fails_fast_on_invalid_key(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    result = TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend()).run(audio_path, verbose=False)

    try:
        TribeRunner(config=build_runner_config(tmp_path), backend=FakeTribeBackend()).translate(result, ["peak", "invalid-key"])
    except ValueError as exc:
        assert "Unsupported translation key" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")
