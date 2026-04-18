from __future__ import annotations

from pathlib import Path

import pandas as pd

from libs.enums import InputKind
from libs.utils.data_input import DataInput


class FakeEventsModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def get_events_dataframe(self, **kwargs: str) -> pd.DataFrame:
        self.calls.append(kwargs)
        return pd.DataFrame([kwargs])


def test_data_input_detects_audio_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    data_input = DataInput.from_path(audio_path)

    assert data_input.kind is InputKind.AUDIO
    assert data_input.path == audio_path.resolve()


def test_data_input_detects_video_file(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    data_input = DataInput.from_path(video_path)

    assert data_input.kind is InputKind.VIDEO


def test_data_input_detects_image_file(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")

    data_input = DataInput.from_path(image_path)

    assert data_input.kind is InputKind.IMAGE


def test_data_input_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.mp4"

    try:
        DataInput.from_path(missing_path)
    except FileNotFoundError as exc:
        assert str(missing_path) in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected FileNotFoundError")


def test_data_input_rejects_unsupported_suffix(tmp_path: Path) -> None:
    document_path = tmp_path / "sample.pdf"
    document_path.write_bytes(b"document")

    try:
        DataInput.from_path(document_path)
    except ValueError as exc:
        assert ".pdf" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")


def test_build_events_dataframe_dispatches_audio_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    model = FakeEventsModel()

    prepared_input, events = DataInput.from_path(audio_path).build_events_dataframe(
        model=model,
        working_dir=tmp_path,
    )

    assert prepared_input.model_kwargs == {"audio_path": str(audio_path.resolve())}
    assert model.calls == [{"audio_path": str(audio_path.resolve())}]
    assert list(events.columns) == ["audio_path"]


def test_build_events_dataframe_dispatches_image_as_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")
    prepared_video_path = tmp_path / "prepared.mp4"
    prepared_video_path.write_bytes(b"video")
    model = FakeEventsModel()

    monkeypatch.setattr(
        DataInput,
        "_convert_image_to_video",
        lambda self, working_dir, clip_duration_seconds, fps: prepared_video_path,
    )

    prepared_input, _ = DataInput.from_path(image_path).build_events_dataframe(
        model=model,
        working_dir=tmp_path,
    )

    assert prepared_input.kind is InputKind.IMAGE
    assert prepared_input.model_path == prepared_video_path
    assert model.calls == [{"video_path": str(prepared_video_path)}]
