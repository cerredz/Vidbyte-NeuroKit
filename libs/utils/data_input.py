from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from libs.dataclasses import PreparedInput
from libs.enums import (
    AUDIO_SUFFIX_VALUES,
    IMAGE_SUFFIX_VALUES,
    SUPPORTED_SUFFIX_VALUES,
    TEXT_SUFFIX_VALUES,
    VIDEO_SUFFIX_VALUES,
    InputKind,
)
from libs.protocols import SupportsEventsDataFrame


@dataclass(frozen=True, slots=True)
class DataInput:
    path: Path
    kind: InputKind

    @classmethod
    def from_path(cls, raw_path: str | Path) -> "DataInput":
        path = Path(raw_path).expanduser()
        resolved_path = cls._resolve_existing_file(path)
        return cls(path=resolved_path, kind=cls._detect_kind(resolved_path))

    @staticmethod
    def _resolve_existing_file(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input path must point to a file: {path}")
        return path.resolve()

    @staticmethod
    def _detect_kind(path: Path) -> InputKind:
        suffix = path.suffix.lower()
        if suffix in AUDIO_SUFFIX_VALUES:
            return InputKind.AUDIO
        if suffix in VIDEO_SUFFIX_VALUES:
            return InputKind.VIDEO
        if suffix in IMAGE_SUFFIX_VALUES:
            return InputKind.IMAGE
        if suffix in TEXT_SUFFIX_VALUES:
            return InputKind.TEXT
        raise ValueError(
            "Unsupported input type. "
            f"Expected one of {sorted(SUPPORTED_SUFFIX_VALUES)}, got '{suffix or '<none>'}'."
        )

    def prepare(
        self,
        working_dir: str | Path,
        image_clip_duration_seconds: float = 1.0,
        image_clip_fps: int = 1,
    ) -> PreparedInput:
        working_path = Path(working_dir).expanduser().resolve()
        working_path.mkdir(parents=True, exist_ok=True)

        if self.kind is InputKind.AUDIO:
            return PreparedInput(
                original_path=self.path,
                model_path=self.path,
                kind=self.kind,
                model_kwargs={"audio_path": str(self.path)},
            )

        if self.kind is InputKind.VIDEO:
            return PreparedInput(
                original_path=self.path,
                model_path=self.path,
                kind=self.kind,
                model_kwargs={"video_path": str(self.path)},
            )

        if self.kind is InputKind.TEXT:
            return PreparedInput(
                original_path=self.path,
                model_path=self.path,
                kind=self.kind,
                model_kwargs={"text_path": str(self.path)},
            )

        image_video_path = self._convert_image_to_video(
            working_dir=working_path,
            clip_duration_seconds=image_clip_duration_seconds,
            fps=image_clip_fps,
        )
        return PreparedInput(
            original_path=self.path,
            model_path=image_video_path,
            kind=self.kind,
            model_kwargs={"video_path": str(image_video_path)},
        )

    def build_events_dataframe(
        self,
        model: SupportsEventsDataFrame,
        working_dir: str | Path,
        image_clip_duration_seconds: float = 1.0,
        image_clip_fps: int = 1,
    ) -> tuple[PreparedInput, pd.DataFrame]:
        prepared_input = self.prepare(
            working_dir=working_dir,
            image_clip_duration_seconds=image_clip_duration_seconds,
            image_clip_fps=image_clip_fps,
        )
        events = model.get_events_dataframe(**prepared_input.model_kwargs)
        return prepared_input, events

    def _convert_image_to_video(
        self,
        working_dir: Path,
        clip_duration_seconds: float,
        fps: int,
    ) -> Path:
        if clip_duration_seconds <= 0:
            raise ValueError("image_clip_duration_seconds must be greater than zero.")
        if fps <= 0:
            raise ValueError("image_clip_fps must be greater than zero.")

        prepared_dir = working_dir / "prepared_inputs"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(
            f"{self.path}:{self.path.stat().st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:12]
        output_path = prepared_dir / f"{self.path.stem}-{digest}.mp4"
        if output_path.exists():
            return output_path

        try:
            from moviepy import ImageClip

            clip = ImageClip(str(self.path)).with_duration(clip_duration_seconds)
            clip.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio=False,
                logger=None,
            )
            clip.close()
        except Exception as exc:  # pragma: no cover - exercised by runtime env
            raise RuntimeError(
                "Failed to convert the image input into a temporary video clip. "
                "Ensure moviepy and ffmpeg are available."
            ) from exc

        return output_path
