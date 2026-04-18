from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LocalFileManager:
    def prepare_runner_directories(self, cache_dir: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
        return self.ensure_directory(cache_dir), self.ensure_directory(output_dir)

    def ensure_directory(self, path: str | Path) -> Path:
        resolved_path = Path(path).expanduser().resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path

    def create_output_directory(self, base_output_dir: Path, input_path: Path, output_path: str | Path | None) -> Path:
        if output_path is not None:
            resolved_path = Path(output_path).expanduser().resolve()
            if resolved_path.exists() and resolved_path.is_file():
                raise ValueError(f"Output path must be a directory, got file: {resolved_path}")
            return self.ensure_directory(resolved_path)

        base_path = base_output_dir / input_path.stem
        if not base_path.exists():
            return self.ensure_directory(base_path)

        suffix = 1
        while True:
            candidate = base_output_dir / f"{input_path.stem}-{suffix}"
            if not candidate.exists():
                return self.ensure_directory(candidate)
            suffix += 1

    def write_events(self, directory: Path, events: pd.DataFrame) -> None:
        events.to_csv(directory / "events.csv", index=False)

    def write_brain_stimulus_array(self, directory: Path, brain_stimulus: np.ndarray) -> None:
        np.save(directory / "brain_stimulus.npy", brain_stimulus)

    def write_brain_stimulus_frame(self, directory: Path, brain_stimulus_frame: pd.DataFrame) -> None:
        brain_stimulus_frame.to_csv(directory / "brain_stimulus.csv", index=False)

    def write_json(self, directory: Path, filename: str, payload: Any) -> None:
        (directory / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
