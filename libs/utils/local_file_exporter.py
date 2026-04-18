from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.dataclasses import ExportArtifact
from libs.enums import ExportFormat
from libs.utils.local_file_manager import LocalFileManager
from libs.utils.tribe_utils import result_to_csv_frame, to_json_safe_value


class LocalFileExporter:
    def __init__(self, file_manager: LocalFileManager | None = None) -> None:
        self.file_manager = file_manager or LocalFileManager()

    def export(self, result: Any, format: ExportFormat = ExportFormat.JSON, path: str | Path = "output") -> ExportArtifact:
        match format:
            case ExportFormat.JSON:
                return ExportArtifact(path=self.write_json_export(path, result), format=format)
            case ExportFormat.CSV:
                return ExportArtifact(path=self.write_csv_export(path, result), format=format)
            case ExportFormat.NIFTI:
                raise NotImplementedError(
                    "NIfTI export is not wired in this wrapper yet. Use tribev2 surface-to-volume projection utilities for MNI export."
                )

    def write_json_export(self, path: str | Path, payload: Any) -> Path:
        target = Path(path).expanduser().with_suffix(".json")
        self.file_manager.ensure_directory(target.parent)
        target.write_text(json.dumps(to_json_safe_value(payload), indent=2), encoding="utf-8")
        return target.resolve()

    def write_csv_export(self, path: str | Path, payload: Any) -> Path:
        target = Path(path).expanduser().with_suffix(".csv")
        self.file_manager.ensure_directory(target.parent)
        result_to_csv_frame(payload).to_csv(target, index=False)
        return target.resolve()
