from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from libs.enums import DataFormat
from libs.utils.formatter.formatter_interface import FormatterInterface


TABULAR_DATA_FORMATS: frozenset[DataFormat] = frozenset(
    {
        DataFormat.CSV,
        DataFormat.TSV,
        DataFormat.BIDS_EVENTS,
        DataFormat.AFNI_1D,
        DataFormat.ADJACENCY_MATRIX,
    }
)


class BaseFormatter(FormatterInterface):
    format_type: DataFormat
    extensions: tuple[str, ...] = ()

    def __init__(self, *, supported_sources: Iterable[DataFormat] | None = None) -> None:
        self.supported_sources = frozenset(supported_sources or {self.format_type})

    def matches_path(self, path: Path) -> bool:
        normalized_path = str(path).lower()
        return any(normalized_path.endswith(extension) for extension in self.extensions)

    def can_convert_from(self, source_format: DataFormat) -> bool:
        return source_format in self.supported_sources

    def from_(self, source: Any) -> Any:
        if isinstance(source, (str, Path)):
            return self._resolve_existing_path(source)
        return source

    def to(self, data: Any, *, output_path: str | Path | None = None) -> Any:
        if output_path is None:
            return data

        source_path = self._resolve_existing_path(data)
        target_path = self._resolve_output_path(output_path)
        if source_path.is_dir():
            raise NotImplementedError(
                f"{self.format_type.value} directory exports require a format-specific implementation."
            )

        shutil.copyfile(source_path, target_path)
        return target_path

    def _resolve_existing_path(self, source: str | Path) -> Path:
        resolved_path = Path(source).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {resolved_path}")
        return resolved_path

    def _resolve_output_path(self, output_path: str | Path) -> Path:
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.extensions[0] if self.extensions else ""
        if suffix and not str(target).lower().endswith(suffix):
            target = Path(f"{target}{suffix}")
        return target.resolve()


class DelimitedTableFormatter(BaseFormatter):
    delimiter = ","
    read_delimiter = ","
    include_header = True

    def __init__(self) -> None:
        super().__init__(supported_sources=TABULAR_DATA_FORMATS)

    def from_(self, source: Any) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, np.ndarray):
            return pd.DataFrame(source)
        if isinstance(source, (list, tuple)):
            return pd.DataFrame(source)

        path = self._resolve_existing_path(source)
        kwargs: dict[str, Any] = {"sep": self.read_delimiter, "engine": "python"}
        if not self.include_header:
            kwargs["header"] = None
        return pd.read_csv(path, **kwargs)

    def to(self, data: Any, *, output_path: str | Path | None = None) -> pd.DataFrame | Path:
        frame = self._coerce_frame(data)
        if output_path is None:
            return frame

        target_path = self._resolve_output_path(output_path)
        frame.to_csv(
            target_path,
            sep=self.delimiter,
            index=False,
            header=self.include_header,
        )
        return target_path

    def _coerce_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        if isinstance(data, (list, tuple)):
            return pd.DataFrame(data)
        if isinstance(data, (str, Path)):
            return self.from_(data)
        raise TypeError(
            f"{self.format_type.value} formatter expects a DataFrame, ndarray, sequence, or path."
        )


class BidsDirectoryFormatter(BaseFormatter):
    format_type = DataFormat.BIDS

    def __init__(self) -> None:
        super().__init__(
            supported_sources={
                DataFormat.BIDS,
                DataFormat.BIDS_EVENTS,
                DataFormat.CSV,
                DataFormat.TSV,
                DataFormat.AFNI_1D,
            }
        )

    def matches_path(self, path: Path) -> bool:
        if path.is_dir():
            return (path / "dataset_description.json").is_file()
        return path.name == "dataset_description.json"

    def from_(self, source: Any) -> pd.DataFrame:
        path = self._resolve_existing_path(source)
        dataset_root = path if path.is_dir() else path.parent
        event_files = sorted(dataset_root.rglob("*_events.tsv"))
        if not event_files:
            raise FileNotFoundError(f"No BIDS events.tsv file found in {dataset_root}")
        return pd.read_csv(event_files[0], sep="\t")

    def to(self, data: Any, *, output_path: str | Path | None = None) -> Path | dict[str, Any]:
        frame = self._coerce_frame(data)
        if output_path is None:
            return {
                "dataset_description": {
                    "Name": "Tribe export",
                    "BIDSVersion": "1.9.0",
                    "DatasetType": "derivative",
                },
                "events": frame.to_dict(orient="records"),
            }

        target_dir = Path(output_path).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "dataset_description.json").write_text(
            json.dumps(
                {
                    "Name": "Tribe export",
                    "BIDSVersion": "1.9.0",
                    "DatasetType": "derivative",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        frame.to_csv(target_dir / "sub-01_task-tribe_events.tsv", sep="\t", index=False)
        return target_dir.resolve()

    @staticmethod
    def _coerce_frame(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        if isinstance(data, (list, tuple)):
            return pd.DataFrame(data)
        raise TypeError("bids formatter expects a DataFrame, ndarray, or sequence.")


class Hdf5TableFormatter(BaseFormatter):
    format_type = DataFormat.HDF5
    extensions = (".h5", ".hdf5")

    def __init__(self) -> None:
        super().__init__(supported_sources=TABULAR_DATA_FORMATS | {DataFormat.HDF5})

    def from_(self, source: Any) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, np.ndarray):
            return pd.DataFrame(source)
        if isinstance(source, (list, tuple)):
            return pd.DataFrame(source)

        path = self._resolve_existing_path(source)
        try:
            return pd.read_hdf(path, key="tribe")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("HDF5 formatting requires optional PyTables support.") from exc

    def to(self, data: Any, *, output_path: str | Path | None = None) -> pd.DataFrame | Path:
        frame = self.from_(data)
        if output_path is None:
            return frame

        target_path = self._resolve_output_path(output_path)
        try:
            frame.to_hdf(target_path, key="tribe", mode="w")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("HDF5 formatting requires optional PyTables support.") from exc
        return target_path
