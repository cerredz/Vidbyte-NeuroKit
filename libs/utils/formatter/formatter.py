from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from libs.dataclasses import FormatRequest
from libs.enums import DataFormat
from libs.utils.formatter.adjacency_matrix_formatter import AdjacencyMatrixFormatter
from libs.utils.formatter.afni_1d_formatter import Afni1DFormatter
from libs.utils.formatter.analyze_formatter import AnalyzeFormatter
from libs.utils.formatter.bids_events_formatter import BidsEventsFormatter
from libs.utils.formatter.bids_formatter import BidsFormatter
from libs.utils.formatter.brainvision_formatter import BrainVisionFormatter
from libs.utils.formatter.cifti_formatter import CiftiFormatter
from libs.utils.formatter.csv_formatter import CsvFormatter
from libs.utils.formatter.edf_formatter import EdfFormatter
from libs.utils.formatter.eeglab_set_formatter import EeglabSetFormatter
from libs.utils.formatter.fif_formatter import FifFormatter
from libs.utils.formatter.formatter_interface import FormatterInterface
from libs.utils.formatter.gifti_formatter import GiftiFormatter
from libs.utils.formatter.graphml_formatter import GraphMlFormatter
from libs.utils.formatter.hdf5_formatter import Hdf5Formatter
from libs.utils.formatter.minc_formatter import MincFormatter
from libs.utils.formatter.nifti_formatter import NiftiFormatter
from libs.utils.formatter.tsv_formatter import TsvFormatter


class Formatter:
    def __init__(self, formatters: Sequence[FormatterInterface] | None = None) -> None:
        formatter_instances = formatters or (
            BidsFormatter(),
            BidsEventsFormatter(),
            CiftiFormatter(),
            NiftiFormatter(),
            AnalyzeFormatter(),
            MincFormatter(),
            Afni1DFormatter(),
            CsvFormatter(),
            TsvFormatter(),
            Hdf5Formatter(),
            GiftiFormatter(),
            GraphMlFormatter(),
            AdjacencyMatrixFormatter(),
            EdfFormatter(),
            BrainVisionFormatter(),
            EeglabSetFormatter(),
            FifFormatter(),
        )
        self._formatters: dict[DataFormat, FormatterInterface] = {
            formatter.format_type: formatter for formatter in formatter_instances
        }

    @property
    def supported_formats(self) -> tuple[DataFormat, ...]:
        return tuple(self._formatters)

    def from_(
        self,
        source: Any,
        format: FormatRequest | DataFormat | str | Mapping[str, Any] | None = None,
    ) -> Any:
        request = self._resolve_request(format)
        source_format = self._resolve_source_format(source, request.source if request else None)
        if source_format is None:
            return source
        return self._get_formatter(source_format).from_(source)

    def to(
        self,
        source: Any,
        format: FormatRequest | DataFormat | str | Mapping[str, Any],
        *,
        output_path: str | Path | None = None,
    ) -> Any:
        request = self._resolve_request(format)
        if request is None:
            raise ValueError("A target format is required.")

        target_format = request.target or request.source
        if target_format is None or not self._is_valid_format(target_format):
            raise ValueError("A supported target format is required.")

        target_formatter = self._get_formatter(target_format)
        source_format = self._resolve_source_format(source, request.source)
        if source_format is None:
            return target_formatter.to(source, output_path=output_path)

        if not self._is_valid_conversion(source_format, target_format):
            raise ValueError(
                f"Conversion from '{source_format.value}' to '{target_format.value}' is not supported."
            )

        normalized_source = self._get_formatter(source_format).from_(source)
        return target_formatter.to(normalized_source, output_path=output_path)

    def validate_request(
        self,
        source: Any | None = None,
        format: FormatRequest | DataFormat | str | Mapping[str, Any] | None = None,
    ) -> FormatRequest | None:
        request = self._resolve_request(format)
        if request is None:
            return None

        if request.source is not None and not self._is_valid_format(request.source):
            raise ValueError(f"Unsupported source format '{request.source.value}'.")
        if request.target is not None and not self._is_valid_format(request.target):
            raise ValueError(f"Unsupported target format '{request.target.value}'.")

        source_format = self._resolve_source_format(source, request.source) if source is not None else request.source
        target_format = request.target or request.source
        if source_format is not None and target_format is not None and not self._is_valid_conversion(source_format, target_format):
            raise ValueError(
                f"Conversion from '{source_format.value}' to '{target_format.value}' is not supported."
            )
        return request

    def _is_valid_format(self, format_value: DataFormat) -> bool:
        return format_value in self._formatters

    def _is_valid_conversion(self, source_format: DataFormat, target_format: DataFormat) -> bool:
        if not self._is_valid_format(source_format) or not self._is_valid_format(target_format):
            return False
        if source_format == target_format:
            return True
        return self._get_formatter(target_format).can_convert_from(source_format)

    def _resolve_source_format(
        self,
        source: Any,
        explicit_source: DataFormat | None = None,
    ) -> DataFormat | None:
        if explicit_source is not None:
            return explicit_source
        if not isinstance(source, (str, Path)):
            return None

        path = Path(source).expanduser()
        for formatter in self._formatters.values():
            if formatter.matches_path(path):
                return formatter.format_type
        return None

    def _get_formatter(self, format_value: DataFormat) -> FormatterInterface:
        return self._formatters[format_value]

    @staticmethod
    def _resolve_request(
        format: FormatRequest | DataFormat | str | Mapping[str, Any] | None,
    ) -> FormatRequest | None:
        return FormatRequest.from_value(format)
