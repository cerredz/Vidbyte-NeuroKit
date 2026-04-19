from __future__ import annotations

from pathlib import Path

import pandas as pd

from libs.dataclasses import FormatRequest
from libs.enums import DataFormat
from libs.utils import Formatter


def test_format_request_accepts_mapping_aliases() -> None:
    request = FormatRequest.from_value({"from": "csv", "to": "tsv"})

    assert request == FormatRequest(source=DataFormat.CSV, target=DataFormat.TSV)


def test_formatter_writes_tsv_from_dataframe(tmp_path: Path) -> None:
    formatter = Formatter()
    frame = pd.DataFrame([{"timepoint": 0, "vertex_0": 1.0}])

    output_path = formatter.to(frame, "tsv", output_path=tmp_path / "brain_stimulus")

    assert output_path == (tmp_path / "brain_stimulus.tsv").resolve()
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "timepoint\tvertex_0"


def test_formatter_writes_bids_directory_from_dataframe(tmp_path: Path) -> None:
    formatter = Formatter()
    frame = pd.DataFrame([{"onset": 0.0, "duration": 1.0, "trial_type": "clip"}])

    output_dir = formatter.to(frame, "bids", output_path=tmp_path / "bids-export")

    assert output_dir == (tmp_path / "bids-export").resolve()
    assert (output_dir / "dataset_description.json").exists()
    assert (output_dir / "sub-01_task-tribe_events.tsv").exists()


def test_formatter_rejects_unsupported_conversion(tmp_path: Path) -> None:
    formatter = Formatter()
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("onset,duration\n0,1\n", encoding="utf-8")

    try:
        formatter.to(csv_path, {"source": "csv", "target": "nifti"}, output_path=tmp_path / "export")
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")
