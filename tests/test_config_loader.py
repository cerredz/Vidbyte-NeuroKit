from __future__ import annotations

from pathlib import Path

from libs.config import ConfigLoader
from libs.dataclasses import TribeConfig


def test_config_loader_reads_root_env_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TRIBE_MODEL_NAME=custom/model",
                "TRIBE_INPUT_PATH=sample.wav",
                "TRIBE_VERBOSE=false",
                "TRIBE_INCLUDE_BRAIN_STIMULUS_CSV=true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = ConfigLoader().load()

    assert config.model_name == "custom/model"
    assert config.input_path == "sample.wav"
    assert config.verbose is False
    assert config.include_brain_stimulus_csv is True


def test_config_loader_prefers_explicit_config_over_env(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("TRIBE_DEVICE=cpu\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = ConfigLoader(TribeConfig(device="cuda")).load()

    assert config.device == "cuda"


def test_tribe_config_rejects_invalid_env_bool(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TRIBE_VERBOSE=maybe\n", encoding="utf-8")

    try:
        TribeConfig.from_env_file(env_path)
    except ValueError as exc:
        assert "Invalid boolean value" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError")
