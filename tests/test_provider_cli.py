from __future__ import annotations

import importlib
from pathlib import Path

from libs.dataclasses.provider_models import DownloadedProviderAsset
from libs.providers.connection_store import ProviderConnectionStore
from services.providers.base import ProviderRunnerBase

cli = importlib.import_module("tribe_cli.main")


class FakeProviderForConnect:
    def __init__(self, credentials) -> None:
        self.credentials = credentials

    def get_account(self):
        return {"id": "acct_1", "name": "Connected"}


class FakeMetaProvider:
    def __init__(self, credentials) -> None:
        self.credentials = credentials


class FakeMetaRunner:
    def __init__(self, provider, tribe_runner=None) -> None:
        self.provider = provider
        self.tribe_runner = tribe_runner

    def analyze_campaign(self, campaign_id: str, *, metrics=None, sort_by=None):
        return {"campaign_id": campaign_id, "metrics": metrics, "sort_by": sort_by}


class FakeSlackProvider:
    def __init__(self, credentials) -> None:
        self.credentials = credentials


class FakeSlackRunner:
    def __init__(self, provider, tribe_runner=None) -> None:
        self.provider = provider
        self.tribe_runner = tribe_runner

    def analyze_file(self, file_id: str, *, metrics=None, save_to=None):
        return {"file_id": file_id, "metrics": metrics, "save_to": save_to}


class FakeTemporalResult:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores


class FakeTribeRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.cache_dir = tmp_path / "cache"
        self.output_dir = tmp_path / "outputs"

    def run(self, input_path, *, save_to=None):
        return {"input_path": str(input_path)}

    def translate(self, prediction, outputs, options=None):
        score = 10.0 if str(prediction["input_path"]).endswith("a.mp4") else 90.0
        return {"temporal": FakeTemporalResult((score,))}


class DummyProvider:
    def __init__(self) -> None:
        self.download_dir = Path(".")


class DummyRunner(ProviderRunnerBase):
    provider_name = "dummy"


def test_connect_command_stores_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "VimeoProvider", FakeProviderForConnect)
    store = ProviderConnectionStore(tmp_path / "connections.json")
    args = cli.build_provider_parser().parse_args(["connect", "vimeo", "--token", "secret-token"])

    response = cli.execute_provider_request(args, connection_store=store)

    assert response["provider"] == "vimeo"
    assert response["stored_credentials"]["token"].startswith("secr")
    assert store.load("vimeo") == {"token": "secret-token"}


def test_analyze_meta_ads_uses_stored_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "MetaMarketingProvider", FakeMetaProvider)
    monkeypatch.setattr(cli, "MetaMarketingRunner", FakeMetaRunner)
    store = ProviderConnectionStore(tmp_path / "connections.json")
    store.save("meta-ads", {"token": "stored-token", "account_id": "act_123"})
    args = cli.build_provider_parser().parse_args(
        [
            "analyze",
            "meta-ads",
            "--campaign-id",
            "123",
            "--all-creatives",
            "--metrics",
            "engagement,cognitive-load",
            "--sort-by",
            "engagement",
        ]
    )

    response = cli.execute_provider_request(args, connection_store=store, runner_factory=lambda config=None: object())

    assert response == {
        "campaign_id": "123",
        "metrics": ["engagement", "cognitive-load"],
        "sort_by": "engagement",
    }


def test_analyze_slack_uses_stored_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "SlackProvider", FakeSlackProvider)
    monkeypatch.setattr(cli, "SlackRunner", FakeSlackRunner)
    store = ProviderConnectionStore(tmp_path / "connections.json")
    store.save("slack", {"token": "stored-token"})
    args = cli.build_provider_parser().parse_args(
        [
            "analyze",
            "slack",
            "--file-id",
            "F123",
            "--metrics",
            "engagement,peak",
        ]
    )

    response = cli.execute_provider_request(args, connection_store=store, runner_factory=lambda config=None: object())

    assert response == {
        "file_id": "F123",
        "metrics": ["engagement", "peak"],
        "save_to": None,
    }


def test_provider_runner_batch_analysis_sorts_by_metric(tmp_path: Path) -> None:
    runner = DummyRunner(provider=DummyProvider(), tribe_runner=FakeTribeRunner(tmp_path))
    asset_a = DownloadedProviderAsset(provider="dummy", remote_id="a", name="A", local_path=tmp_path / "a.mp4")
    asset_b = DownloadedProviderAsset(provider="dummy", remote_id="b", name="B", local_path=tmp_path / "b.mp4")

    result = runner.analyze_assets([asset_a, asset_b], metrics=["engagement"], sort_by="engagement")

    assert [item.asset.remote_id for item in result.items] == ["b", "a"]
