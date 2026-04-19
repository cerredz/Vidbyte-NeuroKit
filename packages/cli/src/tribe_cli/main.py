from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from libs.dataclasses import PredictionResult, TribeConfig
from libs.providers.connection_store import ProviderConnectionStore
from libs.providers.dropbox import DropboxCredentials, DropboxProvider
from libs.providers.google_drive import GoogleDriveCredentials, GoogleDriveProvider
from libs.providers.meta_marketing import MetaMarketingCredentials, MetaMarketingProvider
from libs.providers.slack import SlackCredentials, SlackProvider
from libs.providers.vimeo import VimeoCredentials, VimeoProvider
from libs.utils.tribe_utils import to_json_safe_value
from services.inference import TribeRunner
from services.providers import DropboxRunner, GoogleDriveRunner, MetaMarketingRunner, SlackRunner, VimeoRunner
from tribe_cli.utils import run_cli


LEGACY_COMMANDS = {
    "run",
    "get-event-dataframe",
    "get-brain-stimulus",
    "get-brain-stimulus-dataframe",
    "save-output",
}
PROVIDER_COMMANDS = {"connect", "analyze", "compare"}


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if raw_args and raw_args[0] in LEGACY_COMMANDS:
        return _run_legacy_cli(raw_args)
    if raw_args and raw_args[0] in PROVIDER_COMMANDS:
        return _run_provider_cli(raw_args)
    return run_cli(raw_args)


def _run_legacy_cli(argv: list[str]) -> int:
    parser = build_legacy_parser()
    args = parser.parse_args(argv)
    payload = load_payload(args.json_payload)

    try:
        response = execute_request(args.command, payload)
    except Exception as exc:
        print(json.dumps({"ok": False, "command": args.command, "error": str(exc)}))
        return 1

    print(json.dumps({"ok": True, "command": args.command, "result": response}))
    return 0


def _run_provider_cli(argv: list[str]) -> int:
    parser = build_provider_parser()
    args = parser.parse_args(argv)

    try:
        response = execute_provider_request(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "command": getattr(args, "command", None), "error": str(exc)}))
        return 1

    print(json.dumps({"ok": True, "command": args.command, "result": to_json_safe_value(response)}))
    return 0


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tribe-cli")
    parser.add_argument("command", choices=sorted(LEGACY_COMMANDS))
    parser.add_argument("--json", dest="json_payload", help="Inline JSON payload. If omitted, stdin is used when piped.")
    return parser


def build_provider_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurokit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    connect_parser = subparsers.add_parser("connect")
    connect_subparsers = connect_parser.add_subparsers(dest="provider", required=True)
    _add_token_provider(connect_subparsers, "vimeo")
    _add_token_provider(connect_subparsers, "google-drive")
    _add_token_provider(connect_subparsers, "dropbox")
    _add_token_provider(connect_subparsers, "slack")
    meta_connect = _add_token_provider(connect_subparsers, "meta-ads")
    meta_connect.add_argument("--account-id", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_subparsers = analyze_parser.add_subparsers(dest="provider", required=True)
    vimeo_analyze = _add_common_analyze_parser(analyze_subparsers, "vimeo")
    vimeo_analyze.add_argument("--video-id", required=True)
    drive_analyze = _add_common_analyze_parser(analyze_subparsers, "google-drive")
    drive_analyze.add_argument("--file-id", required=True)
    dropbox_analyze = _add_common_analyze_parser(analyze_subparsers, "dropbox")
    dropbox_analyze.add_argument("--path", required=True)
    slack_analyze = _add_common_analyze_parser(analyze_subparsers, "slack")
    slack_analyze.add_argument("--file-id", required=True)
    meta_analyze = _add_common_analyze_parser(analyze_subparsers, "meta-ads")
    meta_analyze.add_argument("--creative-id")
    meta_analyze.add_argument("--campaign-id")
    meta_analyze.add_argument("--all-creatives", action="store_true")
    meta_analyze.add_argument("--account-id")

    compare_parser = subparsers.add_parser("compare")
    compare_subparsers = compare_parser.add_subparsers(dest="provider", required=True)
    vimeo_compare = _add_common_compare_parser(compare_subparsers, "vimeo")
    vimeo_compare.add_argument("--video-id", action="append", required=True)
    drive_compare = _add_common_compare_parser(compare_subparsers, "google-drive")
    drive_compare.add_argument("--file-id", action="append", required=True)
    dropbox_compare = _add_common_compare_parser(compare_subparsers, "dropbox")
    dropbox_compare.add_argument("--path", action="append", required=True)
    slack_compare = _add_common_compare_parser(compare_subparsers, "slack")
    slack_compare.add_argument("--file-id", action="append", required=True)
    meta_compare = _add_common_compare_parser(compare_subparsers, "meta-ads")
    meta_compare.add_argument("--creative-id", action="append", required=True)
    meta_compare.add_argument("--account-id")

    return parser


def _add_token_provider(subparsers: Any, name: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("--token", required=True)
    return parser


def _add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--verbose", action="store_true")


def _add_common_analyze_parser(subparsers: Any, name: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("--token")
    parser.add_argument("--metrics", default="engagement")
    parser.add_argument("--save-to")
    parser.add_argument("--sort-by")
    _add_common_runner_args(parser)
    return parser


def _add_common_compare_parser(subparsers: Any, name: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("--token")
    parser.add_argument("--metric", required=True)
    _add_common_runner_args(parser)
    return parser


def load_payload(raw_payload: str | None) -> dict[str, Any]:
    payload_text = raw_payload
    if payload_text is None and not sys.stdin.isatty():
        payload_text = sys.stdin.read()
    if payload_text is None or not payload_text.strip():
        return {}

    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("CLI payload must decode to a JSON object.")
    return payload


def execute_request(command: str, payload: dict[str, Any], runner_factory: Callable[..., Any] = TribeRunner) -> dict[str, Any]:
    if command not in LEGACY_COMMANDS:
        raise ValueError(f"Unsupported command: {command}")

    runner = build_runner(payload, runner_factory)
    if command == "run":
        return serialize_prediction_result(
            runner.run(
                payload.get("input_path"),
                verbose=payload.get("verbose"),
                save_to=payload.get("save_to"),
            ),
            runner,
        )
    if command == "get-event-dataframe":
        events = runner.get_event_dataframe(payload.get("input_path"))
        return {"events": events.to_dict(orient="records"), "count": len(events)}
    if command == "get-brain-stimulus":
        brain_stimulus = runner.get_brain_stimulus(payload.get("input_path"), verbose=payload.get("verbose"))
        return {"brain_stimulus": brain_stimulus.tolist(), "shape": list(brain_stimulus.shape)}
    if command == "get-brain-stimulus-dataframe":
        brain_stimulus_frame = runner.get_brain_stimulus_dataframe(payload.get("input_path"), verbose=payload.get("verbose"))
        return {"brain_stimulus": brain_stimulus_frame.to_dict(orient="records"), "count": len(brain_stimulus_frame)}
    if command == "save-output":
        result = runner.run(payload.get("input_path"), verbose=payload.get("verbose"))
        saved_path = runner.save_output(
            result,
            output_path=payload.get("save_to"),
            include_brain_stimulus_csv=payload.get("include_brain_stimulus_csv"),
        )
        return {"saved_to": str(saved_path)}
    raise ValueError(f"Unsupported command: {command}")


def execute_provider_request(
    args: argparse.Namespace,
    *,
    connection_store: ProviderConnectionStore | None = None,
    runner_factory: Callable[..., TribeRunner] = TribeRunner,
) -> Any:
    store = connection_store or ProviderConnectionStore()
    if args.command == "connect":
        return execute_connect(args, store=store)
    if args.command == "analyze":
        return execute_analyze(args, store=store, runner_factory=runner_factory)
    if args.command == "compare":
        return execute_compare(args, store=store, runner_factory=runner_factory)
    raise ValueError(f"Unsupported command: {args.command}")


def execute_connect(args: argparse.Namespace, *, store: ProviderConnectionStore) -> dict[str, Any]:
    if args.provider == "vimeo":
        credentials = VimeoCredentials(token=args.token)
        provider = VimeoProvider(credentials)
    elif args.provider == "google-drive":
        credentials = GoogleDriveCredentials(token=args.token)
        provider = GoogleDriveProvider(credentials)
    elif args.provider == "dropbox":
        credentials = DropboxCredentials(token=args.token)
        provider = DropboxProvider(credentials)
    elif args.provider == "slack":
        credentials = SlackCredentials(token=args.token)
        provider = SlackProvider(credentials)
    elif args.provider == "meta-ads":
        credentials = MetaMarketingCredentials(token=args.token, account_id=args.account_id)
        provider = MetaMarketingProvider(credentials)
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    account = provider.get_account()
    store.save(args.provider, credentials.to_payload())
    return {"provider": args.provider, "stored_credentials": credentials.to_redacted_payload(), "account": account}


def execute_analyze(
    args: argparse.Namespace,
    *,
    store: ProviderConnectionStore,
    runner_factory: Callable[..., TribeRunner],
) -> Any:
    metrics = parse_csv_list(args.metrics)
    if args.provider == "vimeo":
        credentials = resolve_vimeo_credentials(args, store)
        runner = VimeoRunner(VimeoProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.analyze_video(args.video_id, metrics=metrics, save_to=args.save_to)
    if args.provider == "google-drive":
        credentials = resolve_google_drive_credentials(args, store)
        runner = GoogleDriveRunner(GoogleDriveProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.analyze_file(args.file_id, metrics=metrics, save_to=args.save_to)
    if args.provider == "dropbox":
        credentials = resolve_dropbox_credentials(args, store)
        runner = DropboxRunner(DropboxProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.analyze_file(args.path, metrics=metrics, save_to=args.save_to)
    if args.provider == "slack":
        credentials = resolve_slack_credentials(args, store)
        runner = SlackRunner(SlackProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.analyze_file(args.file_id, metrics=metrics, save_to=args.save_to)
    if args.provider == "meta-ads":
        credentials = resolve_meta_credentials(args, store)
        runner = MetaMarketingRunner(MetaMarketingProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        if args.creative_id:
            return runner.analyze_creative(args.creative_id, metrics=metrics, save_to=args.save_to)
        if args.campaign_id and args.all_creatives:
            return runner.analyze_campaign(args.campaign_id, metrics=metrics, sort_by=args.sort_by or "engagement")
        raise ValueError("meta-ads analyze requires either --creative-id or --campaign-id with --all-creatives.")
    raise ValueError(f"Unsupported provider: {args.provider}")


def execute_compare(
    args: argparse.Namespace,
    *,
    store: ProviderConnectionStore,
    runner_factory: Callable[..., TribeRunner],
) -> Any:
    if args.provider == "vimeo":
        credentials = resolve_vimeo_credentials(args, store)
        video_ids = require_exactly_two(args.video_id, "--video-id")
        runner = VimeoRunner(VimeoProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.compare_videos(video_ids[0], video_ids[1], metric=args.metric)
    if args.provider == "google-drive":
        credentials = resolve_google_drive_credentials(args, store)
        file_ids = require_exactly_two(args.file_id, "--file-id")
        runner = GoogleDriveRunner(GoogleDriveProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.compare_files(file_ids[0], file_ids[1], metric=args.metric)
    if args.provider == "dropbox":
        credentials = resolve_dropbox_credentials(args, store)
        paths = require_exactly_two(args.path, "--path")
        runner = DropboxRunner(DropboxProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.compare_files(paths[0], paths[1], metric=args.metric)
    if args.provider == "slack":
        credentials = resolve_slack_credentials(args, store)
        file_ids = require_exactly_two(args.file_id, "--file-id")
        runner = SlackRunner(SlackProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.compare_files(file_ids[0], file_ids[1], metric=args.metric)
    if args.provider == "meta-ads":
        credentials = resolve_meta_credentials(args, store)
        creative_ids = require_exactly_two(args.creative_id, "--creative-id")
        runner = MetaMarketingRunner(MetaMarketingProvider(credentials), tribe_runner=build_provider_runner(args, runner_factory))
        return runner.compare_assets(
            runner.provider.fetch_media(creative_ids[0]),
            runner.provider.fetch_media(creative_ids[1]),
            metric=args.metric,
        )
    raise ValueError(f"Unsupported provider: {args.provider}")


def build_provider_runner(args: argparse.Namespace, runner_factory: Callable[..., TribeRunner]) -> TribeRunner:
    config = TribeConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        verbose=True if getattr(args, "verbose", False) else None,
    )
    return runner_factory(config=config)


def parse_csv_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def require_exactly_two(values: Sequence[str], flag_name: str) -> tuple[str, str]:
    if len(values) != 2:
        raise ValueError(f"{flag_name} must be provided exactly twice.")
    return values[0], values[1]


def resolve_vimeo_credentials(args: argparse.Namespace, store: ProviderConnectionStore) -> VimeoCredentials:
    return VimeoCredentials(token=resolve_required_secret(args.token, store.load("vimeo"), "vimeo"))


def resolve_google_drive_credentials(args: argparse.Namespace, store: ProviderConnectionStore) -> GoogleDriveCredentials:
    return GoogleDriveCredentials(token=resolve_required_secret(args.token, store.load("google-drive"), "google-drive"))


def resolve_dropbox_credentials(args: argparse.Namespace, store: ProviderConnectionStore) -> DropboxCredentials:
    return DropboxCredentials(token=resolve_required_secret(args.token, store.load("dropbox"), "dropbox"))


def resolve_slack_credentials(args: argparse.Namespace, store: ProviderConnectionStore) -> SlackCredentials:
    return SlackCredentials(token=resolve_required_secret(args.token, store.load("slack"), "slack"))


def resolve_meta_credentials(args: argparse.Namespace, store: ProviderConnectionStore) -> MetaMarketingCredentials:
    stored = store.load("meta-ads") or {}
    token = resolve_required_secret(args.token, stored, "meta-ads")
    account_id = args.account_id or stored.get("account_id")
    if not account_id:
        raise ValueError("Meta Ads requires --account-id or a stored connected account.")
    return MetaMarketingCredentials(token=token, account_id=str(account_id))


def resolve_required_secret(explicit_value: str | None, stored: dict[str, Any] | None, provider_name: str) -> str:
    if explicit_value:
        return explicit_value
    stored_payload = stored or {}
    token = stored_payload.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError(f"No stored credentials found for {provider_name}. Run `neurokit connect {provider_name}` or pass --token.")
    return token


def build_runner(payload: dict[str, Any], runner_factory: Callable[..., Any]) -> Any:
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("config must be a JSON object when provided.")
    config = TribeConfig(**config_payload)
    return runner_factory(config=config)


def serialize_prediction_result(result: PredictionResult, runner: Any) -> dict[str, Any]:
    serialized_segments = result.segments
    workflow = getattr(runner, "workflow", None)
    if workflow is not None and hasattr(workflow, "serialize_segments"):
        serialized_segments = workflow.serialize_segments(result.segments)
    return {
        "input_path": str(result.input_path),
        "model_input_path": str(result.model_input_path),
        "input_kind": result.input_kind.value,
        "events": result.events.to_dict(orient="records"),
        "brain_stimulus": result.brain_stimulus.tolist(),
        "brain_stimulus_shape": list(result.brain_stimulus.shape),
        "segments": serialized_segments,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
