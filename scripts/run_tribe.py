from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Let this script import the repo modules when run as `python scripts/run_tribe.py`.
    sys.path.insert(0, str(REPO_ROOT))

VIDEOS_DIR = REPO_ROOT / "videos"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TribeRunner on a video from videos/ and save the output bundle to outputs/."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Optional video filename in videos/ or a direct path to a video file.",
    )
    parser.add_argument(
        "--include-brain-stimulus-csv",
        action="store_true",
        help="Also write brain_stimulus.csv alongside the default output files.",
    )
    return parser


def resolve_video_path(requested_video: str | None) -> Path:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if requested_video:
        requested_path = Path(requested_video).expanduser()
        candidates = [requested_path] if requested_path.is_absolute() else [
            VIDEOS_DIR / requested_path,
            REPO_ROOT / requested_path,
            Path.cwd() / requested_path,
        ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not find video '{requested_video}'. "
            f"Put it in '{VIDEOS_DIR}' or pass a valid path."
        )

    videos = sorted(
        path for path in VIDEOS_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(
            f"No video files found in '{VIDEOS_DIR}'. "
            "Add one video there, or pass a path explicitly."
        )
    if len(videos) > 1:
        video_names = ", ".join(path.name for path in videos)
        raise ValueError(
            "Found multiple videos in the videos folder. "
            f"Pass one explicitly, for example: python scripts/run_tribe.py {videos[0].name}\n"
            f"Available videos: {video_names}"
        )
    return videos[0].resolve()


def build_output_path(input_path: Path) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    base_path = OUTPUTS_DIR / input_path.stem
    if not base_path.exists():
        return base_path

    suffix = 1
    while True:
        candidate = OUTPUTS_DIR / f"{input_path.stem}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def load_runner_dependencies():
    from libs.dataclasses import TribeConfig
    from services.inference import TribeRunner

    return TribeConfig, TribeRunner


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = resolve_video_path(args.video)
        output_path = build_output_path(input_path)
        TribeConfig, TribeRunner = load_runner_dependencies()
        runner = TribeRunner(
            config=TribeConfig(
                output_dir=OUTPUTS_DIR,
                include_brain_stimulus_csv=args.include_brain_stimulus_csv,
            )
        )
        runner.run(input_path, save_to=output_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Input video: {input_path}")
    print(f"Saved output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
