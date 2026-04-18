from __future__ import annotations

from tribe_cli.utils import run_cli


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
