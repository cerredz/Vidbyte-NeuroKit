## Structural Survey

The current repository is a lightweight Python SDK around Meta's `tribev2` model.

Relevant areas for this task:

- `services/inference/tribe_runner.py`
  Main orchestration class. It builds events, runs the backend, exposes convenience accessors, and saves outputs.
- `libs/config/config_loader.py`
  Loads YAML defaults from `libs/config/tribe_runner.yaml` and merges them with a `TribeConfig` override object.
- `libs/dataclasses/tribe_config.py`
  Current config object. It only carries backend/runtime defaults and does not yet carry run-time input values or environment loading behavior.
- `packages/cli/`
  Placeholder only. No CLI package or command wiring exists yet.
- `tests/test_tribe_runner.py`
  Covers the runner happy path, fail-fast path, and YAML-default loading.

Conventions already present:

- Small typed modules with explicit dataclasses and protocols
- Fail-fast validation with readable exceptions
- SDK package re-exports through `packages/sdk/src/tribe_setup`
- `pytest` as the test runner

## Task Mapping

1. Set up a CLI in `packages/cli` that is JSON-in / JSON-out and only exposes commands the repo can currently support.
   This maps to:
   - a real CLI package under `packages/cli`
   - pyproject entry-point wiring
   - request parsing / validation
   - response serialization from existing runner operations

2. Let users provide config values in a root `.env` file and allow `TribeRunner()` plus `runner.run()` with no explicit arguments.
   This maps to:
   - extending `TribeConfig` to carry environment-backed values, including a default `input_path`
   - teaching `ConfigLoader` to merge YAML defaults, root `.env`, and explicit `TribeConfig` overrides
   - updating `TribeRunner.run()` so `input_path` can be omitted when config provides it

## Assumptions and Risks

- The intended `.env` location is the repository root / current working directory root, not an arbitrary nested path.
- The simplest durable precedence order is: YAML defaults < `.env` values < explicit `TribeConfig(...)` values.
- `runner.run()` with no arguments only makes sense if `input_path` is present in config or `.env`; otherwise it should fail fast.
- JSON CLI responses may be large if they include full brain stimulus arrays, but this is acceptable for the current primitive repo since the user explicitly asked for JSON out.
