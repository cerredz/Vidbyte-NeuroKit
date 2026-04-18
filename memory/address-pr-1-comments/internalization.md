### 1a: Structural Survey

The repository is a small Python scaffold around Meta's `tribev2` package. It is organized into four active code areas plus tests and docs:

- `packages/sdk/src/tribe_setup/`
  The public SDK package surface. Today this contains `__init__.py` and `models.py`. `__init__.py` re-exports the public API and uses lazy imports for `DataInput` and `TribeRunner`.
- `services/inference/`
  The inference orchestration layer. `tribe_runner.py` currently contains the `TribeRunner` class plus its backend protocol and all of its private helper methods.
- `libs/utils/`
  Utility layer. `data_input.py` currently contains file-type detection, image-to-video conversion, and an events-dataframe protocol.
- `tests/`
  `test_data_input.py` covers path detection and event dispatch behavior. `test_tribe_runner.py` covers happy-path runner behavior, save-output behavior, and fail-fast validation using a fake backend.
- `docs/` and `scripts/`
  Placeholder directories with minimal documentation only.

Technology stack and conventions:

- Python 3.11
- Packaging via `setuptools` in `pyproject.toml`
- Runtime dependencies: `moviepy`, `numpy`, `pandas`, and upstream `tribev2`
- Tests via `pytest`
- Typing is present throughout, but there is no configured type checker or linter
- The current code prefers small functions, immutable dataclasses, and protocol-based test seams

Current architectural inconsistencies relevant to the review:

- Enums and dataclasses currently live inside the SDK package instead of dedicated `libs` folders
- Protocols are declared inline inside operational modules rather than a dedicated protocols layer
- `TribeRunner` currently owns both orchestration and helper concerns such as backend loading, result coercion, metadata construction, segment serialization, and filesystem path resolution
- Default runner configuration lives in constructor defaults instead of an external config source

### 1b: Task Cross-Reference

The task is to implement the review comments left on PR `#1`, sequentially. The comments map onto the codebase as follows:

1. Move hardcoded suffixes into `libs/enums/suffixes.py` and represent them as enums.
   Affects `libs/utils/data_input.py` and requires new files under `libs/enums/`.
2. Move inline protocols into `libs/protocols/`.
   Affects `libs/utils/data_input.py`, `services/inference/tribe_runner.py`, and requires new files under `libs/protocols/`.
3. Move SDK enum and dataclasses into `libs/enums/` and `libs/dataclasses/`.
   Affects `packages/sdk/src/tribe_setup/models.py`, all imports that currently consume `InputKind`, `PreparedInput`, and `PredictionResult`, and requires new files under `libs/dataclasses/`.
4. Replace constructor defaults in `TribeRunner` with YAML-backed config loading.
   Affects `services/inference/tribe_runner.py`, requires a new `libs/config/` package, a config YAML file, and a configuration dataclass.
5. Add a dedicated filesystem helper class in `libs/utils/`.
   Affects `services/inference/tribe_runner.py` and requires a new utility module.
6. Remove private helper behavior from `TribeRunner` into a dedicated inference support class.
   Affects `services/inference/tribe_runner.py` and requires a new support module in `services/inference/`.
7. Update function signature formatting and add line comments in `services/inference/tribe_runner.py`.
   Affects `services/inference/tribe_runner.py`.
8. Update tests and docs to reflect the refactor.
   Affects `tests/`, `README.md`, and potentially `docs/architecture.md`.

Behavior that must be preserved:

- `DataInput` must still detect `audio`, `video`, and `image` paths and adapt images into a video clip
- `TribeRunner.run()` must still return a structured prediction result
- `get_event_dataframe`, `get_brain_stimulus`, `get_brain_stimulus_dataframe`, and `save_output` must still work
- Existing tests around fail-fast validation and saved output bundles should still pass after the refactor

Blast radius:

- Import paths across the repo
- Public SDK exports
- README examples
- Test fixtures and fake backend compatibility

### 1c: Assumption & Risk Inventory

Assumptions:

- The review comments are authoritative and should all be implemented in the current PR branch rather than split into separate PRs.
- It is acceptable to keep `packages/sdk/src/tribe_setup/models.py` as a thin compatibility re-export layer even after moving the concrete enum/dataclass implementations to `libs/`.
- A YAML-backed config system can merge file defaults with a partially specified config dataclass instance.
- Adding `PyYAML` as a direct dependency is appropriate because the repo will now parse YAML outside upstream `tribev2`.

Risks:

- Moving enums/dataclasses/protocols can create import cycles if the SDK package continues to re-export service and util objects eagerly.
- Refactoring `TribeRunner` to remove private methods can accidentally change runtime behavior or the shape of saved artifacts.
- Converting suffix constants into enums can make membership checks clumsy if not wrapped with clear helper APIs.
- YAML config loading introduces a new failure mode around invalid config files or missing keys; validation must fail fast with readable errors.
- The review request to make signatures one line can reduce readability if applied indiscriminately; the implementation should comply in the affected file without making the code obscure.

Phase 1 complete
