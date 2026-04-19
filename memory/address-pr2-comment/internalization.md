## Structural Survey

The current CLI implementation lives under `packages/cli/src/tribe_cli/`.

- `main.py` currently contains all CLI behavior:
  - parser construction
  - stdin / `--json` payload loading
  - command dispatch
  - runner creation
  - prediction-result serialization
- `tests/test_cli.py` currently imports the helper functions directly from `tribe_cli.main`, which tightly couples tests to the current file layout.

The merged PR comment asks for two concrete structural changes:

1. Move all CLI utility/helper functions into `packages/cli/src/tribe_cli/utils.py`.
2. Keep `main.py` minimal and only responsible for wiring the command entrypoint.
3. Replace hardcoded CLI contract strings with enums in the new `utils.py`.

This is a local CLI refactor only. It should not change the runner or config behavior, and it should preserve the JSON request/response contract and the existing CLI commands.

## Assumptions and Risks

- The request to express hardcoded strings as enums is best applied to the external CLI contract strings:
  - command names
  - JSON request keys
  - JSON response keys
- Error message text does not need to become enums; those are not stable contract keys and forcing them into enums would add noise without improving maintainability.
- `main.py` should remain import-light and delegate everything to `utils.py`.
