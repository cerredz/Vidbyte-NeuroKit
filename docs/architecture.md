# Repository Architecture

## Layout

- `packages/sdk`: installable SDK package and shared public data models.
- `packages/cli`: reserved location for a future CLI package.
- `services/inference`: inference orchestration, including `TribeRunner`.
- `libs/config`: YAML-backed runner defaults plus config loading and validation.
- `libs/dataclasses`: shared immutable data models used across services and utils.
- `libs/enums`: shared enums such as input kinds and supported suffixes.
- `libs/protocols`: cross-layer structural typing contracts for testable seams.
- `libs/utils`: reusable utility modules, including `DataInput` and filesystem helpers.
- `docs`: project documentation.
- `scripts`: helper scripts for local development and maintenance.
