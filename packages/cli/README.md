# CLI Package

This package exposes:

- `vidbyte-neuro-kit` for the task/action wrapper around the current runner operations
- `tribe-cli` for the legacy JSON-in / JSON-out runner commands
- `neurokit` for provider connection and analysis workflows

Examples:

```bash
vidbyte-neuro-kit predict response --json "{\"input_path\": \"sample.wav\"}"
```

```bash
echo "{\"input_path\": \"sample.wav\"}" | vidbyte-neuro-kit inspect events
```

```bash
tribe-cli run --json "{\"input_path\": \"sample.wav\"}"
```

```bash
neurokit compare vimeo --video-id 123 --video-id 456 --metric engagement --token "$VIMEO_TOKEN"
```
