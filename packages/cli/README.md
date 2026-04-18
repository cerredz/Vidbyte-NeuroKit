# CLI Package

`vidbyte-neuro-kit` is a small JSON-in / JSON-out wrapper around the current runner operations.

Examples:

```bash
vidbyte-neuro-kit predict response --json "{\"input_path\": \"sample.wav\"}"
```

```bash
echo "{\"input_path\": \"sample.wav\"}" | vidbyte-neuro-kit inspect events
```
