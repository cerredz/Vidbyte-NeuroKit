# CLI Package

`tribe-cli` is a small JSON-in / JSON-out wrapper around the current runner operations.

Examples:

```bash
tribe-cli run --json "{\"input_path\": \"sample.wav\"}"
```

```bash
echo "{\"input_path\": \"sample.wav\"}" | tribe-cli get-event-dataframe
```
