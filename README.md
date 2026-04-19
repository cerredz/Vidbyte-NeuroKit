# Tribe Setup

Small Python wrapper repo for Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) model.

This repo adds two things on top of the upstream package:

- `TribeRunner`: a clean high-level API for loading the model, building events, running inference, reading the brain stimulus, and saving artifacts.
- `DataInput`: a small utility that validates an input path, detects whether it is `audio`, `video`, or `image`, and dispatches the model call correctly.

## Important note about image support

Meta's upstream TRIBE v2 inference API is documented for `video`, `audio`, and `text`. It does not expose native standalone image inference in the repo or demo notebook.

This wrapper supports images by converting a still image into a short silent `.mp4` clip and then running the video pipeline. That keeps the behavior explicit and compatible with the upstream model.

## Install

```bash
python -m pip install -e ".[dev]"
```

Notes:

- The upstream checkpoint is loaded from Hugging Face on first use.
- Meta's demo notebook notes that you may need Hugging Face access to the Llama 3.2 dependency used by the text pipeline.
- Image conversion relies on `moviepy`, which in practice may require ffmpeg to be available in your environment.

## Root `.env` support

You can place runner defaults in a root `.env` file:

```env
TRIBE_MODEL_NAME=facebook/tribev2
TRIBE_INPUT_PATH=path/to/stimulus.mp4
TRIBE_CACHE_DIR=cache
TRIBE_OUTPUT_DIR=outputs
TRIBE_VERBOSE=true
```

Then this works:

```python
from tribe_setup import TribeRunner

runner = TribeRunner()
result = runner.run()
```

## Example

```python
from tribe_setup import TribeRunner
from tribe_setup.models import TribeConfig

runner = TribeRunner(
    config=TribeConfig(
        model_name="facebook/tribev2",
        cache_dir="cache",
        output_dir="outputs",
    ),
)

result = runner.run("path/to/stimulus.mp4")
events = runner.get_event_dataframe("path/to/stimulus.mp4")
brain_stimulus = runner.get_brain_stimulus(result)
brain_stimulus_frame = runner.get_brain_stimulus_dataframe(result)
saved_to = runner.save_output(result)
```

## CLI

The repo ships three compatible CLI entrypoints:

- `vidbyte-neuro-kit` for the task/action wrapper on top of `TribeRunner`
- `tribe-cli` for the legacy JSON-in / JSON-out runner commands
- `neurokit` for provider connection, analysis, and comparison workflows

```bash
vidbyte-neuro-kit predict response --json "{\"input_path\": \"path/to/stimulus.mp4\"}"
```

```bash
echo "{\"input_path\": \"path/to/stimulus.mp4\"}" | vidbyte-neuro-kit inspect events
```

```bash
tribe-cli run --json "{\"input_path\": \"path/to/stimulus.mp4\"}"
```

```bash
neurokit analyze vimeo --video-id 123 --token "$VIMEO_TOKEN"
```

## Output bundle

`save_output(...)` writes a directory containing:

- `brain_stimulus.npy`
- `events.csv`
- `metadata.json`
- `segments.json`

If you want a tabular brain stimulus file as well:

```python
runner.save_output(result, include_brain_stimulus_csv=True)
```

## Project layout

```text
packages/
  sdk/
    src/tribe_setup/
      __init__.py
      models.py
  cli/
services/
  inference/
    tribe_runner.py
libs/
  utils/
    data_input.py
docs/
scripts/
tests/
```
