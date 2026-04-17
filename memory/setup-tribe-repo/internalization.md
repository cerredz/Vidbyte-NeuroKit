### 1a: Structural Survey

The local workspace started empty. There was no existing Python package, test suite, configuration, or Git metadata in `C:\Users\Michael Cerreto\tribe`.

Because this is a greenfield scaffold, the relevant system to internalize is the upstream `facebookresearch/tribev2` project and its demo notebook:

- Upstream package entry point: `tribev2.TribeModel`
- Main inference helpers: `tribev2/demo_utils.py`
- Packaging and runtime dependencies: `pyproject.toml`
- Usage patterns and expected outputs: `README.md` and `tribe_demo.ipynb`

Observed upstream conventions and behavior:

- `TribeModel.from_pretrained(...)` loads a local checkpoint directory or a Hugging Face repo id.
- `TribeModel.get_events_dataframe(...)` natively accepts exactly one of `text_path`, `audio_path`, or `video_path`.
- `TribeModel.predict(events=...)` returns `(preds, segments)`, where `preds` is a NumPy array of shape `(n_timesteps, n_vertices)`.
- Upstream code already validates file existence and allowed suffixes for audio/video/text.
- The demo notebook documents the model as multimodal for video, audio, and text. It does not expose standalone image inference.

Design implications for this repo:

- This repo should be a thin wrapper around Meta's package, not a fork.
- The wrapper should keep the surface area small and explicit: input detection, fail-fast validation, inference orchestration, and artifact saving.
- Tests should avoid downloading the upstream checkpoint and should instead verify behavior through a fake backend.

### 1b: Task Cross-Reference

User request to codebase mapping:

- "Create a very simply setup repo":
  This requires a new Python package layout, package metadata, docs, and tests from scratch.

- "Create a `TribeRunner` class":
  This maps to a top-level runtime orchestration class that owns model loading, event generation, prediction, brain stimulus access, and artifact persistence.

- "There should be a `DataInput` class in the utils folder":
  This maps to `src/tribe_setup/utils/data_input.py` and should be responsible for modality detection and path validation before inference.

- "Path inputted is a video, image, or audio and then runs the model depending on what it is":
  Audio and video map directly to upstream `TribeModel.get_events_dataframe(audio_path=...)` and `video_path=...`.
  Image is net-new because upstream does not support it directly; the wrapper must adapt an image into a model-compatible input.

- "`TribeRunner` should validate the path inputs and fail fast":
  Validation should happen before any expensive model load or inference call.

- "`TribeRunner` should also have functionalities like getting the event dataframe, outputting the brain stimulus, saving the output to some path":
  This maps to explicit runner methods for `get_event_dataframe`, `run`, `get_brain_stimulus`, `get_brain_stimulus_dataframe`, and `save_output`.

- "Each function should do one thing":
  The implementation should separate concerns into small helpers: path resolution, modality detection, input preparation, model loading, result coercion, and serialization.

Relevant files to create:

- `pyproject.toml`
- `README.md`
- `.gitignore`
- `src/tribe_setup/__init__.py`
- `src/tribe_setup/models.py`
- `src/tribe_setup/tribe_runner.py`
- `src/tribe_setup/utils/__init__.py`
- `src/tribe_setup/utils/data_input.py`
- `tests/test_data_input.py`
- `tests/test_tribe_runner.py`

### 1c: Assumption & Risk Inventory

Assumptions I am making:

- The repo should wrap the published upstream `tribev2` package through a dependency instead of vendoring Meta's source code.
- "Image" support should be implemented by converting a still image into a short silent video clip, because the upstream model is documented for video/audio/text rather than images.
- Saving output to a path is best represented as saving a result bundle directory containing metadata, events, and brain-stimulus arrays.
- The user values a testable scaffold over a notebook-only example, so I will include `pytest` coverage for the wrapper API.

Risks and mitigations:

- Upstream installation can be heavy and platform-sensitive. Mitigation: keep imports lazy and make tests backend-injected so the local test suite does not require the actual model.
- Image-to-video conversion may require `moviepy` and an ffmpeg-capable environment. Mitigation: surface a clear runtime error and document the requirement in the README.
- The exact upstream response object in `segments` may vary. Mitigation: save a conservative serialized summary instead of assuming a rigid schema.

Phase 1 complete
