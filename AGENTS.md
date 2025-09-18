# Agent Guide for macposts

## Repository overview
- The core dynamic traffic assignment logic lives in the C++ sources under
  `src/` and is exposed to Python via extension modules under `macposts/_ext/`.
- Python utilities in `macposts/` provide the user facing API.  Tests live in
  `tests/` and several integration examples reside in `examples/`.

## Coding guidelines
- Prefer using the helper utilities in `macposts.backends` to work with array
  objects.  This keeps CPU/GPU support consistent across the codebase.
- Follow the existing formatting style; this project uses standard `black`
  conventions for Python and clang-format style C++ code.
- When introducing new configuration knobs, document any new environment
  variables or files in the relevant module docstrings.

## Testing requirements
- Always run `pytest` from the repository root after making changes:
  ```bash
  pytest
  ```
- When touching the C++ extension you may need to reinstall the editable build
  (`pip install -e .[dev]`) before running tests.

## GPU support notes
- GPU acceleration is optional and relies on CuPy.  Users can enable it via the
  new helpers in `macposts.backends` or with the `MACPOSTS_BACKEND` and
  `MACPOSTS_PREFER_GPU` environment variables.
- Keep the CPU code path intact—any GPU feature must gracefully fall back to the
  CPU implementation.

## Miscellaneous
- The package is distributed via `setup.cfg`/`pyproject.toml`.  Remember to add
  optional dependencies (e.g. GPU extras) there when new features require them.
- Avoid committing generated build artifacts or large data files.
