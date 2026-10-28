# Environment setup

## Common Python environment

Use Python 3.11 in a virtual environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The core dependencies are declared once in `pyproject.toml`; dev tools are in the `dev` extra. Optional ML runtimes are separate.

## Mac / Apple Silicon

Use native ARM Python. Verify the interpreter architecture:

```bash
python -c "import platform; print(platform.machine())"
```

It should report `arm64` (or `aarch64`). Install the optional MLX runtime with:

```bash
pip install -e ".[dev,mac]"
```

MLX is not required or imported by the core package.

## Google Colab / Linux

In a fresh Colab session, clone or upload the repository, then change to that checkout's root and install there. Replace the placeholder with the checkout path shown in Colab's Files panel; do not use a Mac-local path:

```bash
%cd <repository-root>
!python -m pip install -e ".[dev,colab]"
```

The `colab` extra provides the optional PyTorch/Transformers stack. Confirm hardware and versions in the runtime before final experiments.

## Verification

```bash
python scripts/check_environment.py
python scripts/validate_config.py configs/base.yaml
pytest
ruff check .
ruff format --check .
```
