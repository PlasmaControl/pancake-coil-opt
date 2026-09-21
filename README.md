# Pancake Coil Optimization

GPU-accelerated, differentiable shape optimization of high-temperature superconducting pancake-coil magnets, built on JAX. Axisymmetric Biot-Savart fields, a self-consistent operating current from the local critical-current surface, and constrained multi-objective optimization.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/). The installation guide can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

    uv is a Python project manager. It reads the project definition in [`pyproject.toml`](https://github.com/PlasmaControl/pancakecoilopt/blob/main/pyproject.toml) and takes care of the rest. You don't need to install Python yourself — uv will install a compatible version when you follow the steps below.

2. Clone the repository.
    ```
    git clone https://github.com/PlasmaControl/pancakecoilopt.git
    ```
3. Go to the `pancakecoilopt` directory.
    ```
    cd pancakecoilopt
    ```
4. Create the virtual environment and install all dependencies (including the dev tools).
    ```bash
    uv sync
    ```
    uv creates the environment in `.venv`. Run any command inside it with `uv run`, e.g. `uv run python run.py`.

5. Finally, select the environment in your integrated development environment (IDE). In Visual Studio Code:

    - Press `Ctrl+Shift+P`.
    - Type `Python: Select Interpreter`.
    - Select the interpreter at `.venv/bin/python`.

### Common tasks

Development tasks are defined as [poe](https://poethepoet.natn.io/) tasks and run through uv:

| Command | What it does |
| --- | --- |
| `uv run python run.py` | Run an optimization (see `run.py`) |
| `uv run poe test` | Run the test suite |
| `uv run poe format` | Format with black + ruff |
| `uv run poe lint` | Lint with ruff |
| `uv run poe check-types` | Type-check with pyright |
| `uv run poe test-and-report` | Run tests with a coverage report |
| `uv run poe profile-run-file` | Profile `run.py` and open snakeviz |
