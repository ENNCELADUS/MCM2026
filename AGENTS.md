# Repository Guidelines

This document serves as the contributor guide for the Moon Logistics & Task Network Optimization project. It outlines the project structure, development workflow, and coding standards.

## Project Structure & Module Organization

The repository is organized as follows:

- **`src/`**: Source code root.
  - **`config/`**: Configuration modules (`constants.yaml`, `settings.py`).
  - **`model.py`**: Core mathematical model (MILP) implementation.
  - **`main.py`**: CLI entry point and pipeline orchestrator.
- **`outline/`**: Documentation and problem definition files (`model.md`, `problem.md`).
- **`paper/`**: LaTeX source files for the final report.
- **`results/`**: Output directory for optimization runs (created at runtime).

## Build, Test, and Development Commands

Ensure you have **Python 3.10+** and the `esm` Conda environment activated.

- **Run Pipeline**:
  ```bash
  python src/main.py --scenario Mix --horizon 120
  ```
  Runs the optimization pipeline with specified parameters. Use `--help` for all options.

- **Dry Run**:
  ```bash
  python src/main.py --dry-run --verbose
  ```
  Validates model construction without invoking the solver.

- **Run Tests** (Recommended):
  ```bash
  pytest
  ```
  Executes the test suite (ensure `pytest` is installed and tests are in `tests/`).

## Coding Style & Naming Conventions

- **Style**: Follow **PEP 8** guidelines.
- **Linting**: Use `ruff` for linting and formatting.
  ```bash
  ruff check src/
  ruff format src/
  ```
- **Naming**:
  - Classes: `PascalCase` (e.g., `MoonLogisticsModel`)
  - Functions/Variables: `snake_case` (e.g., `create_settings`, `delta_t`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `REQUIRED_TIME_KEYS`)
- **Type Hints**: strict usage of Python type annotations is required.

## Testing Guidelines

- **Framework**: Use `pytest`.
- **Location**: Place tests in a `tests/` directory at the project root.
- **Naming**: Test files should start with `test_` (e.g., `test_model.py`).
- **Coverage**: Aim for high coverage on core logic (`model.py`, `settings.py`).

## Commit & Pull Request Guidelines

- **Commits**: Use descriptive messages following the Conventional Commits pattern (e.g., `feat: add elevator scenario`, `fix: validate input settings`).
- **Pull Requests**:
  - Provide a clear description of changes.
  - Link relevant tasks or issues.
  - Verify that all tests pass and the code is linted before requesting review.