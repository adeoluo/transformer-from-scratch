# AGENTS.md

Working notes for coding agents and contributors in **Transformer From Scratch**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A reverse-mode automatic differentiation engine and a decoder-only transformer language model, built from scratch, with every gradient verified against finite differences.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/transformer_from_scratch/__init__.py`: Transformer From Scratch - an autodiff engine and a transformer language model.
- `src/transformer_from_scratch/cli.py`: Command line interface for the transformer.
- `src/transformer_from_scratch/data.py`: The induction (copy) task: data the model can only solve by attending back.
- `src/transformer_from_scratch/evaluation.py`: Evaluation: next-token accuracy, copy-region accuracy, perplexity, baseline.
- `src/transformer_from_scratch/inspect.py`: Inspecting what the trained model attends to.
- `src/transformer_from_scratch/model.py`: A decoder-only transformer language model built on the autodiff engine.
- `src/transformer_from_scratch/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/transformer_from_scratch/research.py`: A causal study of how the copy task is solved, and what architecture it needs.
- `src/transformer_from_scratch/service.py`: A standard-library HTTP service exposing the trained transformer.
- `src/transformer_from_scratch/tensor.py`: A small reverse-mode automatic differentiation engine over dense matrices.
- `src/transformer_from_scratch/text.py`: Character-level language modeling on real English text.
- `src/transformer_from_scratch/training.py`: Training the transformer with an Adam optimizer written from scratch.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the HTTP service |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut by the maintainer with `make cut-release`, never from CI.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
