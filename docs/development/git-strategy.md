# Git Initialization And Commit Strategy

Ferric Alpha keeps public source, tests, fixtures, and user-facing
documentation in git. Local process records and generated build artifacts stay
out of history.

## Repository Initialization

Use `main` as the default branch:

```bash
git init -b main
```

If the local Git version does not support `-b`, use:

```bash
git init
git branch -M main
```

The repository should be initialized after `.gitignore` exists so generated
Rust, Python, and planning files do not appear as candidates for the first
commit.

## Tracked Files

Track these categories:

- Rust crates under `crates/`.
- Python package sources under `python/`.
- Public tests under `tests/`, including golden fixtures that define API
  compatibility.
- Public project docs such as `README.md`, `THIRD_PARTY_NOTICES.md`, and
  `docs/development/`.
- Build and packaging metadata such as `Cargo.toml`, `Cargo.lock`,
  `pyproject.toml`, `Makefile`, `ruff.toml`, and `rustfmt.toml`.

Do not track these categories:

- Local process records in `.superpowers/`.
- Superpowers-generated process docs in `docs/superpowers/`.
- Rust build output in `target/`.
- Python virtual environments, caches, bytecode, wheels, and extension modules.
- Local editor, OS, coverage, and profiling output.

## First Commit

The first commit should establish a coherent open-source baseline:

- Source for all currently completed phases.
- Public docs and license.
- Tests and golden fixtures needed to reproduce the baseline.
- `.gitignore` and this strategy document.

Do not include local process notes, generated build directories, virtual
environments, or compiled Python extension modules.

Before creating the first commit, run:

```bash
make verify
make golden-check
make visual-check
```

`make golden-check` verifies the golden artifacts committed in this repository.
It uses only Ferric Alpha-owned fixtures and development dependencies.

If `make visual-check` cannot run on a machine because browser dependencies are
missing, record that fact in the commit message body and run it in CI before
tagging a release.

## Commit Granularity

Use small phase-oriented commits:

- One commit can contain tests, implementation, docs, and golden fixture updates
  for a single phase task.
- Split changes when they affect unrelated layers, such as data preparation and
  plotting.
- Keep generated fixture updates with the code change that intentionally changes
  the contract.
- Keep formatting-only churn separate from behavioral changes.

Recommended message style:

```text
phase-05: harden browser rendering smoke tests
```

For compatibility-affecting changes, include the affected public API or golden
fixture path in the commit body.

## Pre-Commit Checklist

Run the narrowest relevant tests while developing, then run the full checks
before commit:

```bash
cargo test --workspace
pytest -q tests/python
make golden-check
```

For rendering changes, also run:

```bash
make visual-check
```

Review status before staging:

```bash
git status --short --ignored
```

Expected ignored entries include `.superpowers/`, `docs/superpowers/`,
`target/`, `.venv/`, Python caches, and compiled extension modules.

## Release Tags

Use annotated tags for published versions:

```bash
git tag -a v0.1.0 -m "Ferric Alpha v0.1.0"
```

Create release tags only after source tests, golden checks, browser visual
checks, and packaging checks pass on a clean checkout.
