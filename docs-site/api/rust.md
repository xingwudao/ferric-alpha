# Rust API

The Rust workspace is split by responsibility.

## Crates

- `ferric-alpha`: analytics, factor data validation, performance metrics, and
  serializable tear-sheet data.
- `ferric-alpha-render`: native report rendering from `TearSheetData`.
- `ferric-alpha-py`: PyO3 bindings for the Python package.

The analytics crate does not depend on the renderer crate. Rendering consumes
validated report data rather than raw factor inputs.

## Rustdoc

Exhaustive hosted Rustdoc is deferred until crates.io publication. Until then,
the public crate source and tests are the canonical Rust API reference.
