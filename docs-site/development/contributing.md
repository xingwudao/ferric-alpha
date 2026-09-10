# Contributing

Prerequisites are Python 3.10 or newer, Rust 1.91 or newer, and `make`.

Create the local environment and build the Python extension:

```bash
make develop
```

Run the standard verification gate:

```bash
make verify
```

## Local Process Artifacts

Compatibility oracle workspaces and other local process folders are not part
of the public repository. They are used for maintainer checks and must remain
outside committed source.

## Public Artifacts

Public source, examples, tests, golden fixtures, README content, and this
documentation site should stay reproducible from a clean checkout.
