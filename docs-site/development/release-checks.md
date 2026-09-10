# Release Checks

Run the full repository checks before preparing a release:

```bash
make verify
make golden-check
make visual-check
```

`make verify` covers formatting, Rust linting, Rust tests, and Python tests.

`make golden-check` verifies the committed golden artifacts used by report and
rendering tests.

`make visual-check` runs browser rendering smoke tests. It installs Playwright
browser dependencies and may require additional system packages on fresh
machines.

## Docs Site

Run the documentation build separately:

```bash
make docs-site-check
```
