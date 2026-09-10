#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs/superpowers/api-compatibility.md").read_text(encoding="utf-8")
    required_readme = [
        "fa.plotting.render(report)",
        'rendered.save("factor-report.html")',
        "fa.plotting.render_svg(report)",
        "fa.plotting.render_png(report)",
        "pip install 'ferric-alpha[plot]'",
        'backend="matplotlib"',
    ]
    required_api = [
        "[render]",
        "[render_svg]",
        "[render_png]",
        "[render_html]",
        "[RenderedTearSheet]",
        "[create_full_tear_sheet]",
        "[plotting.display]",
    ]
    missing = [item for item in required_readme if item not in readme] + [
        item for item in required_api if item not in api
    ]
    if missing:
        raise SystemExit(
            "missing Phase 05 documentation entries: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
