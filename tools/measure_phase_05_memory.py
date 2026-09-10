#!/usr/bin/env python3
from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import ferric_alpha as fa

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = json.loads(
        (ROOT / "tests/golden/phase-05/report_cases.json").read_text(encoding="utf-8")
    )
    payload = next(case["canonical_json"] for case in cases if case["name"] == "full")
    report = fa.tears.TearSheetData.from_json(payload)
    svg = fa.plotting.render_svg(report)
    html = fa.plotting.render_html(report)
    png = fa.plotting.render_png(report)
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_kb = raw_rss // 1024 if sys.platform == "darwin" else raw_rss
    if rss_kb > 512 * 1024:
        raise SystemExit(f"phase-05 memory smoke exceeded 512 MiB: {rss_kb} KiB")
    print(
        json.dumps(
            {
                "svg_bytes": len(svg.encode()),
                "html_bytes": len(html.encode()),
                "png_bytes": len(png),
                "max_rss_kb": rss_kb,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
