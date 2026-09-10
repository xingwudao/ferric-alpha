from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FERRIC_ALPHA_BROWSER_TESTS") != "1",
    reason="set FERRIC_ALPHA_BROWSER_TESTS=1 to run browser visual smoke tests",
)


def _report(name: str = "full"):
    import ferric_alpha as fa

    cases = json.loads(Path("tests/golden/phase-05/report_cases.json").read_text())
    payload = next(case["canonical_json"] for case in cases if case["name"] == name)
    return fa.tears.TearSheetData.from_json(payload)


def _png_chunks(data: bytes):
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        yield kind, payload
        offset += 12 + length


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _unfilter_png_scanlines(png: bytes) -> tuple[int, int, bytes, int]:
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    for kind, payload in _png_chunks(png):
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            idat.extend(payload)

    assert bit_depth == 8
    assert color_type in (2, 6)
    bytes_per_pixel = 4 if color_type == 6 else 3
    stride = width * bytes_per_pixel
    raw = zlib.decompress(bytes(idat))
    previous = bytearray(stride)
    rows = bytearray()
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for index, value in enumerate(row):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = (
                previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            )
            if filter_type == 1:
                row[index] = (value + left) & 0xFF
            elif filter_type == 2:
                row[index] = (value + up) & 0xFF
            elif filter_type == 3:
                row[index] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (value + _paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise AssertionError(f"unsupported PNG filter {filter_type}")
        rows.extend(row)
        previous = row
    return width, height, bytes(rows), bytes_per_pixel


def _non_background_ratio(png: bytes) -> float:
    width, height, pixels, bytes_per_pixel = _unfilter_png_scanlines(png)
    painted = 0
    for index in range(0, len(pixels), bytes_per_pixel):
        red, green, blue = pixels[index : index + 3]
        if not (red >= 245 and green >= 245 and blue >= 245):
            painted += 1
    return painted / (width * height)


def _assert_browser_layout(page, *, font_family: str | None = None) -> float:
    if font_family is not None:
        page.add_style_tag(
            content=(
                ".ferric-alpha-report, .ferric-alpha-report * "
                f"{{ font-family: {font_family} !important; }}"
            )
        )
    metrics = page.evaluate(
        """() => {
            const root = document.querySelector('.ferric-alpha-report');
            if (!root) {
                return { missingRoot: true };
            }
            const rootRect = root.getBoundingClientRect();
            const panels = [...document.querySelectorAll('.ferric-alpha-panel')]
                .map((panel) => {
                    const rect = panel.getBoundingClientRect();
                    return {
                        left: rect.left,
                        right: rect.right,
                        top: rect.top,
                        bottom: rect.bottom,
                        width: rect.width,
                        height: rect.height,
                    };
                });
            const overlaps = [];
            for (let i = 0; i < panels.length; i += 1) {
                for (let j = i + 1; j < panels.length; j += 1) {
                    const a = panels[i];
                    const b = panels[j];
                    const separated =
                        a.right <= b.left + 0.5 ||
                        b.right <= a.left + 0.5 ||
                        a.bottom <= b.top + 0.5 ||
                        b.bottom <= a.top + 0.5;
                    if (!separated) {
                        overlaps.push([i, j]);
                    }
                }
            }
            const outsidePanels = panels.filter((panel) =>
                panel.left < rootRect.left - 1 ||
                panel.right > rootRect.right + 1 ||
                panel.top < rootRect.top - 1 ||
                panel.width <= 0 ||
                panel.height <= 0
            );
            const tableOverflowLeaks = [...document.querySelectorAll(
                '.ferric-alpha-table-wrap'
            )].filter((wrap) => {
                const table = wrap.querySelector('table');
                if (!table || table.scrollWidth <= wrap.clientWidth + 1) {
                    return false;
                }
                const overflowX = window.getComputedStyle(wrap).overflowX;
                return overflowX !== 'auto' && overflowX !== 'scroll';
            });
            const textOutsideSvg = [...document.querySelectorAll('svg text')]
                .filter((text) => {
                    try {
                        const box = text.getBBox();
                        const viewBox = text.ownerSVGElement.viewBox.baseVal;
                        return (
                            box.x < -1 ||
                            box.y < -1 ||
                            box.x + box.width > viewBox.width + 1 ||
                            box.y + box.height > viewBox.height + 1
                        );
                    } catch (_) {
                        return true;
                    }
                });
            const emptySvgs = [...document.querySelectorAll('svg')].filter((svg) =>
                svg.querySelectorAll('path,line,rect,circle,text,polyline').length === 0
            );
            return {
                missingRoot: false,
                documentWidth: document.documentElement.scrollWidth,
                viewportWidth: document.documentElement.clientWidth,
                panelCount: panels.length,
                overlapCount: overlaps.length,
                outsidePanelCount: outsidePanels.length,
                tableOverflowLeakCount: tableOverflowLeaks.length,
                svgCount: document.querySelectorAll('svg').length,
                emptySvgCount: emptySvgs.length,
                textOutsideSvgCount: textOutsideSvg.length,
            };
        }"""
    )
    assert not metrics["missingRoot"]
    assert metrics["documentWidth"] <= metrics["viewportWidth"] + 1
    assert metrics["panelCount"] >= 6
    assert metrics["overlapCount"] == 0
    assert metrics["outsidePanelCount"] == 0
    assert metrics["tableOverflowLeakCount"] == 0
    assert metrics["svgCount"] >= 3
    assert metrics["emptySvgCount"] == 0
    assert metrics["textOutsideSvgCount"] == 0

    screenshot = page.screenshot(full_page=True)
    ratio = _non_background_ratio(screenshot)
    assert 0.01 < ratio < 0.98
    return ratio


def test_browser_visual_smoke_checks_layout_and_painted_pixels() -> None:
    import ferric_alpha as fa

    playwright = pytest.importorskip("playwright.sync_api")
    html = fa.plotting.render_html(_report(), width=720)
    with playwright.sync_playwright() as sync:
        browser = sync.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 752, "height": 900})
            page.set_content(html, wait_until="load")
            _assert_browser_layout(page)
        finally:
            browser.close()


def test_browser_visual_smoke_survives_wide_layout_and_font_fallback() -> None:
    import ferric_alpha as fa

    playwright = pytest.importorskip("playwright.sync_api")
    html = fa.plotting.render_html(_report(), width=1200)
    with playwright.sync_playwright() as sync:
        browser = sync.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1232, "height": 900})
            page.set_content(html, wait_until="load")
            _assert_browser_layout(page, font_family="serif")
        finally:
            browser.close()
