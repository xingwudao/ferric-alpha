from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import ferric_alpha as fa
import pytest


def _report(name: str = "summary"):
    cases = json.loads(Path("tests/golden/phase-05/report_cases.json").read_text())
    payload = next(case["canonical_json"] for case in cases if case["name"] == name)
    return fa.tears.TearSheetData.from_json(payload)


def test_native_render_default_html_and_rich_repr() -> None:
    result = fa.plotting.render(_report())
    assert isinstance(result, fa.plotting.RenderedTearSheet)
    assert result.format == "html"
    assert result.width == 1200
    assert result.height > 0
    assert isinstance(result.content, str)
    assert result._repr_html_().startswith("<div")
    assert result._repr_svg_() is None
    assert result._repr_png_() is None
    assert "content" not in repr(result)


def test_native_render_svg_png_and_options() -> None:
    report = _report("returns")
    svg = fa.plotting.render(report, backend="svg", width=720, theme="dark")
    assert svg.format == "svg"
    assert svg.width == 720
    assert svg.content.startswith("<svg")
    assert svg._repr_svg_().startswith("<svg")

    png = fa.plotting.render(report, backend="png", width=720, scale=2.0)
    assert png.format == "png"
    assert png.width == 1440
    assert png.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert png._repr_png_().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_rejects_bad_options() -> None:
    report = _report()
    with pytest.raises(ValueError, match="backend"):
        fa.plotting.render(report, backend="bad")
    with pytest.raises(ValueError, match="theme"):
        fa.plotting.render(report, theme="bad")
    with pytest.raises(ValueError, match="width"):
        fa.plotting.render(report, width=True)
    with pytest.raises(ValueError, match="scale"):
        fa.plotting.render(report, scale=float("nan"))


def test_rendered_result_is_immutable_and_saves_with_strict_extension(
    tmp_path: Path,
) -> None:
    html = fa.plotting.render(_report(), backend="html")
    with pytest.raises(AttributeError):
        html.format = "svg"  # type: ignore[misc]

    path = tmp_path / "report.html"
    html.save(path)
    assert path.read_text() == html.content
    with pytest.raises(ValueError, match="extension"):
        html.save(tmp_path / "report.svg")


def test_display_uses_ipython_when_available(monkeypatch) -> None:
    report = _report()
    calls = []

    def fake_display(value):
        calls.append(value)

    ipython = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    result = fa.plotting.display(report)
    assert result._repr_html_().startswith("<div")
    assert calls == [result]
