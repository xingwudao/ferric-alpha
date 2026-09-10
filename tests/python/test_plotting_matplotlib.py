from __future__ import annotations

import json
from pathlib import Path

import ferric_alpha as fa
import polars as pl
import pytest


def _report(name: str):
    cases = json.loads(Path("tests/golden/phase-05/report_cases.json").read_text())
    payload = next(case["canonical_json"] for case in cases if case["name"] == name)
    return fa.tears.TearSheetData.from_json(payload)


def test_matplotlib_backend_is_lazy_and_returns_figure() -> None:
    figure = fa.plotting.render(_report("summary"), backend="matplotlib")
    assert figure.__class__.__name__ == "Figure"
    assert len(figure.axes) > 0


def test_matplotlib_backend_draws_report_data() -> None:
    report = _report("returns")
    figure = fa.plotting.render(report, backend="matplotlib")
    assert _artist_count(figure) > 0

    mutated = _multiply_report_float_columns(report, -3.0)
    changed = fa.plotting.render(mutated, backend="matplotlib")
    assert _data_signature(figure) != _data_signature(changed)


def test_matplotlib_backend_covers_all_report_cases_without_show(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    called = False

    def fake_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(plt, "show", fake_show)
    for name in [
        "summary",
        "returns",
        "information",
        "turnover",
        "full",
        "event_returns",
        "event_study",
    ]:
        figure = fa.plotting.render(_report(name), backend="matplotlib")
        assert figure.__class__.__name__ == "Figure"
        assert len(figure.axes) > 0
    assert not called


def test_alphalens_plotting_compatibility_functions_return_axes() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    plotting = fa.plotting
    frame = pl.DataFrame(
        {
            "date": [1, 2, 3],
            "period": ["1D", "1D", "1D"],
            "factor_quantile": [1, 2, 3],
            "factor": [0.1, 0.2, 0.3],
            "mean_return": [0.01, -0.02, 0.03],
            "mean_return_difference": [0.02, 0.01, -0.01],
            "ic": [0.3, -0.1, 0.2],
            "mean_ic": [0.1, 0.2, -0.1],
            "turnover": [0.2, 0.4, 0.6],
            "autocorrelation": [0.8, 0.6, 0.7],
            "cumulative_return": [1.0, 1.1, 1.05],
            "mean_cumulative_return": [0.0, 0.1, 0.2],
            "event_count": [2, 1, 3],
        }
    )

    axes = [
        plotting.plot_ic_ts(frame),
        plotting.plot_ic_hist(frame),
        plotting.plot_ic_qq(frame),
        plotting.plot_quantile_returns_bar(frame),
        plotting.plot_quantile_returns_violin(frame),
        plotting.plot_mean_quantile_returns_spread_time_series(frame),
        plotting.plot_ic_by_group(frame),
        plotting.plot_factor_rank_auto_correlation(frame),
        plotting.plot_top_bottom_quantile_turnover(frame),
        plotting.plot_monthly_ic_heatmap(frame),
        plotting.plot_cumulative_returns(frame, "1D"),
        plotting.plot_cumulative_returns_by_quantile(frame, "1D"),
        plotting.plot_quantile_average_cumulative_return(frame),
        plotting.plot_events_distribution(frame, num_bars=3),
    ]

    assert all(axis.__class__.__name__ == "Axes" for axis in axes)
    assert all(axis.figure is not None for axis in axes)
    plt.close("all")


def test_alphalens_plotting_context_and_customize_are_lazy() -> None:
    calls = []

    @fa.plotting.customize
    def record(*, set_context=True):
        calls.append(set_context)
        return "ok"

    with fa.plotting.plotting_context(), fa.plotting.axes_style():
        assert record(set_context=False) == "ok"

    assert calls == [True]


def test_alphalens_gridfigure_and_scalarformatter_are_available() -> None:
    pytest.importorskip("matplotlib.pyplot")

    formatter = fa.plotting.ScalarFormatter()
    assert formatter.__class__.__name__ == "ScalarFormatter"

    grid = fa.tears.GridFigure(rows=2, cols=2)
    first = grid.next_cell()
    second = grid.next_row()
    assert first.__class__.__name__ == "Axes"
    assert second.__class__.__name__ == "Axes"
    grid.close()
    assert grid.fig is None


def _multiply_report_float_columns(report, factor: float):
    payload = json.loads(report.to_json())
    _multiply_float64_values(payload, factor)
    return fa.tears.TearSheetData.from_json(json.dumps(payload))


def _multiply_float64_values(value, factor: float) -> None:
    if isinstance(value, dict):
        values = value.get("float64")
        if isinstance(values, list):
            for index, item in enumerate(values):
                if isinstance(item, int | float) and not isinstance(item, bool):
                    values[index] = item * factor
        for item in value.values():
            _multiply_float64_values(item, factor)
    elif isinstance(value, list):
        for item in value:
            _multiply_float64_values(item, factor)


def _artist_count(figure) -> int:
    return sum(
        len(axis.lines) + len(axis.patches) + len(axis.collections)
        for axis in figure.axes
    )


def _data_signature(figure):
    signature = []
    for axis in figure.axes:
        for line in axis.lines:
            signature.append(("line", tuple(line.get_xdata()), tuple(line.get_ydata())))
        for patch in axis.patches:
            signature.append(
                (
                    "patch",
                    round(float(patch.get_x()), 12),
                    round(float(patch.get_y()), 12),
                    round(float(patch.get_width()), 12),
                    round(float(patch.get_height()), 12),
                )
            )
        for collection in axis.collections:
            if hasattr(collection, "get_offsets"):
                offsets = collection.get_offsets()
                signature.append(("collection", tuple(map(tuple, offsets))))
    return tuple(signature)
