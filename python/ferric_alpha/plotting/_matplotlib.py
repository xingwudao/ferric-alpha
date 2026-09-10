from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ferric_alpha import _ferric_alpha
from ferric_alpha.tears import TearSheetData

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def render_matplotlib(
    report: TearSheetData,
    *,
    width: int,
    scale: float,
    theme: str,
) -> Figure:
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError("pip install 'ferric-alpha[plot]'") from error

    plan = json.loads(_ferric_alpha._plan_tear_sheet(report, width, scale, theme))
    panel_count = max(len(plan["panels"]), 1)
    figure = Figure(figsize=(width / 100, max(panel_count * 2.2, 2.2)), dpi=100)
    axes = figure.subplots(panel_count, 1, squeeze=False)
    for axis, panel in zip(axes.flat, plan["panels"], strict=False):
        axis.set_title(panel["title"])
        _draw_panel(axis, report, panel)
        axis.set_frame_on(True)
    figure.tight_layout()
    return figure


def _draw_panel(axis, report: TearSheetData, panel: dict) -> None:
    kind = panel["kind"]["type"]
    if kind == "table":
        _draw_table(axis, report, panel)
        return

    table = report.table(panel["table_ids"][0])
    if kind in {
        "quantile_returns_bar",
        "quantile_returns_distribution",
        "ic_histogram",
        "ic_by_group",
        "event_distribution",
    }:
        _draw_bars(axis, table, _preferred_value_column(kind))
    elif kind == "ic_qq":
        _draw_scatter(axis, table, "theoretical", "observed")
    elif kind == "ic_monthly_heatmap":
        _draw_heatmap(axis, table, "mean_ic")
    else:
        _draw_line(axis, table, _preferred_value_column(kind))


def _draw_table(axis, report: TearSheetData, panel: dict) -> None:
    axis.set_axis_off()
    table = report.table(panel["table_ids"][0])
    frame = table.head(8)
    columns = frame.columns[: min(len(frame.columns), 5)]
    rows = frame.select(columns).rows()
    axis.table(
        cellText=[[_format_cell(value) for value in row] for row in rows],
        colLabels=list(columns),
        loc="center",
        cellLoc="left",
    )


def _draw_bars(axis, table, column: str) -> None:
    values = _numeric_values(table, column)
    axis.bar(range(len(values)), values)
    axis.axhline(0.0, color="#6b7280", linewidth=0.8)


def _draw_line(axis, table, column: str) -> None:
    values = _numeric_values(table, column)
    axis.plot(range(len(values)), values, linewidth=1.4)
    axis.axhline(0.0, color="#6b7280", linewidth=0.8)


def _draw_scatter(axis, table, x_column: str, y_column: str) -> None:
    xs = _numeric_values(table, x_column)
    ys = _numeric_values(table, y_column)
    size = min(len(xs), len(ys))
    axis.scatter(xs[:size], ys[:size], s=12)
    axis.axhline(0.0, color="#6b7280", linewidth=0.8)
    axis.axvline(0.0, color="#6b7280", linewidth=0.8)


def _draw_heatmap(axis, table, column: str) -> None:
    values = _numeric_values(table, column)
    if not values:
        axis.set_xticks([])
        axis.set_yticks([])
        return
    axis.imshow([values], aspect="auto", cmap="RdYlGn", interpolation="nearest")
    axis.set_yticks([])


def _preferred_value_column(kind: str) -> str:
    return {
        "quantile_returns_bar": "mean_return",
        "quantile_returns_distribution": "mean_return",
        "cumulative_returns": "cumulative_return",
        "mean_spread": "mean_return_difference",
        "ic_time_series": "ic",
        "ic_histogram": "ic",
        "ic_monthly_heatmap": "mean_ic",
        "ic_by_group": "mean_ic",
        "quantile_turnover": "turnover",
        "rank_autocorrelation": "autocorrelation",
        "event_distribution": "event_count",
        "event_average_cumulative": "mean_cumulative_return",
    }[kind]


def _numeric_values(table, column: str) -> list[float]:
    if column not in table.columns:
        for candidate in table.columns:
            if table.schema[candidate].is_numeric():
                column = candidate
                break
    return [
        float(value)
        for value in table.get_column(column).to_list()
        if value is not None
    ]


def _format_cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
