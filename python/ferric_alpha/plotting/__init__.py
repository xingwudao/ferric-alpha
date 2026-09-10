from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Literal, overload

import polars as pl

from ferric_alpha import _ferric_alpha, utils
from ferric_alpha.tears import TearSheetData

from ._rendered import RenderedTearSheet

NativeBackend = Literal["html", "svg", "png"]
Backend = Literal["html", "svg", "png", "matplotlib"]
Theme = Literal["light", "dark"]
DECIMAL_TO_BPS = 10_000


def _validate_width(width: int) -> int:
    if isinstance(width, bool) or not isinstance(width, int):
        raise ValueError("invalid option width")
    return width


def _validate_scale(scale: float) -> float:
    if isinstance(scale, bool) or not isinstance(scale, int | float):
        raise ValueError("invalid option scale")
    value = float(scale)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("invalid option scale")
    return value


def _validate_theme(theme: str) -> str:
    if theme not in ("light", "dark"):
        raise ValueError("invalid option theme")
    return theme


@overload
def render(
    report: TearSheetData,
    *,
    backend: Literal["html"] = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> RenderedTearSheet: ...


@overload
def render(
    report: TearSheetData,
    *,
    backend: Literal["svg", "png"],
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> RenderedTearSheet: ...


def render(
    report: TearSheetData,
    *,
    backend: Backend = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> RenderedTearSheet:
    """Render a tear-sheet data model with a native or Matplotlib backend.

    Parameters
    ----------
    report
        Report model produced by a ``ferric_alpha.tears`` data builder.
    backend
        Output backend: ``html``, ``svg``, ``png``, or ``matplotlib``.
    width
        Logical output width in pixels.
    scale
        Positive output scale factor, primarily useful for raster images.
    theme
        ``light`` or ``dark`` color theme.

    Returns
    -------
    RenderedTearSheet or matplotlib.figure.Figure
        Immutable native output for HTML, SVG, or PNG; a Matplotlib figure
        when that optional backend is selected.
    """
    width = _validate_width(width)
    scale = _validate_scale(scale)
    theme = _validate_theme(theme)
    if backend == "html":
        payload, logical_width, logical_height = _ferric_alpha._render_tear_sheet_html(
            report, width, scale, theme
        )
        return RenderedTearSheet(
            format="html",
            payload=payload,
            width=logical_width,
            height=logical_height,
        )
    if backend == "svg":
        payload, logical_width, logical_height = _ferric_alpha._render_tear_sheet_svg(
            report, width, scale, theme
        )
        return RenderedTearSheet(
            format="svg",
            payload=payload,
            width=logical_width,
            height=logical_height,
        )
    if backend == "png":
        payload, physical_width, physical_height = _ferric_alpha._render_tear_sheet_png(
            report, width, scale, theme
        )
        return RenderedTearSheet(
            format="png",
            payload=payload,
            width=physical_width,
            height=physical_height,
        )
    if backend == "matplotlib":
        from ._matplotlib import render_matplotlib

        return render_matplotlib(report, width=width, scale=scale, theme=theme)
    raise ValueError("invalid option backend")


def render_svg(
    report: TearSheetData,
    *,
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> str:
    return str(
        render(report, backend="svg", width=width, scale=scale, theme=theme).content
    )


def render_png(
    report: TearSheetData,
    *,
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> bytes:
    return bytes(
        render(report, backend="png", width=width, scale=scale, theme=theme).content
    )


def render_html(
    report: TearSheetData,
    *,
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> str:
    return str(
        render(report, backend="html", width=width, scale=scale, theme=theme).content
    )


def display(
    report: TearSheetData,
    *,
    width: int = 1200,
    scale: float = 1.0,
    theme: Theme = "light",
) -> RenderedTearSheet:
    result = render(report, backend="html", width=width, scale=scale, theme=theme)
    try:
        from IPython.display import display as ipython_display
    except ImportError:
        return result
    ipython_display(result)
    return result


def customize(func):
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop("set_context", True)
        if set_context:
            with plotting_context(), axes_style():
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return call_w_context


@contextmanager
def plotting_context(context: str = "notebook", font_scale: float = 1.5, rc=None):
    del context
    rc = {"lines.linewidth": 1.5, **(rc or {})}
    with _pyplot().rc_context(rc):
        yield


@contextmanager
def axes_style(style: str = "darkgrid", rc=None):
    del style
    with _pyplot().rc_context(rc or {}):
        yield


def plot_returns_table(alpha_beta, mean_ret_quantile, mean_ret_spread_quantile):
    utils.print_table(_as_frame(alpha_beta), name="Returns Analysis")
    utils.print_table(_as_frame(mean_ret_quantile))
    utils.print_table(_as_frame(mean_ret_spread_quantile))


def plot_turnover_table(autocorrelation_data, quantile_turnover):
    utils.print_table(_as_frame(autocorrelation_data), name="Turnover Analysis")
    utils.print_table(_as_frame(quantile_turnover))


def plot_information_table(ic_data):
    utils.print_table(_as_frame(ic_data), name="Information Analysis")


def plot_quantile_statistics_table(factor_data):
    utils.print_table(_as_frame(factor_data), name="Quantiles Statistics")


def plot_ic_ts(ic, ax=None):
    return _line_plot(ic, "ic", ax, title="Information Coefficient")


def plot_ic_hist(ic, ax=None):
    axis = _axis(ax)
    values = _numeric_values(_as_frame(ic), "ic")
    axis.hist(values, bins=min(max(len(values), 1), 20))
    axis.set(title="Period IC", xlabel="IC")
    return axis


def plot_ic_qq(ic, theoretical_dist=None, ax=None):
    del theoretical_dist
    axis = _axis(ax)
    values = sorted(_numeric_values(_as_frame(ic), "ic"))
    if values:
        n = len(values)
        theoretical = [(index + 0.5) / n for index in range(n)]
        axis.scatter(theoretical, values, s=14)
    axis.set(title="IC Q-Q", xlabel="Theoretical Quantile", ylabel="Observed Quantile")
    return axis


def plot_quantile_returns_bar(
    mean_ret_by_q, by_group=False, ylim_percentiles=None, ax=None
):
    del by_group, ylim_percentiles
    return _bar_plot(mean_ret_by_q, "mean_return", ax, scale=DECIMAL_TO_BPS)


def plot_quantile_returns_violin(return_by_q, ylim_percentiles=None, ax=None):
    del ylim_percentiles
    axis = _axis(ax)
    values = _numeric_values(_as_frame(return_by_q), "mean_return")
    axis.boxplot(values or [0.0])
    axis.set(title="Period Wise Return By Factor Quantile", ylabel="Return (bps)")
    return axis


def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread,
    std_err=None,
    bandwidth=1,
    ax=None,
):
    del std_err, bandwidth
    return _line_plot(
        mean_returns_spread,
        "mean_return_difference",
        ax,
        title="Top Minus Bottom Quantile Mean Return",
        scale=DECIMAL_TO_BPS,
    )


def plot_ic_by_group(ic_group, ax=None):
    return _bar_plot(ic_group, "mean_ic", ax, title="Information Coefficient By Group")


def plot_factor_rank_auto_correlation(factor_autocorrelation, period=1, ax=None):
    return _line_plot(
        factor_autocorrelation,
        "autocorrelation",
        ax,
        title=f"{period}D Period Factor Rank Autocorrelation",
    )


def plot_top_bottom_quantile_turnover(quantile_turnover, period=1, ax=None):
    return _line_plot(
        quantile_turnover,
        "turnover",
        ax,
        title=f"{period}D Period Top and Bottom Quantile Turnover",
    )


def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    axis = _axis(ax)
    values = _numeric_values(_as_frame(mean_monthly_ic), "mean_ic")
    axis.imshow([values or [0.0]], aspect="auto", cmap="RdYlGn")
    axis.set(title="Monthly Mean IC")
    return axis


def plot_cumulative_returns(factor_returns, period, freq=None, title=None, ax=None):
    del freq
    return _line_plot(
        factor_returns,
        "cumulative_return",
        ax,
        title=title or f"Portfolio Cumulative Return ({period} Fwd Period)",
    )


def plot_cumulative_returns_by_quantile(quantile_returns, period, freq=None, ax=None):
    del freq
    return _line_plot(
        quantile_returns,
        "cumulative_return",
        ax,
        title=f"Cumulative Return by Quantile ({period} Period Forward Return)",
    )


def plot_quantile_average_cumulative_return(
    avg_cumulative_returns,
    by_quantile=False,
    std_bar=False,
    title=None,
    ax=None,
):
    del by_quantile, std_bar
    return _line_plot(
        avg_cumulative_returns,
        "mean_cumulative_return",
        ax,
        title=title or "Average Cumulative Returns by Quantile",
        scale=DECIMAL_TO_BPS,
    )


def plot_events_distribution(events, num_bars=50, ax=None):
    del num_bars
    return _bar_plot(events, "event_count", ax, title="Distribution of events in time")


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("pip install 'ferric-alpha[plot]'") from error

    return plt


def _axis(ax=None):
    if ax is not None:
        return ax
    _, axis = _pyplot().subplots(1, 1, figsize=(8, 4))
    return axis


def _as_frame(data) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.Series):
        return data.to_frame()
    if hasattr(data, "to_dict"):
        return pl.DataFrame(data.to_dict())
    return pl.DataFrame({"value": list(data)})


def _line_plot(data, preferred: str, ax=None, title: str | None = None, scale=1.0):
    axis = _axis(ax)
    values = [value * scale for value in _numeric_values(_as_frame(data), preferred)]
    axis.plot(range(len(values)), values)
    axis.axhline(0.0, color="#6b7280", linewidth=0.8)
    if title is not None:
        axis.set_title(title)
    return axis


def _bar_plot(data, preferred: str, ax=None, title: str | None = None, scale=1.0):
    axis = _axis(ax)
    values = [value * scale for value in _numeric_values(_as_frame(data), preferred)]
    axis.bar(range(len(values)), values)
    axis.axhline(0.0, color="#6b7280", linewidth=0.8)
    if title is not None:
        axis.set_title(title)
    return axis


def _numeric_values(table: pl.DataFrame, preferred: str) -> list[float]:
    column = preferred if preferred in table.columns else _first_numeric_column(table)
    if column is None:
        return []
    return [
        float(value)
        for value in table.get_column(column).to_list()
        if value is not None
    ]


def _first_numeric_column(table: pl.DataFrame) -> str | None:
    for column, dtype in table.schema.items():
        if dtype.is_numeric():
            return column
    return None


def __getattr__(name: str):
    if name == "ScalarFormatter":
        try:
            from matplotlib.ticker import ScalarFormatter
        except ImportError as error:
            raise ImportError("pip install 'ferric-alpha[plot]'") from error

        return ScalarFormatter
    raise AttributeError(f"module 'ferric_alpha.plotting' has no attribute {name!r}")


__all__ = [
    "DECIMAL_TO_BPS",
    "RenderedTearSheet",
    "ScalarFormatter",
    "axes_style",
    "customize",
    "display",
    "plot_cumulative_returns",
    "plot_cumulative_returns_by_quantile",
    "plot_events_distribution",
    "plot_factor_rank_auto_correlation",
    "plot_ic_by_group",
    "plot_ic_hist",
    "plot_ic_qq",
    "plot_ic_ts",
    "plot_information_table",
    "plot_mean_quantile_returns_spread_time_series",
    "plot_monthly_ic_heatmap",
    "plot_quantile_average_cumulative_return",
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_quantile_statistics_table",
    "plot_returns_table",
    "plot_top_bottom_quantile_turnover",
    "plot_turnover_table",
    "plotting_context",
    "render",
    "render_html",
    "render_png",
    "render_svg",
]
