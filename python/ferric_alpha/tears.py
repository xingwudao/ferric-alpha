from __future__ import annotations

import polars as pl

from . import _ferric_alpha, plotting
from ._validation import _u32_option, _u32_options

TearSheetData = _ferric_alpha.TearSheetData


class GridFigure:
    def __init__(self, rows: int, cols: int):
        try:
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("pip install 'ferric-alpha[plot]'") from error

        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        axis = self.fig.add_subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return axis

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        axis = self.fig.add_subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return axis

    def close(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.gs = None


def _event_period(value: int, name: str) -> int:
    value = _u32_option(value, name)
    if value > 2**31 - 1:
        raise ValueError(f"invalid option {name}: event window is too large")
    return value


def _avgretplot(value: tuple[int, int] | None, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, tuple | list)
        or len(value) != 2
    ):
        raise ValueError(f"invalid option {name}: expected a two-value window")
    return (_event_period(value[0], name), _event_period(value[1], name))


def create_summary_tear_sheet_data(
    factor_data: pl.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    turnover_periods: list[int] | tuple[int, ...] | None = None,
) -> TearSheetData:
    return _ferric_alpha.create_summary_tear_sheet_data(
        factor_data,
        long_short,
        group_neutral,
        _u32_options(turnover_periods, "turnover_periods"),
    )


def create_summary_tear_sheet(
    factor_data: pl.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    turnover_periods: list[int] | tuple[int, ...] | None = None,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_summary_tear_sheet_data(
        factor_data, long_short, group_neutral, turnover_periods
    )
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def create_returns_tear_sheet_data(
    factor_data: pl.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
) -> TearSheetData:
    return _ferric_alpha.create_returns_tear_sheet_data(
        factor_data,
        long_short,
        group_neutral,
        by_group,
    )


def create_returns_tear_sheet(
    factor_data: pl.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_returns_tear_sheet_data(
        factor_data, long_short, group_neutral, by_group
    )
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def create_information_tear_sheet_data(
    factor_data: pl.DataFrame,
    group_neutral: bool = False,
    by_group: bool = False,
    rolling_window: int = 22,
) -> TearSheetData:
    return _ferric_alpha.create_information_tear_sheet_data(
        factor_data,
        group_neutral,
        by_group,
        _u32_option(rolling_window, "rolling_window"),
    )


def create_information_tear_sheet(
    factor_data: pl.DataFrame,
    group_neutral: bool = False,
    by_group: bool = False,
    rolling_window: int = 22,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_information_tear_sheet_data(
        factor_data, group_neutral, by_group, rolling_window
    )
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def create_turnover_tear_sheet_data(
    factor_data: pl.DataFrame,
    turnover_periods: list[int] | tuple[int, ...] | None = None,
) -> TearSheetData:
    return _ferric_alpha.create_turnover_tear_sheet_data(
        factor_data,
        _u32_options(turnover_periods, "turnover_periods"),
    )


def create_turnover_tear_sheet(
    factor_data: pl.DataFrame,
    turnover_periods: list[int] | tuple[int, ...] | None = None,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_turnover_tear_sheet_data(factor_data, turnover_periods)
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def create_full_tear_sheet_data(
    factor_data: pl.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    turnover_periods: list[int] | tuple[int, ...] | None = None,
    ic_rolling_window: int = 22,
) -> TearSheetData:
    """Build the complete serializable factor-analysis report model.

    Parameters
    ----------
    factor_data
        Clean factor data containing factor values, quantiles, groups when
        available, and forward-return columns.
    long_short
        Use demeaned factor weights for long/short return analysis.
    group_neutral
        Demean returns and normalize weights within groups.
    by_group
        Include group-level return and information-coefficient panels.
    turnover_periods
        Optional observed-session lags for turnover panels. By default the
        available forward-return periods are used.
    ic_rolling_window
        Positive number of observations in the rolling IC panel.

    Returns
    -------
    TearSheetData
        Serializable tables and panel metadata accepted by
        :func:`ferric_alpha.plotting.render`.
    """
    return _ferric_alpha.create_full_tear_sheet_data(
        factor_data,
        long_short,
        group_neutral,
        by_group,
        _u32_options(turnover_periods, "turnover_periods"),
        _u32_option(ic_rolling_window, "ic_rolling_window"),
    )


def create_full_tear_sheet(
    factor_data: pl.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    turnover_periods: list[int] | tuple[int, ...] | None = None,
    ic_rolling_window: int = 22,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_full_tear_sheet_data(
        factor_data,
        long_short,
        group_neutral,
        by_group,
        turnover_periods,
        ic_rolling_window,
    )
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def create_event_returns_tear_sheet_data(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame,
    avgretplot: tuple[int, int] = (5, 15),
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
) -> TearSheetData:
    window = _avgretplot(avgretplot, "avgretplot")
    if window is None:
        raise ValueError("invalid option avgretplot: expected a two-value window")
    periods_before, periods_after = window
    returns = _returns_from_prices_if_needed(returns)
    return _ferric_alpha.create_event_returns_tear_sheet_data(
        factor_data,
        returns,
        periods_before,
        periods_after,
        long_short,
        group_neutral,
        by_group,
    )


def create_event_returns_tear_sheet(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame,
    avgretplot: tuple[int, int] = (5, 15),
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_event_returns_tear_sheet_data(
        factor_data, returns, avgretplot, long_short, group_neutral, by_group
    )
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def create_event_study_tear_sheet_data(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame | None = None,
    avgretplot: tuple[int, int] = (5, 15),
    rate_of_ret: bool = True,
    n_bars: int = 50,
) -> TearSheetData:
    window = _avgretplot(avgretplot, "avgretplot")
    returns = None if returns is None else _returns_from_prices_if_needed(returns)
    return _ferric_alpha.create_event_study_tear_sheet_data(
        factor_data,
        returns,
        None if window is None else list(window),
        rate_of_ret,
        _u32_option(n_bars, "n_bars"),
    )


def create_event_study_tear_sheet(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame | None = None,
    avgretplot: tuple[int, int] = (5, 15),
    rate_of_ret: bool = True,
    n_bars: int = 50,
    backend: str = "html",
    width: int = 1200,
    scale: float = 1.0,
    theme: str = "light",
):
    report = create_event_study_tear_sheet_data(
        factor_data, returns, avgretplot, rate_of_ret, n_bars
    )
    return plotting.render(
        report, backend=backend, width=width, scale=scale, theme=theme
    )


def _returns_from_prices_if_needed(frame: pl.DataFrame) -> pl.DataFrame:
    if "return" in frame.columns:
        return frame
    if "price" not in frame.columns:
        return frame
    return (
        frame.sort("asset", "date")
        .with_columns(
            ((pl.col("price") / pl.col("price").shift(1).over("asset")) - 1.0).alias(
                "return"
            )
        )
        .filter(pl.col("return").is_not_null())
        .select("date", "asset", "return")
        .sort("date", "asset")
    )


__all__ = [
    "GridFigure",
    "TearSheetData",
    "create_event_returns_tear_sheet",
    "create_event_returns_tear_sheet_data",
    "create_event_study_tear_sheet",
    "create_event_study_tear_sheet_data",
    "create_full_tear_sheet",
    "create_full_tear_sheet_data",
    "create_information_tear_sheet",
    "create_information_tear_sheet_data",
    "create_returns_tear_sheet",
    "create_returns_tear_sheet_data",
    "create_summary_tear_sheet",
    "create_summary_tear_sheet_data",
    "create_turnover_tear_sheet",
    "create_turnover_tear_sheet_data",
]
