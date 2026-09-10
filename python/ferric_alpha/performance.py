from __future__ import annotations

import re

import polars as pl

from . import _ferric_alpha
from ._validation import _u32_option


def _quantile_filter(values: list[int] | None) -> list[int] | None:
    if values is None:
        return None
    return [_u32_option(value, "quantiles") for value in values]


def _event_period(value: int, name: str) -> int:
    value = _u32_option(value, name)
    if value > 2**31 - 1:
        raise ValueError(f"invalid option {name}: event window is too large")
    return value


def _period_option(value: int | str, name: str) -> int:
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+)[dD]", value)
        if match is None:
            raise ValueError(f"invalid option {name}: expected an exact day period")
        value = int(match.group(1))
    return _u32_option(value, name)


def factor_information_coefficient(
    factor_data: pl.DataFrame,
    *,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pl.DataFrame:
    """Compute cross-sectional Spearman IC for each date and return period.

    Parameters
    ----------
    factor_data
        Clean factor data containing ``factor`` and forward-return columns.
    group_adjust
        Demean returns within each date and group before computing IC.
    by_group
        Compute separate IC observations for each group.

    Returns
    -------
    polars.DataFrame
        ``date``, optional ``group``, ``period``, and ``ic`` columns.
    """
    return _ferric_alpha.factor_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=by_group,
    )


def mean_information_coefficient(
    factor_data: pl.DataFrame,
    *,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | None = None,
) -> pl.DataFrame:
    """Average information coefficients across dates or calendar buckets.

    Parameters
    ----------
    factor_data
        Clean factor data containing ``factor`` and forward-return columns.
    group_adjust
        Demean returns within each date and group before computing IC.
    by_group
        Keep groups as separate output partitions.
    by_time
        Optional calendar bucket: ``D``, ``W``, ``M``, ``Q``, or ``Y``.

    Returns
    -------
    polars.DataFrame
        Mean IC and observation count by period and requested partitions.
    """
    return _ferric_alpha.mean_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=by_group,
        by_time=by_time,
    )


def factor_weights(
    factor_data: pl.DataFrame,
    *,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pl.DataFrame:
    """Construct normalized portfolio weights from factor values.

    Parameters
    ----------
    factor_data
        Clean factor data with ``date``, ``asset``, and ``factor`` columns.
    demeaned
        Center exposures to create a dollar-neutral long/short portfolio.
    group_adjust
        Normalize each group to equal gross exposure.
    equal_weight
        Use factor signs or median splits instead of factor magnitudes.

    Returns
    -------
    polars.DataFrame
        ``date``, ``asset``, optional ``group``, and normalized ``weight``.
    """
    return _ferric_alpha.factor_weights(
        factor_data,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
    )


def factor_returns(
    factor_data: pl.DataFrame,
    *,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    by_asset: bool = False,
) -> pl.DataFrame:
    """Compute factor-weighted forward returns.

    Parameters
    ----------
    factor_data
        Clean factor data with forward-return columns.
    demeaned, group_adjust, equal_weight
        Portfolio construction options passed to :func:`factor_weights`.
    by_asset
        Return weighted asset contributions instead of date-level totals.

    Returns
    -------
    polars.DataFrame
        Factor returns by ``date`` and ``period``, optionally by ``asset``.
    """
    return _ferric_alpha.factor_returns(
        factor_data,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
        by_asset=by_asset,
    )


def factor_alpha_beta(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame | None = None,
    *,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pl.DataFrame:
    """Estimate factor alpha and beta against a benchmark return series.

    Parameters
    ----------
    factor_data
        Clean factor data with forward-return columns.
    returns
        Optional benchmark DataFrame with ``date``, ``period``, and
        ``factor_return``. When omitted, the equal-weight universe return is
        used.
    demeaned, group_adjust, equal_weight
        Portfolio construction options used for factor returns.

    Returns
    -------
    polars.DataFrame
        ``period``, raw and annualized ``alpha``, ``beta``, and ``count``.
    """
    return _ferric_alpha.factor_alpha_beta(
        factor_data,
        returns,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
    )


def mean_return_by_quantile(
    factor_data: pl.DataFrame,
    *,
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
) -> pl.DataFrame:
    """Aggregate forward returns by factor quantile.

    Parameters
    ----------
    factor_data
        Clean factor data with ``factor_quantile`` and forward returns.
    by_date
        Keep daily observations instead of averaging across dates.
    by_group
        Keep groups as separate output partitions.
    demeaned
        Subtract the date-level universe return before aggregation.
    group_adjust
        Demean within each date and group.

    Returns
    -------
    polars.DataFrame
        Mean return, standard error, and count by quantile and period.
    """
    return _ferric_alpha.mean_return_by_quantile(
        factor_data,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
    )


def compute_mean_returns_spread(
    mean_returns: pl.DataFrame,
    upper_quant: int,
    lower_quant: int,
    std_err: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute an upper-minus-lower quantile return spread.

    Parameters
    ----------
    mean_returns
        Output from :func:`mean_return_by_quantile`.
    upper_quant, lower_quant
        Quantile labels to subtract as upper minus lower.
    std_err
        Optional separate standard-error table. By default the columns in
        ``mean_returns`` are used.

    Returns
    -------
    polars.DataFrame
        Aligned ``mean_return_difference`` and ``joint_std_error`` values.
    """
    return _ferric_alpha.compute_mean_returns_spread(
        mean_returns,
        upper_quant,
        lower_quant,
        std_err,
    )


def quantile_turnover(
    quantile_factor: pl.DataFrame,
    quantile: int,
    period: int = 1,
) -> pl.DataFrame:
    """Measure membership turnover for one factor quantile.

    Parameters
    ----------
    quantile_factor
        DataFrame with ``date``, ``asset``, and ``factor_quantile``. Extra
        columns are allowed.
    quantile
        One-indexed quantile to evaluate.
    period
        Positive observed-session lag.

    Returns
    -------
    polars.DataFrame
        Turnover by date with ``factor_quantile`` and ``period`` labels.
    """
    return _ferric_alpha.quantile_turnover(
        quantile_factor,
        _u32_option(quantile, "quantile"),
        _u32_option(period, "period"),
    )


def factor_rank_autocorrelation(
    factor_data: pl.DataFrame,
    period: int = 1,
) -> pl.DataFrame:
    """Compute lagged cross-sectional Spearman correlation of factor ranks.

    Parameters
    ----------
    factor_data
        Clean factor data with ``date``, ``asset``, and ``factor`` columns.
    period
        Positive observed-session lag.

    Returns
    -------
    polars.DataFrame
        ``date``, ``period``, and ``autocorrelation`` columns.
    """
    return _ferric_alpha.factor_rank_autocorrelation(
        factor_data, _u32_option(period, "period")
    )


def cumulative_returns(returns: pl.DataFrame) -> pl.DataFrame:
    return _ferric_alpha.cumulative_returns(returns)


def positions(
    weights: pl.DataFrame,
    period: int,
    sessions: pl.DataFrame | None = None,
    freq: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if sessions is not None and freq is not None:
        raise ValueError("sessions and freq cannot both be provided")
    sessions = sessions if sessions is not None else freq
    return _ferric_alpha.positions(weights, _period_option(period, "period"), sessions)


def factor_cumulative_returns(
    factor_data: pl.DataFrame,
    period: int | str,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: list[int] | None = None,
    groups: list[str] | None = None,
) -> pl.DataFrame:
    return _ferric_alpha.factor_cumulative_returns(
        factor_data,
        _period_option(period, "period"),
        long_short,
        group_neutral,
        equal_weight,
        _quantile_filter(quantiles),
        groups,
    )


def factor_positions(
    factor_data: pl.DataFrame,
    period: int | str,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: list[int] | None = None,
    groups: list[str] | None = None,
    sessions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    return _ferric_alpha.factor_positions(
        factor_data,
        _period_option(period, "period"),
        long_short,
        group_neutral,
        equal_weight,
        _quantile_filter(quantiles),
        groups,
        sessions,
    )


def create_pyfolio_input(
    factor_data: pl.DataFrame,
    period: int | str,
    capital: float | None = None,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: list[int] | None = None,
    groups: list[str] | None = None,
    benchmark_period: int | None = 1,
    sessions: pl.DataFrame | None = None,
) -> _ferric_alpha.PyfolioInput:
    return _ferric_alpha.create_pyfolio_input(
        factor_data,
        _period_option(period, "period"),
        capital,
        long_short,
        group_neutral,
        equal_weight,
        _quantile_filter(quantiles),
        groups,
        None
        if benchmark_period is None
        else _period_option(benchmark_period, "period"),
        sessions,
    )


def average_cumulative_return_by_quantile(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame,
    periods_before: int = 10,
    periods_after: int = 15,
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pl.DataFrame:
    return _ferric_alpha.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        _event_period(periods_before, "periods_before"),
        _event_period(periods_after, "periods_after"),
        demeaned,
        group_adjust,
        by_group,
    )


def common_start_returns(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame,
    periods_before: int,
    periods_after: int,
    cumulative: bool = False,
    mean_by_date: bool = False,
    demeaned: bool = False,
) -> pl.DataFrame:
    return _ferric_alpha.common_start_returns(
        factor_data,
        returns,
        _event_period(periods_before, "periods_before"),
        _event_period(periods_after, "periods_after"),
        cumulative,
        mean_by_date,
        demeaned,
    )


def average_cumulative_return_by_quantile_from_prices(
    factor_data: pl.DataFrame,
    returns: pl.DataFrame,
    periods_before: int = 10,
    periods_after: int = 15,
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pl.DataFrame:
    return _ferric_alpha.average_cumulative_return_by_quantile_from_prices(
        factor_data,
        returns,
        _event_period(periods_before, "periods_before"),
        _event_period(periods_after, "periods_after"),
        demeaned,
        group_adjust,
        by_group,
    )
