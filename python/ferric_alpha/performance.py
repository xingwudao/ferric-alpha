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
    return _ferric_alpha.quantile_turnover(
        quantile_factor,
        _u32_option(quantile, "quantile"),
        _u32_option(period, "period"),
    )


def factor_rank_autocorrelation(
    factor_data: pl.DataFrame,
    period: int = 1,
) -> pl.DataFrame:
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
