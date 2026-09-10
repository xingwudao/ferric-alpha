import math
import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta

import polars as pl

from ._ferric_alpha import (
    CleanFactorResult,
    LossReport,
)
from ._ferric_alpha import (
    compute_forward_returns as _compute_forward_returns,
)
from ._ferric_alpha import (
    get_clean_factor as _get_clean_factor,
)
from ._ferric_alpha import (
    get_clean_factor_and_forward_returns as _get_clean_factor_and_forward_returns,
)
from ._ferric_alpha import (
    quantize_factor as _quantize_factor,
)
from ._ferric_alpha import (
    validate_factor_frame as _validate_factor_frame,
)


class NonMatchingTimezoneError(ValueError):
    """Raised when factor and price date columns use different timezones."""


class MaxLossExceededError(ValueError):
    """Raised when clean-factor row loss exceeds the configured threshold."""


def rethrow(exception: Exception, additional_message: str):
    if not exception.args:
        exception.args = (additional_message,)
    else:
        exception.args = (
            str(exception.args[0]) + additional_message,
            *exception.args[1:],
        )
    raise exception


def non_unique_bin_edges_error(func):
    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as error:
            if "Bin edges must be unique" in str(error):
                rethrow(error, _NON_UNIQUE_BIN_EDGES_MESSAGE)
            raise

    return dec


def validate_factor_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return _validate_factor_frame(frame)


def compute_forward_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    periods: tuple[int, ...] = (1, 5, 10),
    filter_zscore: float | None = None,
    cumulative_returns: bool = True,
) -> pl.DataFrame:
    try:
        return _compute_forward_returns(
            factor, prices, list(periods), filter_zscore, cumulative_returns
        )
    except ValueError as error:
        raise _compat_error(error) from error


def quantize_factor(
    factor: pl.DataFrame,
    *,
    quantiles: int | tuple[float, ...] | None = 5,
    bins: int | tuple[float, ...] | None = None,
    by_group: bool = False,
    zero_aware: bool = False,
) -> pl.DataFrame:
    return _quantize_factor(
        factor,
        quantiles,
        _normalize_bins(bins),
        by_group,
        zero_aware,
    )


def get_clean_factor(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int | tuple[float, ...] | None = 5,
    bins: int | tuple[float, ...] | None = None,
    groupby: Mapping[str, object] | None = None,
    groupby_labels: Mapping[object, object] | None = None,
    max_loss: float = 0.35,
    by_group: bool = False,
    binning_by_group: bool = False,
    zero_aware: bool = False,
) -> CleanFactorResult:
    factor = _attach_groupby(factor, groupby, groupby_labels)
    by_group = by_group or binning_by_group
    try:
        return _get_clean_factor(
            factor,
            forward_returns,
            quantiles,
            _normalize_bins(bins),
            max_loss,
            by_group,
            zero_aware,
        )
    except ValueError as error:
        raise _compat_error(error) from error


def get_clean_factor_and_forward_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    periods: tuple[int, ...] = (1, 5, 10),
    quantiles: int | tuple[float, ...] | None = 5,
    bins: int | tuple[float, ...] | None = None,
    groupby: Mapping[str, object] | None = None,
    groupby_labels: Mapping[object, object] | None = None,
    max_loss: float = 0.35,
    by_group: bool = False,
    binning_by_group: bool = False,
    zero_aware: bool = False,
    cumulative_returns: bool = True,
    filter_zscore: float | None = 20.0,
) -> CleanFactorResult:
    factor = _attach_groupby(factor, groupby, groupby_labels)
    by_group = by_group or binning_by_group
    try:
        return _get_clean_factor_and_forward_returns(
            factor,
            prices,
            list(periods),
            quantiles,
            _normalize_bins(bins),
            max_loss,
            by_group,
            zero_aware,
            cumulative_returns,
            filter_zscore,
        )
    except ValueError as error:
        raise _compat_error(error) from error


def get_forward_returns_columns(
    columns: list[str] | tuple[str, ...],
    require_exact_day_multiple: bool = False,
) -> list[str]:
    return [
        column
        for column in columns
        if _forward_return_duration(column, require_exact_day_multiple) is not None
    ]


def print_table(
    table: pl.DataFrame | pl.Series, name: str | None = None, fmt=None
) -> None:
    if isinstance(table, pl.Series):
        table = table.to_frame()
    if not isinstance(table, pl.DataFrame):
        raise TypeError("table must be a polars DataFrame or Series")

    if name:
        print(name)
    print(_format_table(table, fmt))


def rate_of_return(period_ret: pl.Series, base_period: str) -> pl.Series:
    conversion_factor = _duration_microseconds(base_period) / _series_period(period_ret)
    return (period_ret + 1.0).pow(conversion_factor) - 1.0


def std_conversion(period_std: pl.Series, base_period: str) -> pl.Series:
    conversion_factor = _series_period(period_std) / _duration_microseconds(base_period)
    return period_std / math.sqrt(conversion_factor)


def demean_forward_returns(
    factor_data: pl.DataFrame,
    grouper: str | tuple[str, ...] | list[str] | None = None,
) -> pl.DataFrame:
    group_columns = _group_columns(grouper)
    forward_columns = get_forward_returns_columns(factor_data.columns)
    if not forward_columns:
        return factor_data.clone()
    return factor_data.with_columns(
        [
            (pl.col(column) - pl.col(column).mean().over(group_columns)).alias(column)
            for column in forward_columns
        ]
    )


def backshift_returns_series(
    series: pl.DataFrame,
    periods: int,
    value_column: str = "return",
) -> pl.DataFrame:
    if periods <= 0:
        raise ValueError("periods must be positive")
    dates = series.get_column("date").unique(maintain_order=True).to_list()
    if periods >= len(dates):
        return series.head(0)
    date_map = dict(zip(dates[periods:], dates, strict=False))
    rows = [
        {
            "date": date_map[row["date"]],
            "asset": row["asset"],
            value_column: row[value_column],
        }
        for row in series.iter_rows(named=True)
        if row["date"] in date_map
    ]
    if not rows:
        return series.select("date", "asset", value_column).head(0)
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(series.schema["date"]))
        .sort("date", "asset")
    )


def timedelta_to_string(value: timedelta) -> str:
    days = value.days
    remainder = value - timedelta(days=days)
    seconds = remainder.seconds
    microseconds = remainder.microseconds
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds, microseconds = divmod(microseconds, 1_000)
    parts = []
    if days:
        parts.append(f"{days}D")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    if milliseconds:
        parts.append(f"{milliseconds}ms")
    if microseconds:
        parts.append(f"{microseconds}us")
    return "".join(parts)


def add_custom_calendar_timedelta(
    value: datetime,
    delta: timedelta,
    holidays: set[date] | None = None,
    weekmask: set[int] | None = None,
) -> datetime:
    holidays = holidays or set()
    weekmask = weekmask or {0, 1, 2, 3, 4}
    days = delta.days
    offset = delta - timedelta(days=days)
    current = value
    step = 1 if days >= 0 else -1
    for _ in range(abs(days)):
        current = _next_calendar_day(current, step, holidays, weekmask)
    return current + offset


def diff_custom_calendar_timedeltas(
    start: datetime,
    end: datetime,
    holidays: set[date] | None = None,
    weekmask: set[int] | None = None,
) -> timedelta:
    if end < start:
        return -diff_custom_calendar_timedeltas(end, start, holidays, weekmask)
    holidays = holidays or set()
    weekmask = weekmask or {0, 1, 2, 3, 4}
    business_days = 0
    current = start
    while True:
        next_day = _next_calendar_day(current, 1, holidays, weekmask)
        if next_day > end:
            break
        business_days += 1
        current = next_day
    return timedelta(days=business_days) + (end - current)


def timedelta_strings_to_integers(sequence: list[str] | tuple[str, ...]) -> list[int]:
    return [_duration_microseconds(value) // _DAY_MICROSECONDS for value in sequence]


def infer_trading_calendar(
    factor_dates: list[datetime] | tuple[datetime, ...],
    price_dates: list[datetime] | tuple[datetime, ...],
) -> dict[str, set[date] | set[int]]:
    observed = sorted({value.date() for value in [*factor_dates, *price_dates]})
    if not observed:
        return {"weekmask": set(), "holidays": set()}
    weekmask = {value.weekday() for value in observed}
    start = observed[0]
    end = observed[-1]
    holidays = {
        current
        for current in _date_range(start, end)
        if current.weekday() in weekmask and current not in observed
    }
    return {"weekmask": weekmask, "holidays": holidays}


def _normalize_bins(bins: int | tuple[float, ...] | None) -> int | list[float] | None:
    if bins is None or isinstance(bins, int):
        return bins
    return list(bins)


def _attach_groupby(
    factor: pl.DataFrame,
    groupby: Mapping[str, object] | None,
    groupby_labels: Mapping[object, object] | None = None,
) -> pl.DataFrame:
    if groupby is None:
        return factor
    assets = factor.get_column("asset").to_list()
    missing_assets = sorted(set(assets) - set(groupby.keys()))
    if missing_assets:
        raise KeyError(f"Assets {missing_assets} not in group mapping")
    raw_groups = [groupby[asset] for asset in assets]
    if groupby_labels is not None:
        missing_groups = sorted(set(raw_groups) - set(groupby_labels.keys()))
        if missing_groups:
            raise KeyError(f"groups {missing_groups} not in passed group names")
    groups = [_format_group(group, groupby_labels) for group in raw_groups]
    return factor.with_columns(pl.Series("group", groups, dtype=pl.String))


def _format_group(
    group: object,
    groupby_labels: Mapping[object, object] | None,
) -> str:
    if groupby_labels is not None:
        return str(groupby_labels.get(group, group))
    return str(group)


def _group_columns(grouper: str | tuple[str, ...] | list[str] | None) -> list[str]:
    if grouper is None:
        return ["date"]
    if isinstance(grouper, str):
        return [grouper]
    return list(grouper)


def _series_period(series: pl.Series) -> float:
    return _duration_microseconds(series.name)


def _forward_return_duration(
    column: str, require_exact_day_multiple: bool
) -> int | None:
    label = column.removeprefix("forward_return_")
    duration = _try_duration_microseconds(label)
    if duration is None:
        return None
    if require_exact_day_multiple and duration % _DAY_MICROSECONDS != 0:
        return None
    return duration


def _duration_microseconds(label: str) -> int:
    duration = _try_duration_microseconds(label)
    if duration is None:
        raise ValueError(f"invalid period label: {label}")
    return duration


def _try_duration_microseconds(label: str) -> int | None:
    position = 0
    total = 0
    matches = list(_DURATION_TOKEN_RE.finditer(label))
    if not matches:
        return None
    for match in matches:
        if match.start() != position:
            return None
        position = match.end()
        total += int(match.group("value")) * _UNIT_MICROSECONDS[match.group("unit")]
    if position != len(label):
        return None
    return total


def _next_calendar_day(
    value: datetime,
    step: int,
    holidays: set[date],
    weekmask: set[int],
) -> datetime:
    current = value
    while True:
        current += timedelta(days=step)
        if current.weekday() in weekmask and current.date() not in holidays:
            return current


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _compat_error(error: ValueError) -> ValueError:
    message = str(error)
    if "timezone" in message:
        return NonMatchingTimezoneError(message)
    if "data loss" in message and "max_loss" in message:
        return MaxLossExceededError(message)
    return error


def _format_table(table: pl.DataFrame, fmt) -> str:
    rows = [table.columns]
    for row in table.iter_rows():
        rows.append([_format_print_value(value, fmt) for value in row])
    widths = [max(len(str(row[index])) for row in rows) for index in range(table.width)]
    return "\n".join(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _format_print_value(value, fmt) -> str:
    if value is None:
        return "null"
    if (
        fmt is not None
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ):
        return fmt.format(value)
    return str(value)


_DAY_MICROSECONDS = 86_400_000_000
_UNIT_MICROSECONDS = {
    "D": _DAY_MICROSECONDS,
    "d": _DAY_MICROSECONDS,
    "h": 3_600_000_000,
    "H": 3_600_000_000,
    "m": 60_000_000,
    "M": 60_000_000,
    "s": 1_000_000,
    "S": 1_000_000,
    "ms": 1_000,
    "MS": 1_000,
    "us": 1,
    "US": 1,
    "ns": 0,
    "NS": 0,
}
_DURATION_TOKEN_RE = re.compile(
    r"(?P<value>\d+)(?P<unit>ms|us|ns|[Dhms])", re.IGNORECASE
)
_NON_UNIQUE_BIN_EDGES_MESSAGE = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range
    3 - Use 'bins' option instead of 'quantiles'

"""


__all__ = [
    "add_custom_calendar_timedelta",
    "backshift_returns_series",
    "CleanFactorResult",
    "demean_forward_returns",
    "diff_custom_calendar_timedeltas",
    "get_forward_returns_columns",
    "infer_trading_calendar",
    "LossReport",
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
    "compute_forward_returns",
    "get_clean_factor",
    "get_clean_factor_and_forward_returns",
    "print_table",
    "non_unique_bin_edges_error",
    "quantize_factor",
    "rate_of_return",
    "rethrow",
    "std_conversion",
    "timedelta_strings_to_integers",
    "timedelta_to_string",
    "validate_factor_frame",
]
