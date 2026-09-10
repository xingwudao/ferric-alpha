from datetime import datetime, timedelta

import ferric_alpha
import polars as pl
import pytest
from ferric_alpha import utils


def test_alphalens_compat_exceptions_are_exported() -> None:
    assert ferric_alpha.NonMatchingTimezoneError is utils.NonMatchingTimezoneError
    assert ferric_alpha.MaxLossExceededError is utils.MaxLossExceededError
    assert issubclass(utils.NonMatchingTimezoneError, ValueError)
    assert issubclass(utils.MaxLossExceededError, ValueError)


def test_print_table_formats_polars_frames_and_series(capsys) -> None:
    frame = pl.DataFrame({"metric": ["ic"], "value": [12.3456]})
    series = pl.Series("spread", [0.1234, None])

    assert utils.print_table(frame, name="Summary", fmt="{0:.2f}") is None
    assert utils.print_table(series, fmt="{0:.1%}") is None

    output = capsys.readouterr().out
    assert "Summary" in output
    assert "metric" in output
    assert "12.35" in output
    assert "spread" in output
    assert "12.3%" in output
    assert "null" in output


def test_rethrow_preserves_exception_type_and_appends_message() -> None:
    with pytest.raises(ValueError, match="base detail"):
        try:
            raise ValueError("base")
        except ValueError as error:
            utils.rethrow(error, " detail")


def test_non_unique_bin_edges_error_adds_quantile_guidance() -> None:
    @utils.non_unique_bin_edges_error
    def fail_with_pandas_bin_message():
        raise ValueError("Bin edges must be unique")

    with pytest.raises(ValueError, match="Decrease the number of quantiles"):
        fail_with_pandas_bin_message()


def test_get_forward_returns_columns_detects_period_labels() -> None:
    columns = [
        "date",
        "asset",
        "factor",
        "forward_return_1D",
        "forward_return_3h",
        "forward_return_1D1h",
        "not_a_return",
    ]

    assert utils.get_forward_returns_columns(columns) == [
        "forward_return_1D",
        "forward_return_3h",
        "forward_return_1D1h",
    ]
    assert utils.get_forward_returns_columns(
        columns,
        require_exact_day_multiple=True,
    ) == ["forward_return_1D"]


def test_rate_and_std_conversion_use_named_period_duration() -> None:
    returns = pl.Series("2D", [0.21, -0.19])
    std = pl.Series("2D", [0.20, 0.40])

    assert utils.rate_of_return(returns, "1D").round(12).to_list() == [0.1, -0.1]
    assert utils.std_conversion(std, "1D").round(12).to_list() == [
        0.141421356237,
        0.282842712475,
    ]


def test_demean_forward_returns_by_date_and_group() -> None:
    frame = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
            ],
            "asset": ["A", "B", "C", "D"],
            "group": ["g1", "g1", "g2", "g2"],
            "factor": [1.0, 2.0, 3.0, 4.0],
            "forward_return_1D": [0.10, 0.30, -0.20, 0.20],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    date_adjusted = utils.demean_forward_returns(frame)
    group_adjusted = utils.demean_forward_returns(frame, grouper=("date", "group"))

    assert date_adjusted.get_column("forward_return_1D").round(12).to_list() == [
        0.0,
        0.2,
        -0.3,
        0.1,
    ]
    assert group_adjusted.get_column("forward_return_1D").round(12).to_list() == [
        -0.1,
        0.1,
        -0.2,
        0.2,
    ]


def test_backshift_returns_series_aligns_prior_dates_to_future_returns() -> None:
    frame = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 3),
            ],
            "asset": ["A", "B", "A", "B", "A", "B"],
            "return": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    shifted = utils.backshift_returns_series(frame, 1)

    assert shifted.get_column("date").to_list() == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 2),
    ]
    assert shifted.get_column("return").to_list() == [0.02, -0.02, 0.03, -0.03]


def test_timedelta_to_string_emits_alphalens_duration_labels() -> None:
    assert (
        utils.timedelta_to_string(timedelta(days=1, hours=3, minutes=15)) == "1D3h15m"
    )
    assert (
        utils.timedelta_to_string(timedelta(milliseconds=2, microseconds=7)) == "2ms7us"
    )


def test_custom_calendar_timedelta_add_and_diff_skip_holidays() -> None:
    start = datetime(2024, 1, 5, 9, 30)
    end = datetime(2024, 1, 10, 12, 30)
    holidays = {datetime(2024, 1, 8).date()}

    assert (
        utils.add_custom_calendar_timedelta(
            start,
            timedelta(days=2, hours=3),
            holidays=holidays,
        )
        == end
    )
    assert utils.diff_custom_calendar_timedeltas(
        start,
        end,
        holidays=holidays,
    ) == timedelta(days=2, hours=3)


def test_timedelta_strings_to_integers_extracts_day_counts() -> None:
    assert utils.timedelta_strings_to_integers(["1D", "5D", "1D3h"]) == [1, 5, 1]


def test_infer_trading_calendar_reports_weekmask_and_holidays() -> None:
    factor_dates = [
        datetime(2024, 1, 2),
        datetime(2024, 1, 9),
        datetime(2024, 1, 11),
    ]
    price_dates = [
        datetime(2024, 1, 2),
        datetime(2024, 1, 9),
        datetime(2024, 1, 11),
    ]

    calendar = utils.infer_trading_calendar(factor_dates, price_dates)

    assert calendar["weekmask"] == {1, 3}
    assert calendar["holidays"] == {datetime(2024, 1, 4).date()}
