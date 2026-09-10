from datetime import datetime, timedelta

import polars as pl
import pytest
from ferric_alpha import utils


def _factor_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            ],
            "asset": ["A", "B", "A"],
            "factor": [1.0, 2.0, 1.5],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _price_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "asset": ["A", "A", "A", "B", "B", "B"],
            "price": [100.0, 110.0, 121.0, 50.0, 40.0, 60.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _upstream_quantize_fixture(name: str) -> pl.DataFrame:
    dates = [datetime(2015, 1, 1), datetime(2015, 1, 2)]
    if name == "regular":
        rows = [
            (dates[0], "A", 1.0, "1"),
            (dates[0], "B", 2.0, "1"),
            (dates[0], "C", 3.0, "2"),
            (dates[0], "D", 4.0, "2"),
            (dates[1], "A", 4.0, "1"),
            (dates[1], "B", 3.0, "1"),
            (dates[1], "C", 2.0, "2"),
            (dates[1], "D", 1.0, "2"),
        ]
    elif name == "biased":
        rows = [
            (dates[0], "A", -1.0, "1"),
            (dates[0], "B", 3.0, "1"),
            (dates[0], "C", -2.0, "2"),
            (dates[0], "D", 4.0, "2"),
            (dates[0], "E", -5.0, "1"),
            (dates[0], "F", 7.0, "1"),
            (dates[0], "G", -6.0, "2"),
            (dates[0], "H", 8.0, "2"),
            (dates[1], "A", -4.0, "1"),
            (dates[1], "B", 2.0, "1"),
            (dates[1], "C", -3.0, "2"),
            (dates[1], "D", 1.0, "2"),
            (dates[1], "E", -8.0, "1"),
            (dates[1], "F", 6.0, "1"),
            (dates[1], "G", -7.0, "2"),
            (dates[1], "H", 5.0, "2"),
        ]
    else:
        raise ValueError(f"unknown fixture: {name}")
    return pl.DataFrame(
        rows,
        schema=["date", "asset", "factor", "group"],
        orient="row",
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _upstream_clean_prices(
    dates: list[datetime],
    *,
    intraday: bool = False,
) -> pl.DataFrame:
    tickers = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for offset, current_date in enumerate(dates, start=1):
        values = [
            1.10**offset,
            0.50**offset,
            3.00**offset,
            0.90**offset,
            0.50**offset,
            1.00**offset,
        ]
        if intraday:
            for hour, multiplier in [(9, 1.0), (10, 1.001), (12, 0.998)]:
                timestamp = current_date.replace(hour=hour, minute=30)
                for asset, price in zip(tickers, values, strict=True):
                    rows.append((timestamp, asset, price * multiplier))
        else:
            for asset, price in zip(tickers, values, strict=True):
                rows.append((current_date, asset, price))
    return pl.DataFrame(
        rows,
        schema=["date", "asset", "price"],
        orient="row",
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _upstream_clean_factor(
    dates: list[datetime],
    pattern: list[list[float | None]],
    *,
    intraday: bool = False,
) -> pl.DataFrame:
    tickers = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for current_date, values in zip(dates, pattern, strict=True):
        timestamp = (
            current_date.replace(hour=9, minute=30) if intraday else current_date
        )
        for asset, value in zip(tickers, values, strict=True):
            if value is not None:
                rows.append((timestamp, asset, float(value)))
    return pl.DataFrame(
        rows,
        schema=["date", "asset", "factor"],
        orient="row",
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _business_days(
    start: datetime, count: int, holidays: set[datetime] | None = None
) -> list[datetime]:
    holidays = holidays or set()
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5 and current not in holidays:
            days.append(current)
        current += timedelta(days=1)
    return days


def _assert_upstream_clean_prefix(
    result: pl.DataFrame,
    *,
    expected_columns: list[str],
    expected_dates: list[datetime],
    expected_assets: list[str],
    expected_factor_quantiles: list[int],
    expected_returns: dict[str, list[float]],
) -> None:
    assert result.columns == expected_columns
    assert (
        result.get_column("date").head(len(expected_dates)).to_list() == expected_dates
    )
    assert (
        result.get_column("asset").head(len(expected_assets)).to_list()
        == expected_assets
    )
    assert (
        result.get_column("factor_quantile")
        .head(len(expected_factor_quantiles))
        .to_list()
        == expected_factor_quantiles
    )
    for column, expected in expected_returns.items():
        assert (
            result.get_column(column).head(len(expected)).round(6).to_list() == expected
        )


def test_compute_forward_returns_uses_polars_frames() -> None:
    result = utils.compute_forward_returns(
        _factor_frame(), _price_frame(), periods=(1,)
    )

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["date", "asset", "forward_return_1D"]
    assert result.get_column("forward_return_1D").to_list() == [0.1, -0.2, 0.1]


def test_compute_forward_returns_supports_non_cumulative_returns() -> None:
    result = utils.compute_forward_returns(
        _factor_frame(),
        _price_frame(),
        periods=(2,),
        cumulative_returns=False,
    )

    assert result.get_column("forward_return_2D").to_list() == [0.1, 0.5, None]


def test_compute_forward_returns_filters_asset_zscore_outliers() -> None:
    factor_dates = [datetime(2024, 1, day) for day in range(1, 5)]
    factor = pl.DataFrame(
        {
            "date": factor_dates,
            "asset": ["A"] * len(factor_dates),
            "factor": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    prices = pl.DataFrame(
        {
            "date": [datetime(2024, 1, day) for day in range(1, 6)],
            "asset": ["A"] * 5,
            "price": [100.0, 101.0, 102.01, 204.02, 206.0602],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.compute_forward_returns(
        factor,
        prices,
        periods=(1,),
        filter_zscore=1.0,
    )

    assert result.get_column("forward_return_1D").to_list() == [0.01, 0.01, None, 0.01]


def test_quantize_factor_returns_integer_quantiles() -> None:
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.quantize_factor(frame, quantiles=2)

    assert result.get_column("factor_quantile").to_list() == [1, 2]


def test_get_clean_factor_and_forward_returns_returns_frame_and_loss() -> None:
    result = utils.get_clean_factor_and_forward_returns(
        _factor_frame(),
        _price_frame(),
        periods=(1,),
        quantiles=2,
        max_loss=0.5,
    )

    assert isinstance(result.frame, pl.DataFrame)
    assert result.loss.input_rows == 3
    assert result.loss.forward_return_loss == 0
    assert result.frame.columns == [
        "date",
        "asset",
        "factor",
        "factor_quantile",
        "forward_return_1D",
    ]


def test_forward_returns_reject_bad_periods_as_value_error() -> None:
    with pytest.raises(ValueError, match="period count must be positive"):
        utils.compute_forward_returns(_factor_frame(), _price_frame(), periods=(0,))


def test_quantize_factor_supports_group_and_zero_aware() -> None:
    grouped = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "group": ["g1", "g1", "g2", "g2"],
            "factor": [1.0, 2.0, 10.0, 20.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    zero_aware = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": [-2.0, -1.0, 1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    group_result = utils.quantize_factor(grouped, quantiles=2, by_group=True)
    zero_result = utils.quantize_factor(zero_aware, quantiles=4, zero_aware=True)

    assert group_result.get_column("factor_quantile").to_list() == [1, 2, 1, 2]
    assert zero_result.get_column("factor_quantile").to_list() == [1, 2, 3, 4]


def test_quantize_factor_supports_explicit_bins() -> None:
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": [-2.0, -0.5, 0.5, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.quantize_factor(frame, quantiles=None, bins=(-3.0, 0.0, 3.0))

    assert result.get_column("factor_quantile").to_list() == [1, 1, 2, 2]


@pytest.mark.parametrize(
    ("fixture_name", "quantiles", "bins", "by_group", "zero_aware", "expected"),
    [
        ("regular", 4, None, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
        ("regular", 2, None, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
        ("regular", 2, None, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
        (
            "biased",
            4,
            None,
            False,
            True,
            [2, 3, 2, 3, 1, 4, 1, 4, 2, 3, 2, 3, 1, 4, 1, 4],
        ),
        (
            "biased",
            2,
            None,
            False,
            True,
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        ),
        (
            "biased",
            2,
            None,
            True,
            True,
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        ),
        (
            "biased",
            None,
            4,
            False,
            True,
            [2, 3, 2, 3, 1, 4, 1, 4, 2, 3, 2, 3, 1, 4, 1, 4],
        ),
        (
            "biased",
            None,
            2,
            False,
            True,
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        ),
        (
            "biased",
            None,
            2,
            True,
            True,
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        ),
        (
            "regular",
            (0.0, 0.25, 0.5, 0.75, 1.0),
            None,
            False,
            False,
            [1, 2, 3, 4, 4, 3, 2, 1],
        ),
        (
            "regular",
            (0.0, 0.5, 0.75, 1.0),
            None,
            False,
            False,
            [1, 1, 2, 3, 3, 2, 1, 1],
        ),
        (
            "regular",
            (0.0, 0.25, 0.5, 1.0),
            None,
            False,
            False,
            [1, 2, 3, 3, 3, 3, 2, 1],
        ),
        (
            "regular",
            (0.0, 0.5, 1.0),
            None,
            False,
            False,
            [1, 1, 2, 2, 2, 2, 1, 1],
        ),
        (
            "regular",
            (0.25, 0.5, 0.75),
            None,
            False,
            False,
            [None, 1, 2, None, None, 2, 1, None],
        ),
        (
            "regular",
            (0.0, 0.5, 1.0),
            None,
            True,
            False,
            [1, 2, 1, 2, 2, 1, 2, 1],
        ),
        (
            "regular",
            (0.5, 1.0),
            None,
            True,
            False,
            [None, 1, None, 1, 1, None, 1, None],
        ),
        (
            "regular",
            (0.0, 1.0),
            None,
            True,
            False,
            [1, 1, 1, 1, 1, 1, 1, 1],
        ),
        ("regular", None, 4, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
        ("regular", None, 2, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
        ("regular", None, 3, False, False, [1, 1, 2, 3, 3, 2, 1, 1]),
        ("regular", None, 8, False, False, [1, 3, 6, 8, 8, 6, 3, 1]),
        (
            "regular",
            None,
            (0.0, 1.0, 2.0, 3.0, 5.0),
            False,
            False,
            [1, 2, 3, 4, 4, 3, 2, 1],
        ),
        (
            "regular",
            None,
            (1.0, 2.0, 3.0),
            False,
            False,
            [None, 1, 2, None, None, 2, 1, None],
        ),
        (
            "regular",
            None,
            (0.0, 2.0, 5.0),
            False,
            False,
            [1, 1, 2, 2, 2, 2, 1, 1],
        ),
        (
            "regular",
            None,
            (0.5, 2.5, 4.5),
            False,
            False,
            [1, 1, 2, 2, 2, 2, 1, 1],
        ),
        (
            "regular",
            None,
            (0.5, 2.5),
            True,
            False,
            [1, 1, None, None, None, None, 1, 1],
        ),
        ("regular", None, 2, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
    ],
)
def test_quantize_factor_matches_upstream_parameterized_cases(
    fixture_name: str,
    quantiles: int | tuple[float, ...] | None,
    bins: int | tuple[float, ...] | None,
    by_group: bool,
    zero_aware: bool,
    expected: list[int | None],
) -> None:
    frame = _upstream_quantize_fixture(fixture_name)

    result = utils.quantize_factor(
        frame,
        quantiles=quantiles,
        bins=bins,
        by_group=by_group,
        zero_aware=zero_aware,
    )

    assert result.get_column("factor_quantile").to_list() == expected


def test_quantize_factor_accepts_alphalens_integer_bin_counts() -> None:
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.quantize_factor(frame, quantiles=None, bins=3)

    assert result.get_column("factor_quantile").to_list() == [1, 1, 2, 3]


def test_quantize_factor_accepts_alphalens_quantile_edge_lists() -> None:
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.quantize_factor(frame, quantiles=(0.0, 0.5, 1.0))

    assert result.get_column("factor_quantile").to_list() == [1, 1, 2, 2]


def test_get_clean_factor_and_forward_returns_supports_explicit_bins() -> None:
    result = utils.get_clean_factor_and_forward_returns(
        _factor_frame(),
        _price_frame(),
        periods=(1,),
        quantiles=None,
        bins=(0.0, 1.4, 3.0),
        max_loss=0.5,
    )

    assert result.frame.get_column("factor_quantile").to_list() == [1, 2, 2]


def test_get_clean_factor_and_forward_returns_accepts_groupby_mapping() -> None:
    result = utils.get_clean_factor_and_forward_returns(
        _factor_frame(),
        _price_frame(),
        periods=(1,),
        quantiles=None,
        bins=(0.0, 3.0),
        groupby={"A": "growth", "B": "value"},
        max_loss=0.5,
    )

    assert result.frame.get_column("group").to_list() == [
        "growth",
        "value",
        "growth",
    ]


def test_get_clean_factor_and_forward_returns_applies_groupby_labels() -> None:
    result = utils.get_clean_factor_and_forward_returns(
        _factor_frame(),
        _price_frame(),
        periods=(1,),
        quantiles=None,
        bins=(0.0, 3.0),
        groupby={"A": 1, "B": 2},
        groupby_labels={1: "growth", 2: "value"},
        max_loss=0.5,
    )

    assert result.frame.get_column("group").to_list() == [
        "growth",
        "value",
        "growth",
    ]


def test_get_clean_factor_and_forward_returns_rejects_incomplete_groupby_mapping() -> (
    None
):
    with pytest.raises(KeyError, match="Assets"):
        utils.get_clean_factor_and_forward_returns(
            _factor_frame(),
            _price_frame(),
            periods=(1,),
            quantiles=None,
            bins=(0.0, 3.0),
            groupby={"A": "growth"},
            max_loss=0.5,
        )


def test_get_clean_factor_and_forward_returns_rejects_incomplete_groupby_labels() -> (
    None
):
    with pytest.raises(KeyError, match="groups"):
        utils.get_clean_factor_and_forward_returns(
            _factor_frame(),
            _price_frame(),
            periods=(1,),
            quantiles=None,
            bins=(0.0, 3.0),
            groupby={"A": 1, "B": 2},
            groupby_labels={1: "growth"},
            max_loss=0.5,
        )


def test_get_clean_factor_and_forward_returns_accepts_binning_by_group_alias() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": [1.0, 2.0, 10.0, 20.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    prices = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
            ],
            "asset": ["A", "B", "C", "D"] * 2,
            "price": [100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        quantiles=2,
        groupby={"A": "g1", "B": "g1", "C": "g2", "D": "g2"},
        binning_by_group=True,
        max_loss=0.5,
    )

    assert result.frame.get_column("factor_quantile").to_list() == [1, 2, 1, 2]


def test_get_clean_factor_and_forward_returns_matches_upstream_daily_fixture() -> None:
    factor_dates = [datetime(2015, 1, day) for day in (11, 12, 13)]
    price_dates = [datetime(2015, 1, day) for day in range(11, 17)]
    factor = _upstream_clean_factor(
        factor_dates,
        [
            [3.0, 4.0, 2.0, 1.0, None, None],
            [3.0, None, None, 1.0, 4.0, 2.0],
            [3.0, 4.0, 2.0, 1.0, None, None],
        ],
    )

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        _upstream_clean_prices(price_dates),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=4,
        periods=(1, 2, 3),
    ).frame

    assert result.height == 12
    _assert_upstream_clean_prefix(
        result,
        expected_columns=[
            "date",
            "asset",
            "factor",
            "group",
            "factor_quantile",
            "forward_return_1D",
            "forward_return_2D",
            "forward_return_3D",
        ],
        expected_dates=[
            datetime(2015, 1, 11),
            datetime(2015, 1, 11),
            datetime(2015, 1, 11),
            datetime(2015, 1, 11),
            datetime(2015, 1, 12),
            datetime(2015, 1, 12),
            datetime(2015, 1, 12),
            datetime(2015, 1, 12),
            datetime(2015, 1, 13),
            datetime(2015, 1, 13),
            datetime(2015, 1, 13),
            datetime(2015, 1, 13),
        ],
        expected_assets=["A", "B", "C", "D", "A", "D", "E", "F", "A", "B", "C", "D"],
        expected_factor_quantiles=[3, 4, 2, 1, 3, 1, 4, 2, 3, 4, 2, 1],
        expected_returns={
            "forward_return_1D": [
                0.1,
                -0.5,
                2.0,
                -0.1,
                0.1,
                -0.1,
                -0.5,
                0.0,
                0.1,
                -0.5,
                2.0,
                -0.1,
            ],
            "forward_return_2D": [
                0.21,
                -0.75,
                8.0,
                -0.19,
                0.21,
                -0.19,
                -0.75,
                0.0,
                0.21,
                -0.75,
                8.0,
                -0.19,
            ],
            "forward_return_3D": [
                0.331,
                -0.875,
                26.0,
                -0.271,
                0.331,
                -0.271,
                -0.875,
                0.0,
                0.331,
                -0.875,
                26.0,
                -0.271,
            ],
        },
    )


def test_clean_factor_matches_upstream_business_day_fixture() -> None:
    factor_dates = _business_days(datetime(2017, 1, 12), 3)
    price_dates = _business_days(datetime(2017, 1, 12), 6)
    factor = _upstream_clean_factor(
        factor_dates,
        [
            [3.0, 4.0, 2.0, 1.0, None, None],
            [3.0, None, None, 1.0, 4.0, 2.0],
            [3.0, 4.0, 2.0, 1.0, None, None],
        ],
    )

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        _upstream_clean_prices(price_dates),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=4,
        periods=(1, 2, 3),
    ).frame

    assert result.height == 12
    assert (
        result.get_column("date").unique(maintain_order=True).to_list() == factor_dates
    )
    assert result.select(
        "forward_return_1D", "forward_return_2D", "forward_return_3D"
    ).head(4).to_dict(as_series=False) == {
        "forward_return_1D": [0.1, -0.5, 2.0, -0.1],
        "forward_return_2D": [0.21, -0.75, 8.0, -0.19],
        "forward_return_3D": [0.331, -0.875, 26.0, -0.271],
    }


def test_get_clean_factor_and_forward_returns_matches_upstream_intraday_fixture() -> (
    None
):
    factor_dates = _business_days(datetime(2017, 1, 12), 3)
    price_dates = _business_days(datetime(2017, 1, 12), 4)
    factor = _upstream_clean_factor(
        factor_dates,
        [
            [3.0, 4.0, 2.0, 1.0, None, None],
            [3.0, None, None, 1.0, 4.0, 2.0],
            [3.0, 4.0, 2.0, 1.0, None, None],
        ],
        intraday=True,
    )

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        _upstream_clean_prices(price_dates, intraday=True),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=4,
        periods=(1, 2, 3),
    ).frame

    assert result.height == 12
    assert result.columns[-3:] == [
        "forward_return_1h",
        "forward_return_3h",
        "forward_return_1D",
    ]
    returns = (
        result.select("forward_return_1h", "forward_return_3h", "forward_return_1D")
        .head(4)
        .with_columns(pl.all().round(6))
    )
    assert returns.to_dict(as_series=False) == {
        "forward_return_1h": [0.001, 0.001, 0.001, 0.001],
        "forward_return_3h": [-0.002, -0.002, -0.002, -0.002],
        "forward_return_1D": [0.1, -0.5, 2.0, -0.1],
    }


def test_get_clean_factor_and_forward_returns_matches_upstream_event_fixture() -> None:
    factor_dates = _business_days(datetime(2017, 1, 12), 5)
    price_dates = _business_days(datetime(2017, 1, 12), 8)
    factor = _upstream_clean_factor(
        factor_dates,
        [
            [1.0, None, None, None, None, 6.0],
            [4.0, None, None, 7.0, None, None],
            [None, None, None, None, None, None],
            [None, 3.0, None, 2.0, None, None],
            [None, None, 1.0, None, 3.0, None],
        ],
    )

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        _upstream_clean_prices(price_dates),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=4,
        periods=(1, 2, 3),
    ).frame

    assert result.height == 8
    _assert_upstream_clean_prefix(
        result,
        expected_columns=[
            "date",
            "asset",
            "factor",
            "group",
            "factor_quantile",
            "forward_return_1D",
            "forward_return_2D",
            "forward_return_3D",
        ],
        expected_dates=[
            datetime(2017, 1, 12),
            datetime(2017, 1, 12),
            datetime(2017, 1, 13),
            datetime(2017, 1, 13),
            datetime(2017, 1, 17),
            datetime(2017, 1, 17),
            datetime(2017, 1, 18),
            datetime(2017, 1, 18),
        ],
        expected_assets=["A", "F", "A", "D", "B", "D", "C", "E"],
        expected_factor_quantiles=[1, 4, 1, 4, 4, 1, 1, 4],
        expected_returns={
            "forward_return_1D": [0.1, 0.0, 0.1, -0.1, -0.5, -0.1, 2.0, -0.5],
            "forward_return_2D": [0.21, 0.0, 0.21, -0.19, -0.75, -0.19, 8.0, -0.75],
            "forward_return_3D": [
                0.331,
                0.0,
                0.331,
                -0.271,
                -0.875,
                -0.271,
                26.0,
                -0.875,
            ],
        },
    )


def test_clean_factor_matches_upstream_intraday_holiday_fixture() -> None:
    holidays = {
        datetime(2017, 1, 13),
        datetime(2017, 1, 18),
        datetime(2017, 1, 30),
        datetime(2017, 2, 7),
    }
    factor_dates = _business_days(datetime(2017, 1, 12), 18, holidays)
    price_dates = _business_days(datetime(2017, 1, 12), 19, holidays)
    base_pattern = [
        [3.0, 4.0, 2.0, 1.0, None, None],
        [3.0, None, None, 1.0, 4.0, 2.0],
        [3.0, 4.0, 2.0, 1.0, None, None],
    ]
    factor = _upstream_clean_factor(
        factor_dates,
        base_pattern * 6,
        intraday=True,
    )

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        _upstream_clean_prices(price_dates, intraday=True),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=4,
        periods=(1, 2, 3),
    ).frame

    assert result.height == 72
    assert result.get_column("date").unique(maintain_order=True).to_list() == [
        date.replace(hour=9, minute=30) for date in factor_dates
    ]
    assert result.columns[-3:] == [
        "forward_return_1h",
        "forward_return_3h",
        "forward_return_1D",
    ]
    assert result.select(
        "forward_return_1h", "forward_return_3h", "forward_return_1D"
    ).head(12).with_columns(pl.all().round(6)).to_dict(as_series=False) == {
        "forward_return_1h": [0.001] * 12,
        "forward_return_3h": [-0.002] * 12,
        "forward_return_1D": [
            0.1,
            -0.5,
            2.0,
            -0.1,
            0.1,
            -0.1,
            -0.5,
            0.0,
            0.1,
            -0.5,
            2.0,
            -0.1,
        ],
    }


def test_clean_factor_matches_upstream_daily_holiday_fixture() -> None:
    holidays = {
        datetime(2017, 1, 13),
        datetime(2017, 1, 18),
        datetime(2017, 1, 30),
        datetime(2017, 2, 7),
    }
    factor_dates = _business_days(datetime(2017, 1, 12), 18, holidays)
    price_dates = _business_days(datetime(2017, 1, 12), 21, holidays)
    base_pattern = [
        [3.0, 4.0, 2.0, 1.0, None, None],
        [3.0, None, None, 1.0, 4.0, 2.0],
        [3.0, 4.0, 2.0, 1.0, None, None],
    ]
    factor = _upstream_clean_factor(factor_dates, base_pattern * 6)

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        _upstream_clean_prices(price_dates),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=4,
        periods=(1, 2, 3),
    ).frame

    assert result.height == 72
    assert (
        result.get_column("date").unique(maintain_order=True).to_list() == factor_dates
    )
    assert result.columns[-3:] == [
        "forward_return_1D",
        "forward_return_2D",
        "forward_return_3D",
    ]
    assert result.select(
        "forward_return_1D", "forward_return_2D", "forward_return_3D"
    ).head(12).to_dict(as_series=False) == {
        "forward_return_1D": [
            0.1,
            -0.5,
            2.0,
            -0.1,
            0.1,
            -0.1,
            -0.5,
            0.0,
            0.1,
            -0.5,
            2.0,
            -0.1,
        ],
        "forward_return_2D": [
            0.21,
            -0.75,
            8.0,
            -0.19,
            0.21,
            -0.19,
            -0.75,
            0.0,
            0.21,
            -0.75,
            8.0,
            -0.19,
        ],
        "forward_return_3D": [
            0.331,
            -0.875,
            26.0,
            -0.271,
            0.331,
            -0.271,
            -0.875,
            0.0,
            0.331,
            -0.875,
            26.0,
            -0.271,
        ],
    }


def test_get_clean_factor_and_forward_returns_infers_intraday_period_labels() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 2, 9, 30)] * 2,
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    price_dates = [
        datetime(2024, 1, 2, 9, 30),
        datetime(2024, 1, 2, 10, 30),
        datetime(2024, 1, 2, 12, 30),
        datetime(2024, 1, 3, 9, 30),
    ]
    prices = pl.DataFrame(
        {
            "date": [date for date in price_dates for _ in ("A", "B")],
            "asset": ["A", "B"] * len(price_dates),
            "price": [100.0, 50.0, 101.0, 51.0, 98.0, 49.0, 110.0, 45.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2, 3),
        quantiles=2,
        max_loss=0.5,
    )

    assert result.frame.columns == [
        "date",
        "asset",
        "factor",
        "factor_quantile",
        "forward_return_1h",
        "forward_return_3h",
        "forward_return_1D",
    ]
    assert result.frame.select(
        "forward_return_1h", "forward_return_3h", "forward_return_1D"
    ).to_dict(as_series=False) == {
        "forward_return_1h": [0.01, 0.02],
        "forward_return_3h": [-0.02, -0.02],
        "forward_return_1D": [0.1, -0.1],
    }


def test_get_clean_factor_and_forward_returns_uses_trading_day_intraday_labels() -> (
    None
):
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 5, 9, 30)] * 2,
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    price_dates = [
        datetime(2024, 1, 5, 9, 30),
        datetime(2024, 1, 5, 10, 30),
        datetime(2024, 1, 5, 12, 30),
        datetime(2024, 1, 8, 9, 30),
    ]
    prices = pl.DataFrame(
        {
            "date": [date for date in price_dates for _ in ("A", "B")],
            "asset": ["A", "B"] * len(price_dates),
            "price": [100.0, 50.0, 101.0, 51.0, 98.0, 49.0, 110.0, 45.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2, 3),
        quantiles=2,
        max_loss=0.5,
    )

    assert result.frame.columns[-3:] == [
        "forward_return_1h",
        "forward_return_3h",
        "forward_return_1D",
    ]


def test_get_clean_factor_and_forward_returns_uses_observed_business_sessions() -> None:
    sessions = [
        datetime(2024, 1, 4),
        datetime(2024, 1, 8),
        datetime(2024, 1, 9),
        datetime(2024, 1, 10),
    ]
    factor = pl.DataFrame(
        {
            "date": [sessions[0], sessions[0]],
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    prices = pl.DataFrame(
        {
            "date": [date for date in sessions for _ in ("A", "B")],
            "asset": ["A", "B"] * len(sessions),
            "price": [100.0, 50.0, 110.0, 45.0, 121.0, 40.5, 133.1, 36.45],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1, 2, 3),
        quantiles=2,
        max_loss=0.5,
    )

    assert result.frame.columns[-3:] == [
        "forward_return_1D",
        "forward_return_2D",
        "forward_return_3D",
    ]
    assert result.frame.select(
        "forward_return_1D", "forward_return_2D", "forward_return_3D"
    ).to_dict(as_series=False) == {
        "forward_return_1D": [0.1, -0.1],
        "forward_return_2D": [0.21, -0.19],
        "forward_return_3D": [0.331, -0.271],
    }


def test_get_clean_factor_and_forward_returns_enforces_max_loss_threshold() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    prices = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 1),
            ],
            "asset": ["A", "A", "B"],
            "price": [100.0, 110.0, 50.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    with pytest.raises(utils.MaxLossExceededError, match="data loss"):
        utils.get_clean_factor_and_forward_returns(
            factor,
            prices,
            periods=(1,),
            quantiles=2,
            max_loss=0.49,
        )

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        quantiles=2,
        max_loss=0.5,
    )

    assert result.loss.input_rows == 2
    assert result.loss.forward_return_loss == 1
    assert result.frame.get_column("asset").to_list() == ["A"]


def test_get_clean_factor_and_forward_returns_filters_zscore_outliers() -> None:
    factor_dates = [datetime(2024, 1, day) for day in range(1, 5)]
    factor = pl.DataFrame(
        {
            "date": factor_dates,
            "asset": ["A"] * len(factor_dates),
            "factor": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    prices = pl.DataFrame(
        {
            "date": [datetime(2024, 1, day) for day in range(1, 6)],
            "asset": ["A"] * 5,
            "price": [100.0, 101.0, 102.01, 204.02, 206.0602],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        quantiles=None,
        bins=(0.0, 5.0),
        filter_zscore=1.0,
        max_loss=0.5,
    )

    assert result.loss.forward_return_loss == 1
    assert result.frame.get_column("date").to_list() == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 4),
    ]


def test_get_clean_factor_and_forward_returns_preserves_matching_timezone() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms", "UTC")))
    prices = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
            ],
            "asset": ["A", "B", "A", "B"],
            "price": [100.0, 50.0, 110.0, 45.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms", "UTC")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        quantiles=2,
        max_loss=0.5,
    )

    assert result.frame.schema["date"] == pl.Datetime("ms", "UTC")


def test_get_clean_factor_and_forward_returns_rejects_timezone_mismatch() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms", "UTC")))
    prices = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
            ],
            "asset": ["A", "B", "A", "B"],
            "price": [100.0, 50.0, 110.0, 45.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms", "Asia/Shanghai")))

    with pytest.raises(utils.NonMatchingTimezoneError, match="timezone"):
        utils.get_clean_factor_and_forward_returns(
            factor,
            prices,
            periods=(1,),
            quantiles=2,
            max_loss=0.5,
        )


def test_get_clean_factor_and_forward_returns_handles_sparse_event_factors() -> None:
    dates = [datetime(2024, 1, 2) + timedelta(days=day) for day in range(5)]
    factor = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[3], dates[3]],
            "asset": ["A", "B", "A", "B", "A", "B"],
            "factor": [1.0, 6.0, 4.0, 7.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    prices = pl.DataFrame(
        {
            "date": [date for date in dates for _ in ("A", "B")],
            "asset": ["A", "B"] * len(dates),
            "price": [
                100.0,
                50.0,
                110.0,
                45.0,
                121.0,
                40.5,
                133.1,
                36.45,
                146.41,
                32.805,
            ],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        quantiles=4,
        max_loss=0.5,
    )

    assert result.frame.get_column("asset").to_list() == ["A", "B", "A", "B", "A", "B"]
    assert result.frame.get_column("factor_quantile").to_list() == [1, 4, 1, 4, 1, 4]
    assert result.frame.get_column("forward_return_1D").to_list() == [
        0.1,
        -0.1,
        0.1,
        -0.1,
        0.1,
        -0.1,
    ]


def test_get_clean_factor_rejects_invalid_max_loss() -> None:
    returns = utils.compute_forward_returns(
        _factor_frame(), _price_frame(), periods=(1,)
    )

    with pytest.raises(ValueError, match="max_loss must be finite"):
        utils.get_clean_factor(
            _factor_frame(), returns, quantiles=2, max_loss=float("nan")
        )


def test_get_clean_factor_counts_nan_forward_returns_as_loss() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    returns = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "asset": ["A", "B"],
            "forward_return_1D": [0.1, float("nan")],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor(
        factor,
        returns,
        quantiles=2,
        max_loss=0.5,
    )

    assert result.loss.forward_return_loss == 1
    assert result.frame.get_column("asset").to_list() == ["A"]


def test_get_clean_factor_accepts_groupby_and_binning_alias() -> None:
    factor = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "factor": [1.0, 2.0, 10.0, 20.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
    returns = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 4,
            "asset": ["A", "B", "C", "D"],
            "forward_return_1D": [0.01, 0.02, 0.03, 0.04],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.get_clean_factor(
        factor,
        returns,
        quantiles=2,
        groupby={"A": 1, "B": 1, "C": 2, "D": 2},
        groupby_labels={1: "g1", 2: "g2"},
        binning_by_group=True,
        max_loss=0.5,
    )

    assert result.frame.get_column("group").to_list() == ["g1", "g1", "g2", "g2"]
    assert result.frame.get_column("factor_quantile").to_list() == [1, 2, 1, 2]
