from __future__ import annotations

import inspect
from datetime import datetime

import polars as pl
import pytest
from ferric_alpha import performance


def _factor_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
                datetime(2024, 1, 2),
            ],
            "asset": ["A", "B", "C", "A", "B", "C"],
            "group": ["g1", "g1", "g2", "g1", "g1", "g2"],
            "factor": [-1.0, 0.0, 1.0, 1.0, 0.0, -1.0],
            "factor_quantile": [1, 3, 5, 5, 3, 1],
            "forward_return_1D": [-0.01, 0.0, 0.01, -0.02, 0.0, 0.02],
            "forward_return_5D": [-0.05, 0.0, 0.05, -0.10, 0.0, 0.10],
            "forward_return_10D": [-0.10, 0.0, 0.10, -0.20, 0.0, 0.20],
        }
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
        pl.col("factor_quantile").cast(pl.UInt32),
    )


def test_performance_module_imports_and_signatures() -> None:
    assert list(
        inspect.signature(performance.factor_information_coefficient).parameters
    ) == [
        "factor_data",
        "group_adjust",
        "by_group",
    ]
    assert list(
        inspect.signature(performance.compute_mean_returns_spread).parameters
    ) == [
        "mean_returns",
        "upper_quant",
        "lower_quant",
        "std_err",
    ]


def test_performance_wrappers_return_polars_frames() -> None:
    frame = _factor_data()
    mean_returns = performance.mean_return_by_quantile(frame, by_date=True)

    outputs = [
        performance.factor_information_coefficient(frame),
        performance.mean_information_coefficient(frame, by_time="daily"),
        performance.factor_weights(frame),
        performance.factor_returns(frame),
        performance.factor_returns(frame, by_asset=True),
        performance.factor_alpha_beta(frame),
        mean_returns,
        performance.compute_mean_returns_spread(mean_returns, 5, 1),
    ]

    assert all(isinstance(output, pl.DataFrame) for output in outputs)


def test_performance_maps_invalid_options_to_value_error() -> None:
    with pytest.raises(ValueError, match="invalid option by_time"):
        performance.mean_information_coefficient(_factor_data(), by_time="quarterly")


def test_phase_03_performance_signatures() -> None:
    assert list(inspect.signature(performance.quantile_turnover).parameters) == [
        "quantile_factor",
        "quantile",
        "period",
    ]
    assert list(
        inspect.signature(performance.factor_rank_autocorrelation).parameters
    ) == [
        "factor_data",
        "period",
    ]
    assert list(inspect.signature(performance.positions).parameters) == [
        "weights",
        "period",
        "sessions",
        "freq",
    ]
    assert list(inspect.signature(performance.create_pyfolio_input).parameters) == [
        "factor_data",
        "period",
        "capital",
        "long_short",
        "group_neutral",
        "equal_weight",
        "quantiles",
        "groups",
        "benchmark_period",
        "sessions",
    ]


def test_phase_03_wrappers_return_polars_data() -> None:
    frame = _factor_data()
    sessions = frame.select("date").unique().sort("date")
    weights = performance.factor_weights(frame)
    return_series = (
        frame.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.Series("return", [0.1, -0.1]))
    )

    outputs = [
        performance.quantile_turnover(frame, 5),
        performance.factor_rank_autocorrelation(frame),
        performance.cumulative_returns(return_series),
        performance.positions(weights, 1, sessions=sessions),
        performance.positions(weights, 1, freq=sessions),
        performance.factor_cumulative_returns(frame, 1),
        performance.factor_positions(frame, 1, sessions=sessions),
    ]
    portfolio = performance.create_pyfolio_input(frame, 1, sessions=sessions)

    assert all(isinstance(output, pl.DataFrame) for output in outputs)
    assert isinstance(portfolio.returns, pl.DataFrame)
    assert isinstance(portfolio.positions, pl.DataFrame)
    assert isinstance(portfolio.benchmark, pl.DataFrame)


def test_phase_03_invalid_period_maps_to_value_error() -> None:
    with pytest.raises(ValueError, match="period count must be positive"):
        performance.factor_rank_autocorrelation(_factor_data(), period=0)

    with pytest.raises(ValueError, match="invalid option period"):
        performance.factor_rank_autocorrelation(_factor_data(), period=-1)

    with pytest.raises(ValueError, match="invalid option quantile"):
        performance.quantile_turnover(_factor_data(), quantile=-1)

    result = performance.factor_rank_autocorrelation(_factor_data(), period=2**32 - 1)
    assert result["autocorrelation"].null_count() == result.height

    for value in [2**32, 10**100]:
        with pytest.raises(ValueError, match="invalid option period"):
            performance.factor_rank_autocorrelation(_factor_data(), period=value)


def test_phase_03_pyfolio_getters_clone_and_optional_benchmark() -> None:
    portfolio = performance.create_pyfolio_input(
        _factor_data(), 1, benchmark_period=None
    )

    assert portfolio.positions is not portfolio.positions
    assert portfolio.returns is not portfolio.returns
    assert portfolio.benchmark is None


def test_create_pyfolio_input_accepts_alphalens_benchmark_period_label() -> None:
    numeric = performance.create_pyfolio_input(_factor_data(), 1, benchmark_period=1)
    labeled = performance.create_pyfolio_input(_factor_data(), 1, benchmark_period="1D")

    assert labeled.benchmark.equals(numeric.benchmark)


def test_portfolio_helpers_accept_alphalens_period_labels() -> None:
    frame = _factor_data()

    assert performance.factor_cumulative_returns(frame, "1D").equals(
        performance.factor_cumulative_returns(frame, 1)
    )
    assert performance.factor_positions(frame, "1D").equals(
        performance.factor_positions(frame, 1)
    )

    labeled = performance.create_pyfolio_input(frame, "1D", benchmark_period="1D")
    numeric = performance.create_pyfolio_input(frame, 1, benchmark_period=1)
    assert labeled.returns.equals(numeric.returns)
    assert labeled.positions.equals(numeric.positions)
    assert labeled.benchmark.equals(numeric.benchmark)


def test_positions_accepts_alphalens_freq_alias() -> None:
    weights = performance.factor_weights(_factor_data())
    sessions = _factor_data().select("date").unique().sort("date")

    assert performance.positions(weights, 1, freq=sessions).equals(
        performance.positions(weights, 1, sessions=sessions)
    )
    with pytest.raises(ValueError, match="sessions and freq"):
        performance.positions(weights, 1, sessions=sessions, freq=sessions)


def test_phase_03_positions_reject_reserved_cash_asset() -> None:
    weights = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)],
            "asset": ["cash"],
            "weight": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    with pytest.raises(ValueError, match="reserved"):
        performance.positions(weights, 1)


def test_phase_03_python_session_and_capital_contracts() -> None:
    frame = _factor_data()
    weights = performance.factor_weights(frame)
    bad_dtype_sessions = pl.DataFrame({"date": [1, 2]})
    extra_column_sessions = frame.select("date").unique().with_columns(extra=pl.lit(1))

    with pytest.raises(ValueError, match="expected Datetime"):
        performance.positions(weights, 1, sessions=bad_dtype_sessions)
    with pytest.raises(ValueError, match="exactly one date column"):
        performance.positions(weights, 1, sessions=extra_column_sessions)
    with pytest.raises(ValueError, match="invalid option capital"):
        performance.create_pyfolio_input(frame, 1, capital=float("inf"))

    scaled = performance.create_pyfolio_input(
        frame, 1, capital=1_000.0, benchmark_period=None
    )
    daily_equity = scaled.positions.group_by("date").agg(pl.col("position").sum())
    assert daily_equity.sort("date")["position"].to_list() == pytest.approx(
        [1_010.0, 989.8]
    )
