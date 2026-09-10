"""Run a complete factor analysis with deterministic synthetic equity data."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import ferric_alpha as fa
import polars as pl

N_ASSETS = 20
GROUPS = ("Technology", "Financials", "Healthcare", "Energy")
N_BUSINESS_DAYS = 80
PERIODS = (1, 5, 10)
N_QUANTILES = 5


def business_days(start: datetime, count: int) -> list[datetime]:
    days: list[datetime] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def make_example_data() -> tuple[pl.DataFrame, pl.DataFrame, dict[str, str]]:
    """Create a known positive factor signal and matching price history."""
    rng = random.Random(1234)
    assets = [f"ASSET_{index:02d}" for index in range(N_ASSETS)]
    group_by = {
        asset: GROUPS[index % len(GROUPS)] for index, asset in enumerate(assets)
    }
    quality = {
        asset: 2.0 * (index - (N_ASSETS - 1) / 2) / (N_ASSETS - 1)
        for index, asset in enumerate(assets)
    }
    group_drift = {
        "Technology": 0.0001,
        "Financials": -0.0001,
        "Healthcare": 0.0002,
        "Energy": -0.0002,
    }
    prices_by_asset = {
        asset: 100.0 * (1.0 + 0.01 * index) for index, asset in enumerate(assets)
    }
    factor_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []

    for day_index, day in enumerate(
        business_days(datetime(2023, 1, 2), N_BUSINESS_DAYS)
    ):
        market_return = 0.0003 + 0.0001 * math.sin(day_index / 10.0)
        for asset_index, asset in enumerate(assets):
            factor_rows.append(
                {
                    "date": day,
                    "asset": asset,
                    "factor": quality[asset]
                    + 0.1 * math.sin(day_index / 4.0 + asset_index * 0.7)
                    + rng.gauss(0.0, 0.05),
                }
            )
            daily_return = (
                market_return
                + 0.01 * quality[asset]
                + group_drift[group_by[asset]]
                + rng.gauss(0.0, 0.01)
            )
            prices_by_asset[asset] *= 1.0 + daily_return
            price_rows.append(
                {
                    "date": day,
                    "asset": asset,
                    "price": prices_by_asset[asset],
                }
            )

    factor = pl.DataFrame(factor_rows).with_columns(pl.col("date").cast(pl.Datetime))
    prices = pl.DataFrame(price_rows).with_columns(pl.col("date").cast(pl.Datetime))
    return factor, prices, group_by


def values_by_period(frame: pl.DataFrame, column: str) -> dict[str, float]:
    return {row["period"]: row[column] for row in frame.to_dicts()}


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    factor, prices, group_by = make_example_data()
    clean = fa.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=PERIODS,
        quantiles=N_QUANTILES,
        groupby=group_by,
    )
    factor_data = clean.frame

    mean_ic_frame = fa.performance.mean_information_coefficient(factor_data)
    quantile_returns = fa.performance.mean_return_by_quantile(factor_data)
    spread_frame = fa.performance.compute_mean_returns_spread(
        quantile_returns, N_QUANTILES, 1
    )
    weights = fa.performance.factor_weights(factor_data)
    factor_returns = fa.performance.factor_returns(factor_data)
    alpha_beta = fa.performance.factor_alpha_beta(factor_data)
    turnover = fa.performance.quantile_turnover(
        factor_data.select("date", "asset", "factor_quantile"),
        N_QUANTILES,
        period=1,
    )
    rank_autocorrelation = fa.performance.factor_rank_autocorrelation(
        factor_data, period=1
    )

    report = fa.tears.create_full_tear_sheet_data(factor_data)
    report_path = output_dir / "factor-report.html"
    fa.plotting.render(report).save(report_path)

    mean_ic = values_by_period(mean_ic_frame, "mean_ic")
    spread = values_by_period(spread_frame, "mean_return_difference")
    one_day_quantiles = (
        quantile_returns.filter(pl.col("period") == "1D")
        .sort("factor_quantile")
        .get_column("mean_return")
        .to_list()
    )
    gross_exposure = (
        weights.group_by("date")
        .agg(pl.col("weight").abs().sum().alias("gross"))
        .get_column("gross")
    )

    loss = clean.loss
    results = {
        "case": {
            "assets": N_ASSETS,
            "business_days": N_BUSINESS_DAYS,
            "groups": len(GROUPS),
            "periods": [f"{period}D" for period in PERIODS],
        },
        "loss": {
            "input_rows": loss.input_rows,
            "output_rows": loss.output_rows,
            "forward_return_loss": loss.forward_return_loss,
            "quantile_loss": loss.quantile_loss,
        },
        "mean_ic": mean_ic,
        "spread": spread,
        "alpha_beta": alpha_beta.to_dicts(),
        "mean_top_quantile_turnover": turnover.get_column("turnover").mean(),
        "mean_rank_autocorrelation": rank_autocorrelation.get_column(
            "autocorrelation"
        ).mean(),
        "factor_return_rows": factor_returns.height,
        "sanity": {
            "positive_mean_ic": all(value > 0.0 for value in mean_ic.values()),
            "positive_top_bottom_spread": all(value > 0.0 for value in spread.values()),
            "monotonic_one_day_quantile_returns": all(
                lower < upper
                for lower, upper in zip(
                    one_day_quantiles, one_day_quantiles[1:], strict=False
                )
            ),
            "normalized_gross_exposure": all(
                abs(value - 1.0) < 1e-12 for value in gross_exposure
            ),
        },
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quickstart-output"),
        help="directory for results.json and factor-report.html",
    )
    args = parser.parse_args()
    results = run(args.output_dir)
    print(f"Mean IC: {results['mean_ic']}")
    print(f"Top-minus-bottom spread: {results['spread']}")
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'factor-report.html'}")


if __name__ == "__main__":
    main()
