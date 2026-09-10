#!/usr/bin/env python3
"""Generate Ferric Alpha Phase 02 golden fixtures from public Alphalens APIs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import alphalens.performance as perf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "phase-02"
INPUT_PATH = GOLDEN_DIR / "input.json"
ALPHALENS_COMMIT = "77084f1e4c2c0be407e032d444fb19e4be4b0f37"
PERIOD_COLUMNS = {
    "forward_return_1D": "1D",
    "forward_return_5D": "5D",
    "forward_return_10D": "10D",
}


def load_input() -> dict[str, Any]:
    with INPUT_PATH.open() as handle:
        return json.load(handle)


def build_factor_data(raw: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(raw["rows"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.rename(
        columns={column: period for column, period in PERIOD_COLUMNS.items()}
    )
    frame = frame.set_index(["date", "asset"]).sort_index()
    return frame[["factor", "group", "factor_quantile", "1D", "5D", "10D"]]


def normalize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def date_label(value: Any) -> str:
    if isinstance(value, str):
        return pd.Timestamp(value).date().isoformat()
    return normalize_value(value)


def long_from_dataframe(
    frame: pd.DataFrame,
    value_name: str,
    extra_index_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    extra_index_names = extra_index_names or []
    index_names = ["date", *extra_index_names]
    named = frame.copy()
    named.index = named.index.set_names(index_names)
    stacked = named.stack(dropna=False)
    stacked.index = stacked.index.set_names([*index_names, "period"])

    records: list[dict[str, Any]] = []
    for key, value in stacked.items():
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {"date": date_label(key[0])}
        for name, part in zip(extra_index_names, key[1:-1], strict=True):
            row[name] = normalize_value(part)
        row["period"] = str(key[-1])
        row[value_name] = normalize_value(value)
        records.append(row)

    return sorted(
        records,
        key=lambda row: (
            row["date"],
            *(row[key] for key in extra_index_names),
            int(row["period"][:-1]),
        ),
    )


def series_by_period(series: pd.Series, value_name: str) -> list[dict[str, Any]]:
    records = [
        {"period": str(period), value_name: normalize_value(value)}
        for period, value in series.items()
    ]
    return sorted(records, key=lambda row: int(row["period"][:-1]))


def weights_to_records(weights: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (date, asset), row in weights.sort_index().iterrows():
        records.append(
            {
                "date": date_label(date),
                "asset": str(asset),
                "weight": normalize_value(row["factor"]),
            }
        )
    return records


def returns_to_records(returns: pd.DataFrame) -> list[dict[str, Any]]:
    return long_from_dataframe(returns, "factor_return")


def alpha_beta_to_records(alpha_beta: pd.DataFrame) -> list[dict[str, Any]]:
    alpha_beta = alpha_beta.rename(
        index={
            "Ann. alpha": "annualized_alpha",
            "beta": "beta",
        }
    )
    stacked = alpha_beta.stack(dropna=False)
    records = [
        {
            "period": str(period),
            "metric": str(metric),
            "value": normalize_value(value),
        }
        for (metric, period), value in stacked.items()
    ]
    return sorted(records, key=lambda row: (int(row["period"][:-1]), row["metric"]))


def quantile_return_records(
    mean_returns: pd.DataFrame,
    value_name: str,
) -> list[dict[str, Any]]:
    named = mean_returns.copy()
    named.index = named.index.set_names(["factor_quantile", "date"])
    stacked = named.stack(dropna=False)
    stacked.index = stacked.index.set_names(["factor_quantile", "date", "period"])

    records = [
        {
            "date": date_label(date),
            "factor_quantile": int(quantile),
            "period": str(period),
            value_name: normalize_value(value),
        }
        for (quantile, date, period), value in stacked.items()
    ]
    return sorted(
        records,
        key=lambda row: (row["date"], row["factor_quantile"], int(row["period"][:-1])),
    )


def spread_records(
    spread: pd.DataFrame, spread_error: pd.DataFrame
) -> list[dict[str, Any]]:
    spread_named = spread.copy()
    error_named = spread_error.copy()
    spread_named.index = spread_named.index.set_names(["date"])
    error_named.index = error_named.index.set_names(["date"])

    spread_stack = spread_named.stack(dropna=False)
    error_stack = error_named.stack(dropna=False)
    records = []
    for (date, period), value in spread_stack.items():
        records.append(
            {
                "date": date_label(date),
                "period": str(period),
                "mean_return_difference": normalize_value(value),
                "joint_std_error": normalize_value(error_stack.loc[(date, period)]),
            }
        )
    return sorted(records, key=lambda row: (row["date"], int(row["period"][:-1])))


def dependency_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scipy", "statsmodels", "empyrical", "alphalens"]
    return {name: importlib.metadata.version(name) for name in names}


def build_outputs(raw: dict[str, Any]) -> dict[str, Any]:
    factor_data = build_factor_data(raw)

    ic = perf.factor_information_coefficient(
        factor_data, group_adjust=False, by_group=False
    )
    mean_ic = perf.mean_information_coefficient(
        factor_data,
        group_adjust=False,
        by_group=False,
        by_time=None,
    )
    weights = perf.factor_weights(
        factor_data,
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
    )
    factor_returns = perf.factor_returns(
        factor_data,
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
        by_asset=False,
    )
    alpha_beta = perf.factor_alpha_beta(
        factor_data,
        returns=None,
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
    )
    mean_returns, std_err = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=True,
        group_adjust=False,
    )
    spread, spread_error = perf.compute_mean_returns_spread(
        mean_returns,
        upper_quant=5,
        lower_quant=1,
        std_err=std_err,
    )

    return {
        "ic.json": long_from_dataframe(ic, "ic"),
        "mean_ic.json": series_by_period(mean_ic, "mean_ic"),
        "weights.json": weights_to_records(weights.to_frame(name="factor")),
        "factor_returns.json": returns_to_records(factor_returns),
        "alpha_beta.json": alpha_beta_to_records(alpha_beta),
        "mean_return_by_quantile.json": {
            "mean_return": quantile_return_records(mean_returns, "mean_return"),
            "std_error": quantile_return_records(std_err, "std_error"),
        },
        "spread.json": spread_records(spread, spread_error),
    }


def manifest() -> dict[str, Any]:
    return {
        "alphalens_commit": ALPHALENS_COMMIT,
        "dependency_versions": dependency_versions(),
        "fixtures": [
            "input.json",
            "ic.json",
            "mean_ic.json",
            "weights.json",
            "factor_returns.json",
            "alpha_beta.json",
            "mean_return_by_quantile.json",
            "spread.json",
        ],
        "periods": ["1D", "5D", "10D"],
        "sort_keys": {
            "ic": ["date", "period"],
            "mean_ic": ["period"],
            "weights": ["date", "asset"],
            "factor_returns": ["date", "period"],
            "alpha_beta": ["period", "metric"],
            "mean_return_by_quantile": ["date", "factor_quantile", "period"],
            "spread": ["period"],
        },
        "null_policy": "NaN and infinite numeric values are serialized as JSON null.",
    }


def normalized_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_or_check(outputs: dict[str, Any], check: bool) -> None:
    all_outputs = {"manifest.json": manifest(), **outputs}
    for filename, data in all_outputs.items():
        path = GOLDEN_DIR / filename
        rendered = normalized_json(data)
        if check:
            existing = path.read_text()
            if existing != rendered:
                raise SystemExit(f"{path} is not up to date")
        else:
            path.write_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    raw = load_input()
    outputs = build_outputs(raw)
    write_or_check(outputs, check=args.check)


if __name__ == "__main__":
    main()
