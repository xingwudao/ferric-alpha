#!/usr/bin/env python3
"""Generate Phase 03 synthetic golden data from public Alphalens APIs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import alphalens.performance as perf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "phase-03"
INPUT_PATH = GOLDEN_DIR / "input.json"
ALPHALENS_COMMIT = "77084f1e4c2c0be407e032d444fb19e4be4b0f37"
ALPHALENS_PERFORMANCE_SHA256 = (
    "b4e4d72d0f1fa8da6de5699136c5e1ca47b611cabcfd25ef50d6196e56d10b5f"
)


def verify_oracle_source() -> None:
    source = Path(perf.__file__).read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != ALPHALENS_PERFORMANCE_SHA256:
        raise SystemExit(
            "alphalens.performance.py does not match the pinned oracle source: "
            f"expected {ALPHALENS_PERFORMANCE_SHA256}, got {actual}"
        )


def load_input() -> dict[str, Any]:
    with INPUT_PATH.open() as handle:
        return json.load(handle)


def build_factor_data(raw: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(raw["rows"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.rename(columns={"forward_return_1D": "1D", "forward_return_3D": "3D"})
    return frame.set_index(["date", "asset"]).sort_index()


def build_weights(raw: dict[str, Any]) -> pd.Series:
    frame = pd.DataFrame(raw["weights"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.set_index(["date", "asset"])["weight"].sort_index()


def normalize(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def series_records(series: pd.Series, value_name: str) -> list[dict[str, Any]]:
    return [
        {"date": normalize(date), value_name: normalize(value)}
        for date, value in series.sort_index().items()
    ]


def position_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for date, row in frame.sort_index().iterrows():
        for asset in sorted(frame.columns):
            records.append(
                {
                    "date": normalize(date),
                    "asset": str(asset),
                    "position": normalize(row[asset]),
                }
            )
    return records


def capture_defect(call: Callable[[], Any]) -> dict[str, str]:
    try:
        call()
    except Exception as error:  # noqa: BLE001 - oracle records upstream behavior
        return {
            "exception": type(error).__name__,
            "message": str(error).splitlines()[0],
        }
    return {"exception": "none", "message": "call unexpectedly succeeded"}


def outputs(raw: dict[str, Any]) -> dict[str, Any]:
    factor_data = build_factor_data(raw)
    quantiles = factor_data["factor_quantile"]
    turnover = perf.quantile_turnover(quantiles, quantile=5, period=2)
    autocorrelation = perf.factor_rank_autocorrelation(factor_data, period=1)

    simple = pd.DataFrame(raw["simple_returns"])
    simple["date"] = pd.to_datetime(simple["date"], utc=True)
    simple_returns = simple.set_index("date")["return"]
    cumulative = perf.cumulative_returns(simple_returns)

    weights = build_weights(raw)
    raw_positions = perf.positions(weights, "2D")
    factor_position_values = perf.factor_positions(factor_data, "1D")

    defects = {
        "factor_cumulative_returns": capture_defect(
            lambda: perf.factor_cumulative_returns(factor_data, "1D")
        ),
        "create_pyfolio_input": capture_defect(
            lambda: perf.create_pyfolio_input(factor_data, "1D")
        ),
    }

    return {
        "turnover.json": series_records(turnover, "turnover"),
        "rank_autocorrelation.json": series_records(autocorrelation, "autocorrelation"),
        "cumulative_returns.json": series_records(cumulative, "cumulative_return"),
        "positions.json": position_records(raw_positions),
        "factor_positions.json": position_records(factor_position_values),
        "upstream_defects.json": defects,
    }


def manifest() -> dict[str, Any]:
    dependencies = ["numpy", "pandas", "empyrical", "alphalens"]
    return {
        "alphalens_commit": ALPHALENS_COMMIT,
        "alphalens_performance_sha256": ALPHALENS_PERFORMANCE_SHA256,
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in dependencies
        },
        "fixtures": [
            "input.json",
            "turnover.json",
            "rank_autocorrelation.json",
            "cumulative_returns.json",
            "positions.json",
            "factor_positions.json",
            "ferric_positions_contract.json",
            "upstream_defects.json",
        ],
        "period_semantics": (
            "Alphalens BDay oracle; Ferric uses explicit observed sessions."
        ),
        "null_policy": "NaN and infinite numeric values serialize as JSON null.",
    }


def rendered(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_or_check(values: dict[str, Any], check: bool) -> None:
    for filename, value in {"manifest.json": manifest(), **values}.items():
        path = GOLDEN_DIR / filename
        content = rendered(value)
        if check:
            if path.read_text() != content:
                raise SystemExit(f"{path} is not up to date")
        else:
            path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_oracle_source()
    write_or_check(outputs(load_input()), args.check)


if __name__ == "__main__":
    main()
