#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ferric", "alphalens"], required=True)
    parser.add_argument("--assets", type=int, default=250)
    parser.add_argument("--sessions", type=int, default=252)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if args.assets < 10:
        raise SystemExit("--assets must be at least 10")
    if args.sessions < 3:
        raise SystemExit("--sessions must be at least 3")
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")

    if args.engine == "ferric":
        result = run_ferric(args.assets, args.sessions, args.iterations)
    else:
        result = run_alphalens(args.assets, args.sessions, args.iterations)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_ferric(assets: int, sessions: int, iterations: int) -> dict[str, Any]:
    import ferric_alpha as fa
    import polars as pl

    frame = ferric_frame(assets, sessions, pl)
    benchmark = {
        "factor_information_coefficient": lambda: (
            fa.performance.factor_information_coefficient(frame)
        ),
        "factor_weights": lambda: fa.performance.factor_weights(frame),
        "factor_returns": lambda: fa.performance.factor_returns(frame),
        "mean_return_by_quantile": lambda: fa.performance.mean_return_by_quantile(
            frame
        ),
        "quantile_turnover": lambda: fa.performance.quantile_turnover(frame, 5, 1),
        "factor_rank_autocorrelation": lambda: (
            fa.performance.factor_rank_autocorrelation(frame, 1)
        ),
    }
    return run_benchmarks(
        "ferric-alpha",
        getattr(fa, "__version__", "unknown"),
        {"polars": pl.__version__},
        assets,
        sessions,
        iterations,
        benchmark,
        lambda value: value.shape[0],
    )


def run_alphalens(assets: int, sessions: int, iterations: int) -> dict[str, Any]:
    import alphalens
    import numpy as np
    import pandas as pd
    from alphalens import performance

    frame = alphalens_frame(assets, sessions, np, pd)
    quantile_factor = frame["factor_quantile"]
    benchmark = {
        "factor_information_coefficient": lambda: (
            performance.factor_information_coefficient(frame)
        ),
        "factor_weights": lambda: performance.factor_weights(frame),
        "factor_returns": lambda: performance.factor_returns(frame),
        "mean_return_by_quantile": lambda: performance.mean_return_by_quantile(frame)[
            0
        ],
        "quantile_turnover": lambda: performance.quantile_turnover(
            quantile_factor, 5, 1
        ),
        "factor_rank_autocorrelation": lambda: performance.factor_rank_autocorrelation(
            frame, 1
        ),
    }
    return run_benchmarks(
        "alphalens",
        getattr(alphalens, "__version__", "unknown"),
        {"numpy": np.__version__, "pandas": pd.__version__},
        assets,
        sessions,
        iterations,
        benchmark,
        output_rows,
    )


def ferric_frame(assets: int, sessions: int, pl: Any) -> Any:
    dates = []
    asset_names = []
    groups = []
    factors = []
    quantiles = []
    returns_1d = []
    returns_5d = []
    start = 1_704_067_200_000

    for session in range(sessions):
        date = start + session * 86_400_000
        for asset in range(assets):
            factor, ret_1d, ret_5d = synthetic_values(asset, session, assets)
            dates.append(date)
            asset_names.append(f"A{asset:05d}")
            groups.append(f"G{asset % 5}")
            factors.append(factor)
            quantiles.append(asset % 5 + 1)
            returns_1d.append(ret_1d)
            returns_5d.append(ret_5d)

    return pl.DataFrame(
        {
            "date": dates,
            "asset": asset_names,
            "factor": factors,
            "group": groups,
            "factor_quantile": quantiles,
            "forward_return_1D": returns_1d,
            "forward_return_5D": returns_5d,
        }
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
        pl.col("factor_quantile").cast(pl.UInt32),
    )


def alphalens_frame(assets: int, sessions: int, np: Any, pd: Any) -> Any:
    dates = []
    asset_names = []
    groups = []
    factors = []
    quantiles = []
    returns_1d = []
    returns_5d = []
    start = pd.Timestamp("2024-01-01")

    for session in range(sessions):
        date = start + pd.Timedelta(days=session)
        for asset in range(assets):
            factor, ret_1d, ret_5d = synthetic_values(asset, session, assets)
            dates.append(date)
            asset_names.append(f"A{asset:05d}")
            groups.append(f"G{asset % 5}")
            factors.append(factor)
            quantiles.append(asset % 5 + 1)
            returns_1d.append(ret_1d)
            returns_5d.append(ret_5d)

    index = pd.MultiIndex.from_arrays([dates, asset_names], names=["date", "asset"])
    return pd.DataFrame(
        {
            "factor": np.asarray(factors, dtype="float64"),
            "group": groups,
            "factor_quantile": np.asarray(quantiles, dtype="int64"),
            "1D": np.asarray(returns_1d, dtype="float64"),
            "5D": np.asarray(returns_5d, dtype="float64"),
        },
        index=index,
    )


def synthetic_values(
    asset: int, session: int, assets: int
) -> tuple[float, float, float]:
    centered = asset - assets / 2.0
    wave = ((asset * 17 + session * 13) % 101) / 1000.0
    factor = centered + (session % 7) * 0.01 + wave
    ret_1d = centered * 0.00001 + (session % 11) * 0.00001 - wave * 0.0001
    ret_5d = centered * 0.00004 + (session % 17) * 0.00002 - wave * 0.0002
    return factor, ret_1d, ret_5d


def run_benchmarks(
    engine: str,
    version: str,
    dependencies: dict[str, str],
    assets: int,
    sessions: int,
    iterations: int,
    benchmark: dict[str, Any],
    consume: Any,
) -> dict[str, Any]:
    results = {}
    for name, func in benchmark.items():
        value = func()
        rows = consume(value)
        timings = []
        for _ in range(iterations):
            started = time.perf_counter()
            value = func()
            rows = consume(value)
            timings.append(time.perf_counter() - started)
        median_seconds = statistics.median(timings)
        results[name] = {
            "median_ms": round(median_seconds * 1000.0, 3),
            "rows": rows,
        }

    return {
        "assets": assets,
        "dependencies": dependencies,
        "engine": engine,
        "iterations": iterations,
        "python": sys.version.split()[0],
        "rows": assets * sessions,
        "sessions": sessions,
        "system": platform.platform(),
        "version": version,
        "results": results,
    }


def output_rows(value: Any) -> int:
    if hasattr(value, "shape"):
        return int(value.shape[0])
    return int(len(value))


if __name__ == "__main__":
    main()
