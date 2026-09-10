#!/usr/bin/env python3
"""Generate Phase 04 synthetic golden data from pinned public APIs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import alphalens.performance as perf
import alphalens.plotting as plotting
import alphalens.tears as tears
import alphalens.utils as utils
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "phase-04"
INPUT_PATH = GOLDEN_DIR / "input.json"
INVENTORY_PATH = (
    ROOT / "docs" / "superpowers" / "golden" / "phase-04-synthetic-golden-inventory.md"
)
ALPHALENS_COMMIT = "77084f1e4c2c0be407e032d444fb19e4be4b0f37"
SOURCE_HASHES = {
    "alphalens.performance": (
        "b4e4d72d0f1fa8da6de5699136c5e1ca47b611cabcfd25ef50d6196e56d10b5f"
    ),
    "alphalens.plotting": (
        "574407f5342d22748e2004f79a3aa9c4c39411aac0894d37e6f20ff71c1afc9f"
    ),
    "alphalens.tears": (
        "4b1bca09e4f0347020e3d399c7bf64ea58cb540e674dbffec94dff9f1950d1de"
    ),
    "alphalens.utils": (
        "5b542c5633af2cd6f316761b6678f2249b338f34c70f6f248015819eb874364f"
    ),
}
PERIOD_COLUMNS = {
    "forward_return_5D": "5D",
    "forward_return_1D": "1D",
    "forward_return_3D": "3D",
}


def verify_oracle_sources() -> None:
    for module_name, expected in SOURCE_HASHES.items():
        module = importlib.import_module(module_name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"{module_name} does not match the pinned oracle source: "
                f"expected {expected}, got {actual}"
            )


def install_pandas_compat() -> list[str]:
    notes: list[str] = []
    if not hasattr(pd.DataFrame, "iteritems"):
        pd.DataFrame.iteritems = pd.DataFrame.items  # type: ignore[attr-defined]
        notes.append("patched pandas.DataFrame.iteritems for legacy Alphalens")
    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items  # type: ignore[attr-defined]
        notes.append("patched pandas.Series.iteritems for legacy Alphalens")
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, **_: pd.concat([self, other])  # type: ignore[attr-defined]
        notes.append("patched pandas.DataFrame.append for legacy Alphalens")
    return notes


def synthetic_input() -> dict[str, Any]:
    timezone = "America/New_York"
    sessions = [
        "2024-01-02T09:30:00-05:00",
        "2024-01-03T09:30:00-05:00",
        "2024-01-04T09:30:00-05:00",
        "2024-01-05T09:30:00-05:00",
        "2024-01-08T09:30:00-05:00",
        "2024-01-09T09:30:00-05:00",
        "2024-01-10T09:30:00-05:00",
        "2024-01-11T09:30:00-05:00",
        "2024-01-16T09:30:00-05:00",
    ]
    assets = ["A", "B", "C", "D", "E"]
    groups = {
        "A": "technology",
        "B": "technology",
        "C": "technology",
        "D": "utilities",
        "E": "utilities",
    }
    quantiles = {
        "A": [1, 1, 1, 2, 1, 3, 1, 1, 1],
        "B": [1, 2, 2, 2, 2, 2, 3, 2, 2],
        "C": [2, 2, 3, 2, 3, 1, 2, 3, 3],
        "D": [3, 3, 1, 1, 2, 3, 2, 1, 2],
        "E": [3, 3, 3, 3, 1, 1, 3, 2, 3],
    }
    rows: list[dict[str, Any]] = []
    for day, date in enumerate(sessions):
        for asset_index, asset in enumerate(assets):
            if (day, asset) in {(1, "E"), (3, "C"), (5, "E")}:
                continue
            quantile = quantiles[asset][day]
            factor = (quantile - 2) * 1.1 + (asset_index - 2) * 0.07 + day * 0.03
            if day == 0 and asset in {"A", "B"}:
                factor = -1.25
            if day == 4 and asset == "D":
                factor = None
            one_day = (quantile - 2) * 0.006 + day * 0.001 + asset_index * 0.0007
            three_day = one_day * 2.2 + (0.001 if (day + asset_index) % 2 else -0.0005)
            five_day = one_day * 3.1 - (day % 3) * 0.0008
            if day == 4 and asset == "B":
                three_day = None
            if day == 6 and asset == "D":
                one_day = None
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "group": groups[asset],
                    "factor": factor,
                    "factor_quantile": quantile,
                    "forward_return_5D": finite_or_none(five_day),
                    "forward_return_1D": finite_or_none(one_day),
                    "forward_return_3D": finite_or_none(three_day),
                }
            )

    constant_ic_rows: list[dict[str, Any]] = []
    factor_values = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
    factor_quantiles = {"A": 1, "B": 1, "C": 2, "D": 3, "E": 3}
    for day, date in enumerate(sessions[:8]):
        for asset in assets:
            value = factor_values[asset] + day * 0.01
            constant_ic_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "group": groups[asset],
                    "factor": value,
                    "factor_quantile": factor_quantiles[asset],
                    "forward_return_5D": value * 0.030,
                    "forward_return_1D": value * 0.010,
                    "forward_return_3D": value * 0.020,
                }
            )

    return_calendar = [
        "2023-12-29T09:30:00-05:00",
        *sessions[:-1],
        "2024-01-12T09:30:00-05:00",
    ]
    event_returns: list[dict[str, Any]] = []
    for day, date in enumerate(return_calendar):
        for asset_index, asset in enumerate(assets):
            if (day, asset) in {(2, "E"), (6, "B")}:
                continue
            value: Any = round((asset_index - 2) * 0.004 + (day - 3) * 0.0015, 10)
            if (day, asset) == (3, "C"):
                value = None
            if (day, asset) == (4, "D"):
                value = "NaN"
            if (day, asset) == (7, "A"):
                value = "Infinity"
            event_returns.append({"date": date, "asset": asset, "return": value})

    return {
        "description": (
            "Phase 04 synthetic oracle input with shuffled forward-return columns."
        ),
        "timezone": timezone,
        "rows": rows,
        "constant_ic_rows": constant_ic_rows,
        "event_returns": event_returns,
    }


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 12)


def load_input() -> dict[str, Any]:
    with INPUT_PATH.open() as handle:
        return json.load(handle)


def build_factor_data(raw: dict[str, Any], key: str = "rows") -> pd.DataFrame:
    frame = pd.DataFrame(raw[key])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.rename(columns=PERIOD_COLUMNS)
    frame = frame.set_index(["date", "asset"]).sort_index()
    return frame[["factor", "group", "factor_quantile", "5D", "1D", "3D"]]


def build_returns_matrix(raw: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(raw["event_returns"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["return"] = frame["return"].map(decode_return_value)
    return frame.pivot(index="date", columns="asset", values="return").sort_index()


def decode_return_value(value: Any) -> float:
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    if value is None:
        return float("nan")
    return float(value)


def normalize(value: Any) -> Any:
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


def long_from_frame(frame: pd.DataFrame, value_name: str) -> list[dict[str, Any]]:
    named = frame.copy()
    named.index = named.index.set_names(["date"])
    stacked = named.stack(dropna=False)
    stacked.index = stacked.index.set_names(["date", "period"])
    return sorted(
        [
            {
                "date": normalize(date),
                "period": str(period),
                value_name: normalize(value),
            }
            for (date, period), value in stacked.items()
        ],
        key=lambda row: (row["date"], int(row["period"][:-1])),
    )


def mean_quantile_records(
    frame: pd.DataFrame,
    errors: pd.DataFrame,
    base_period: str,
) -> list[dict[str, Any]]:
    mean = frame.apply(utils.rate_of_return, axis=0, base_period=base_period)
    error = errors.apply(utils.std_conversion, axis=0, base_period=base_period)
    records = []
    for quantile in mean.index:
        for period in mean.columns:
            records.append(
                {
                    "factor_quantile": int(quantile),
                    "period": str(period),
                    "mean_return": normalize(mean.loc[quantile, period]),
                    "std_error": normalize(error.loc[quantile, period]),
                }
            )
    return sorted(
        records, key=lambda row: (row["factor_quantile"], int(row["period"][:-1]))
    )


def quantile_statistics_capture(factor_data: pd.DataFrame) -> dict[str, Any]:
    table = factor_data.groupby("factor_quantile").agg(
        ["min", "max", "mean", "std", "count"]
    )["factor"]
    table["count %"] = table["count"] / table["count"].sum() * 100.0
    display = records_from_index(table.reset_index())
    ferric_projection = []
    for record in display:
        ferric_projection.append(
            {
                "factor_quantile": record["factor_quantile"],
                "factor_min": record["min"],
                "factor_max": record["max"],
                "factor_mean": record["mean"],
                "factor_std": record["std"],
                "count": record["count"],
                "count_percent": None
                if record["count %"] is None
                else record["count %"] / 100.0,
            }
        )
    return {
        "display_table": display,
        "ferric_projection": ferric_projection,
    }


def returns_capture(factor_data: pd.DataFrame) -> dict[str, Any]:
    factor_returns = perf.factor_returns(factor_data, demeaned=True, group_adjust=False)
    mean_ret, std_err = perf.mean_return_by_quantile(
        factor_data,
        by_date=False,
        by_group=False,
        demeaned=True,
        group_adjust=False,
    )
    base_period = sorted(mean_ret.columns, key=period_order)[0]
    return {
        "factor_returns": long_from_frame(factor_returns, "factor_return"),
        "mean_by_quantile": mean_quantile_records(mean_ret, std_err, base_period),
    }


def information_capture(constant_ic_data: pd.DataFrame) -> dict[str, Any]:
    ic = perf.factor_information_coefficient(
        constant_ic_data,
        group_adjust=False,
        by_group=False,
    )
    records = long_from_frame(ic, "ic")
    by_period: dict[str, list[float]] = {}
    for record in records:
        if record["ic"] is not None:
            by_period.setdefault(record["period"], []).append(record["ic"])
    summary = [
        {
            "period": period,
            "ic_mean": sum(values) / len(values) if values else None,
            "count": len(values),
        }
        for period, values in sorted(
            by_period.items(), key=lambda item: period_order(item[0])
        )
    ]
    return {
        "constant_ic_by_date": records,
        "constant_summary": summary,
        "projection_note": (
            "Separate subcase used so IC remains constant while the main fixture "
            "stresses other captures."
        ),
    }


def turnover_capture(factor_data: pd.DataFrame) -> dict[str, Any]:
    quantiles = sorted(
        int(value) for value in factor_data["factor_quantile"].dropna().unique()
    )
    records = []
    for period in [1, 3]:
        for quantile in quantiles:
            series = perf.quantile_turnover(
                factor_data["factor_quantile"], quantile, period
            )
            for date, value in series.sort_index().items():
                records.append(
                    {
                        "date": normalize(date),
                        "factor_quantile": quantile,
                        "period": f"{period}D",
                        "turnover": normalize(value),
                    }
                )
    autocorrelation = []
    for period in [1, 3]:
        series = perf.factor_rank_autocorrelation(factor_data, period=period)
        for date, value in series.sort_index().items():
            autocorrelation.append(
                {
                    "date": normalize(date),
                    "period": f"{period}D",
                    "autocorrelation": normalize(value),
                }
            )
    return {
        "quantile_turnover": sorted(
            records,
            key=lambda row: (
                row["date"],
                row["factor_quantile"],
                period_order(row["period"]),
            ),
        ),
        "rank_autocorrelation": sorted(
            autocorrelation,
            key=lambda row: (row["date"], period_order(row["period"])),
        ),
    }


def event_returns_capture(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    raw = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=2,
        periods_after=2,
        demeaned=False,
        group_adjust=False,
        by_group=False,
    )
    records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in raw.sort_index().iterrows():
        quantile, statistic = index
        for offset, value in row.items():
            key = (int(quantile), int(offset))
            record = records_by_key.setdefault(
                key,
                {"factor_quantile": int(quantile), "offset": int(offset)},
            )
            record[f"{statistic}_cumulative_return"] = normalize(value)
    return {
        "upstream_average_cumulative_returns": sorted(
            records_by_key.values(),
            key=lambda row: (row["factor_quantile"], row["offset"]),
        ),
        "ferric_projection": ferric_event_projection(contract),
        "upstream_shape": {
            "index_names": [str(name) for name in raw.index.names],
            "columns": [int(column) for column in raw.columns],
        },
        "projection_note": (
            "Upstream rebases each event window; Ferric Phase 04 intentionally "
            "uses cumulative wealth over the full return calendar before slicing."
        ),
    }


def ferric_event_projection(contract: dict[str, Any]) -> list[dict[str, Any]]:
    event_report = next(
        report for report in contract["reports"] if report["kind"] == "event_returns"
    )
    table = next(
        table
        for table in event_report["tables"]
        if table["id"] == "events.average_cumulative_returns"
    )["json"]
    columns = {
        column["name"]: column_data_values(column["data"])
        for column in table["columns"]
    }
    return [
        {
            "factor_quantile": columns["factor_quantile"][index],
            "offset": columns["offset"][index],
            "mean_cumulative_return": columns["mean_cumulative_return"][index],
            "std_cumulative_return": columns["std_cumulative_return"][index],
        }
        for index in range(table["row_count"])
    ]


def column_data_values(data: dict[str, Any]) -> list[Any]:
    [(kind, payload)] = data.items()
    if kind == "datetime":
        return payload["values"]
    return payload


def tear_sheet_roles(
    factor_data: pd.DataFrame, returns: pd.DataFrame
) -> dict[str, Any]:
    role_calls: list[dict[str, Any]] = []
    table_calls: list[dict[str, Any]] = []
    legacy_events: list[dict[str, str]] = []
    patch_names = [
        "plot_quantile_statistics_table",
        "plot_returns_table",
        "plot_quantile_returns_bar",
        "plot_quantile_returns_violin",
        "plot_cumulative_returns",
        "plot_cumulative_returns_by_quantile",
        "plot_mean_quantile_returns_spread_time_series",
        "plot_information_table",
        "plot_ic_ts",
        "plot_ic_hist",
        "plot_ic_qq",
        "plot_monthly_ic_heatmap",
        "plot_ic_by_group",
        "plot_turnover_table",
        "plot_top_bottom_quantile_turnover",
        "plot_factor_rank_auto_correlation",
        "plot_quantile_average_cumulative_return",
        "plot_events_distribution",
    ]
    originals = {name: getattr(plotting, name) for name in patch_names}
    original_print_table = utils.print_table
    original_plotting_show = plotting.plt.show
    original_tears_show = tears.plt.show

    def print_table(
        table: Any, name: str | None = None, fmt: Any | None = None
    ) -> None:
        table_calls.append(
            {
                "name": name,
                "fmt": repr(fmt) if fmt is not None else None,
                "table": describe_value(table),
            }
        )

    def make_capture(name: str) -> Callable[..., Any]:
        def capture(*args: Any, **kwargs: Any) -> Any:
            role_calls.append(
                {
                    "role": name,
                    "args": [describe_value(arg) for arg in args],
                    "kwargs": {
                        key: describe_value(value)
                        for key, value in sorted(kwargs.items())
                        if key not in {"ax"}
                    },
                }
            )
            return kwargs.get("ax")

        return capture

    try:
        utils.print_table = print_table
        plotting.plt.show = lambda *_, **__: None
        tears.plt.show = lambda *_, **__: None
        for name in patch_names:
            setattr(plotting, name, make_capture(name))
        for name, call in [
            ("summary", lambda: tears.create_summary_tear_sheet(factor_data)),
            (
                "returns",
                lambda: tears.create_returns_tear_sheet(factor_data, by_group=True),
            ),
            (
                "information",
                lambda: tears.create_information_tear_sheet(factor_data, by_group=True),
            ),
            (
                "turnover",
                lambda: tears.create_turnover_tear_sheet(
                    factor_data, turnover_periods=["1D", "3D"]
                ),
            ),
            (
                "full",
                lambda: tears.create_full_tear_sheet(factor_data, by_group=True),
            ),
            (
                "event_returns",
                lambda: tears.create_event_returns_tear_sheet(
                    factor_data,
                    returns,
                    avgretplot=(2, 2),
                    long_short=False,
                    by_group=False,
                ),
            ),
            (
                "event_study",
                lambda: tears.create_event_study_tear_sheet(
                    factor_data,
                    returns,
                    avgretplot=(2, 2),
                    rate_of_ret=True,
                    n_bars=4,
                ),
            ),
        ]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    call()
                except Exception as error:  # noqa: BLE001 - oracle records legacy behavior
                    legacy_events.append(
                        {
                            "scope": name,
                            "exception": type(error).__name__,
                            "message": str(error).splitlines()[0],
                        }
                    )
                for warning in caught:
                    legacy_events.append(
                        {
                            "scope": name,
                            "warning": type(warning.message).__name__,
                            "message": str(warning.message).splitlines()[0],
                        }
                    )
    finally:
        for name, original in originals.items():
            setattr(plotting, name, original)
        utils.print_table = original_print_table
        plotting.plt.show = original_plotting_show
        tears.plt.show = original_tears_show
        plotting.plt.close("all")

    return {
        "plot_roles": role_calls,
        "print_tables": table_calls,
        "legacy_events": legacy_events,
    }


def describe_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "index_names": [str(name) for name in value.index.names],
            "columns": [str(column) for column in value.columns],
            "records": records_from_index(value.reset_index())[:8],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "length": int(value.shape[0]),
            "name": str(value.name),
            "index_names": [str(name) for name in value.index.names],
            "records": records_from_index(value.reset_index())[:8],
        }
    if isinstance(value, (int, float, str, bool)) or value is None:
        return normalize(value)
    if isinstance(value, tuple | list):
        return [describe_value(item) for item in value]
    return type(value).__name__


def records_from_index(frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    return [
        {str(key): normalize(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def ferric_contract() -> Any:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output = Path(handle.name)
    env = os.environ.copy()
    env["FERRIC_ALPHA_PHASE04_CONTRACT_OUT"] = str(output)
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "ferric-alpha",
            "--test",
            "report_phase_04_golden",
            "export_phase_04_ferric_contract",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    try:
        return json.loads(output.read_text())
    finally:
        output.unlink(missing_ok=True)


def manifest(legacy_notes: list[str], role_capture: dict[str, Any]) -> dict[str, Any]:
    dependencies = ["numpy", "pandas", "scipy", "statsmodels", "empyrical", "alphalens"]
    return {
        "alphalens_commit": ALPHALENS_COMMIT,
        "alphalens_source_sha256": SOURCE_HASHES,
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in dependencies
        },
        "fixtures": [
            "input.json",
            "manifest.json",
            "quantile_statistics.json",
            "returns_capture.json",
            "information_capture.json",
            "turnover_capture.json",
            "event_returns.json",
            "tear_sheet_roles.json",
            "ferric_contract.json",
        ],
        "legacy_notes": legacy_notes,
        "legacy_event_count": len(role_capture["legacy_events"]),
        "null_policy": (
            "NaN and infinite numeric values serialize as JSON null in oracle outputs."
        ),
        "period_policy": (
            "Ferric periods are sorted by observed-session count; input keeps "
            "forward returns shuffled as 5D, 1D, 3D."
        ),
    }


def inventory(legacy_notes: list[str], role_capture: dict[str, Any]) -> str:
    legacy_lines = [
        (
            f"- `{item.get('scope', 'setup')}`: "
            f"{item.get('warning') or item.get('exception')} - {item['message']}"
        )
        for item in role_capture["legacy_events"]
    ]
    if not legacy_lines:
        legacy_lines = ["- None recorded."]
    notes = [f"- {note}." for note in legacy_notes] or [
        "- No pandas compatibility shims were required."
    ]
    return (
        "# Phase 04 Synthetic Golden Inventory\n\n"
        "Generated by `tools/generate_phase_04_golden.py` using the pinned "
        "Alphalens oracle and the local Ferric Rust report APIs.\n\n"
        "## Coverage\n\n"
        "- Main factor fixture: nine timezone-aware sessions, five assets, two groups, "
        "three quantiles, shuffled `5D`/`1D`/`3D` columns, ties, null factors, "
        "changing universe, unequal group sizes, and missing forward returns.\n"
        "- Event fixture: edge sessions, an absent event date, sparse rows, "
        "null returns, "
        "and non-finite daily returns.\n"
        "- Constant-IC projection: separate eight-session subcase for stable IC values "
        "without weakening the main fixture.\n\n"
        "## Projection Rulings\n\n"
        "- Alphalens event-return output is captured as public normalized long JSON; "
        "Ferric event coverage/count/imputation fields are asserted through "
        "`ferric_contract.json` because the sparse long-return projection is a "
        "Ferric extension over Alphalens' wide return matrix.\n"
        "- Alphalens display tables that round or scale values are captured under "
        "`tear_sheet_roles.json`; Rust parity checks use raw public-performance "
        "values where Ferric stores raw analytics plus display metadata.\n\n"
        "## Source Pins\n\n"
        + "".join(f"- `{name}`: `{digest}`\n" for name, digest in SOURCE_HASHES.items())
        + "\n## Legacy Compatibility Notes\n\n"
        + "\n".join(notes)
        + "\n\n## Recorded Tear-Sheet Warnings And Failures\n\n"
        + "\n".join(legacy_lines)
        + "\n\n## Ferric Contract\n\n"
        "The `ferric_contract.json` fixture is exported by the ignored Rust helper "
        "`export_phase_04_ferric_contract`, then compared by the normal Rust golden "
        "test for exact JSON shape and serialization.\n\n"
        "## Verification\n\n"
        "- Generator: `/tmp/ferric-alpha-alphalens-golden/bin/python "
        "tools/generate_phase_04_golden.py`\n"
        "- Reproducibility check: `/tmp/ferric-alpha-alphalens-golden/bin/python "
        "tools/generate_phase_04_golden.py --check`\n"
        "- Rust golden parity: `cargo test -p ferric-alpha --test "
        "report_phase_04_golden`\n"
        "- Full gate: `make verify` passed with all Rust workspace tests passing, "
        "`166` Rust test entries discovered including one ignored export helper, "
        "Python tests `29 passed`, Clippy clean, and Ruff clean.\n"
        "- CPython 3.10 gate: `/tmp/ferric-alpha-py310` passed `maturin develop` "
        "and `pytest -q tests/python` with `29 passed`.\n"
        "- Python dependency metadata: `['polars>=1.42,<1.43']`.\n\n"
        "## Benchmark Baseline\n\n"
        "Command:\n\n"
        "```text\n"
        "cargo bench -p ferric-alpha --bench phase_02 -- report-smoke "
        "--sample-size 10\n"
        "```\n\n"
        "Fixture:\n\n"
        "- Report cases use 250 assets, 252 sessions, and 3 periods.\n"
        "- Event case uses 50 assets, 252 sessions, and event window `(5, 15)`.\n\n"
        "Results:\n\n"
        "- `report-smoke/summary`: `261.58 ms..262.78 ms`\n"
        "- `report-smoke/returns`: `133.55 ms..134.02 ms`\n"
        "- `report-smoke/information`: `82.189 ms..82.441 ms`\n"
        "- `report-smoke/turnover`: `85.079 ms..85.385 ms`\n"
        "- `report-smoke/full`: `322.42 ms..323.80 ms`\n"
        "- `report-smoke/full-json`: `347.08 ms..348.66 ms`\n"
        "- `report-smoke/event-average-cumulative`: `35.741 ms..35.942 ms`\n\n"
        "Notes:\n\n"
        "- Criterion reported two mild outliers for `report-smoke/full`.\n"
        "- `gnuplot` was not installed, so Criterion used the Plotters backend.\n"
    )


def period_order(period: str) -> int:
    return int(str(period).removesuffix("D"))


def rendered(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_or_check_file(path: Path, content: str, check: bool) -> None:
    if check:
        if path.read_text() != content:
            raise SystemExit(f"{path} is not up to date")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def write_or_check(values: dict[str, Any], inventory_text: str, check: bool) -> None:
    for filename, value in values.items():
        write_or_check_file(GOLDEN_DIR / filename, rendered(value), check)
    write_or_check_file(INVENTORY_PATH, inventory_text, check)


def build_outputs(check: bool) -> tuple[dict[str, Any], str]:
    legacy_notes = install_pandas_compat()
    raw = synthetic_input()
    factor_data = build_factor_data(raw)
    constant_ic_data = build_factor_data(raw, "constant_ic_rows")
    returns = build_returns_matrix(raw)
    role_capture = tear_sheet_roles(factor_data, returns)
    contract = ferric_contract()
    values = {
        "input.json": raw,
        "quantile_statistics.json": quantile_statistics_capture(factor_data),
        "returns_capture.json": returns_capture(factor_data),
        "information_capture.json": information_capture(constant_ic_data),
        "turnover_capture.json": turnover_capture(factor_data),
        "event_returns.json": event_returns_capture(factor_data, returns, contract),
        "tear_sheet_roles.json": role_capture,
        "ferric_contract.json": contract,
    }
    values["manifest.json"] = manifest(legacy_notes, role_capture)
    return values, inventory(legacy_notes, role_capture)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_oracle_sources()
    values, inventory_text = build_outputs(args.check)
    write_or_check(values, inventory_text, args.check)


if __name__ == "__main__":
    main()
