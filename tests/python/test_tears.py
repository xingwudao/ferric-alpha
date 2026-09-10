from __future__ import annotations

import importlib
import inspect
import json
import sys
from datetime import datetime, timedelta

import polars as pl
import pytest
from ferric_alpha import performance, tears, utils

BLOCKED_IMPORTS = {"pandas", "scipy", "statsmodels", "matplotlib", "seaborn"}


def _factor_data(days: int = 24) -> pl.DataFrame:
    rows = []
    assets = ["a", "b", "c", "d"]
    for day in range(days):
        date = datetime(2024, 1, 1) + timedelta(days=day)
        rotation = day % 4
        for asset_index, asset in enumerate(assets):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "group": "g1" if asset_index < 2 else "g2",
                    "factor": float(asset_index + 1),
                    "factor_quantile": asset_index + 1,
                    "forward_return_1D": (rotation + asset_index + 1) * 0.01,
                    "forward_return_3D": (rotation + asset_index + 1) * 0.03,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms", "UTC")),
        pl.col("factor_quantile").cast(pl.UInt32),
    )


def _event_factor_data() -> pl.DataFrame:
    jan_1 = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "date": [
                jan_1,
                jan_1,
                jan_1 + timedelta(days=1),
                jan_1 + timedelta(days=1),
            ],
            "asset": ["A", "B", "A", "B"],
            "group": ["g1", "g1", "g1", "g1"],
            "factor": [-1.0, 1.0, -0.5, 0.5],
            "factor_quantile": [1, 2, 1, 2],
            "forward_return_1D": [-0.01, 0.02, -0.02, 0.03],
        }
    ).with_columns(
        pl.col("date").cast(pl.Datetime("ms", "UTC")),
        pl.col("factor_quantile").cast(pl.UInt32),
    )


def _event_returns_data() -> pl.DataFrame:
    jan_1 = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "date": [
                jan_1,
                jan_1,
                jan_1 + timedelta(days=1),
                jan_1 + timedelta(days=1),
                jan_1 + timedelta(days=2),
            ],
            "asset": ["A", "B", "A", "B", "A"],
            "return": [0.10, 0.20, None, 0.00, 0.10],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms", "UTC")))


def _upstream_tears_prices(
    days: int = 60, *, timezone: str | None = None
) -> pl.DataFrame:
    assets = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for offset in range(days):
        date = datetime(2015, 1, 10) + timedelta(days=offset)
        values = [
            1.25 ** (offset + 1),
            1.50 ** (offset + 1),
            1.00 ** (offset + 1),
            0.50 ** (offset + 1),
            1.50 ** (offset + 1),
            1.00 ** (offset + 1),
        ]
        for asset, price in zip(assets, values, strict=True):
            rows.append((date, asset, price))
    dtype = pl.Datetime("ms", timezone) if timezone else pl.Datetime("ms")
    return pl.DataFrame(
        rows,
        schema=["date", "asset", "price"],
        orient="row",
    ).with_columns(pl.col("date").cast(dtype))


def _upstream_tears_returns(
    days: int = 60, *, timezone: str | None = None
) -> pl.DataFrame:
    prices = _upstream_tears_prices(days, timezone=timezone)
    return (
        prices.sort("asset", "date")
        .with_columns(
            ((pl.col("price") / pl.col("price").shift(1).over("asset")) - 1.0).alias(
                "return"
            )
        )
        .filter(pl.col("return").is_not_null())
        .select("date", "asset", "return")
        .sort("date", "asset")
    )


def _upstream_tears_factor(
    *,
    event: bool = False,
    timezone: str | None = None,
) -> pl.DataFrame:
    factor_rows = [
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, 4, 2, 1, None, None],
        [3, None, None, 1, 4, 2],
        [3, None, None, 1, 4, 2],
    ]
    event_rows = [
        [1, None, None, None, None, None],
        [4, None, None, 7, None, None],
        [None, None, None, None, None, None],
        [None, 3, None, 2, None, None],
        [1, None, None, None, None, None],
        [None, None, 2, None, None, None],
        [None, None, None, 2, None, None],
        [None, None, None, 1, None, None],
        [2, None, None, None, None, None],
        [None, None, None, None, 5, None],
        [None, None, None, 2, None, None],
        [None, None, None, None, None, None],
        [2, None, None, None, None, None],
        [None, None, None, None, None, 5],
        [None, None, None, 1, None, None],
        [None, None, None, None, 4, None],
        [5, None, None, 4, None, None],
        [None, None, None, 3, None, None],
        [None, None, None, 4, None, None],
        [None, None, 2, None, None, None],
        [5, None, None, None, None, None],
        [None, 1, None, None, None, None],
        [None, None, None, None, 4, None],
        [0, None, None, None, None, None],
        [None, 5, None, None, None, 4],
        [None, None, None, None, None, None],
        [None, None, 5, None, None, 3],
        [None, None, 1, 2, 3, None],
        [None, None, None, 5, None, None],
        [None, None, 1, None, 3, None],
    ]
    assets = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for offset, values in enumerate(event_rows if event else factor_rows):
        date = datetime(2015, 1, 15) + timedelta(days=offset)
        for asset, value in zip(assets, values, strict=True):
            if value is not None:
                rows.append((date, asset, float(value)))
    dtype = pl.Datetime("ms", timezone) if timezone else pl.Datetime("ms")
    return pl.DataFrame(
        rows,
        schema=["date", "asset", "factor"],
        orient="row",
    ).with_columns(pl.col("date").cast(dtype))


def _upstream_tears_factor_data(
    *,
    quantiles: int | None = 2,
    bins: int | None = None,
    periods: tuple[int, ...] = (1, 5, 10),
    filter_zscore: float | None = None,
    timezone: str | None = None,
    event: bool = False,
) -> pl.DataFrame:
    return utils.get_clean_factor_and_forward_returns(
        _upstream_tears_factor(event=event, timezone=timezone),
        _upstream_tears_prices(timezone=timezone),
        groupby={"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2},
        quantiles=quantiles,
        bins=bins,
        periods=periods,
        filter_zscore=filter_zscore,
    ).frame


def _assert_signature(function, expected: list[tuple[str, object]]) -> None:
    signature = inspect.signature(function)
    assert list(signature.parameters) == [name for name, _ in expected]
    for name, default in expected:
        assert signature.parameters[name].default == default


def test_tears_import_does_not_load_plotting_or_stats_dependencies() -> None:
    for name in list(BLOCKED_IMPORTS):
        sys.modules.pop(name, None)

    importlib.reload(tears)

    assert BLOCKED_IMPORTS.isdisjoint(sys.modules)


def test_tears_and_event_performance_signatures_match_plan() -> None:
    _assert_signature(
        tears.create_summary_tear_sheet_data,
        [
            ("factor_data", inspect.Parameter.empty),
            ("long_short", True),
            ("group_neutral", False),
            ("turnover_periods", None),
        ],
    )
    _assert_signature(
        tears.create_returns_tear_sheet_data,
        [
            ("factor_data", inspect.Parameter.empty),
            ("long_short", True),
            ("group_neutral", False),
            ("by_group", False),
        ],
    )
    _assert_signature(
        tears.create_information_tear_sheet_data,
        [
            ("factor_data", inspect.Parameter.empty),
            ("group_neutral", False),
            ("by_group", False),
            ("rolling_window", 22),
        ],
    )
    _assert_signature(
        tears.create_turnover_tear_sheet_data,
        [("factor_data", inspect.Parameter.empty), ("turnover_periods", None)],
    )
    _assert_signature(
        tears.create_full_tear_sheet_data,
        [
            ("factor_data", inspect.Parameter.empty),
            ("long_short", True),
            ("group_neutral", False),
            ("by_group", False),
            ("turnover_periods", None),
            ("ic_rolling_window", 22),
        ],
    )
    _assert_signature(
        tears.create_event_returns_tear_sheet_data,
        [
            ("factor_data", inspect.Parameter.empty),
            ("returns", inspect.Parameter.empty),
            ("avgretplot", (5, 15)),
            ("long_short", True),
            ("group_neutral", False),
            ("by_group", False),
        ],
    )
    _assert_signature(
        tears.create_event_study_tear_sheet_data,
        [
            ("factor_data", inspect.Parameter.empty),
            ("returns", None),
            ("avgretplot", (5, 15)),
            ("rate_of_ret", True),
            ("n_bars", 50),
        ],
    )
    _assert_signature(
        performance.average_cumulative_return_by_quantile,
        [
            ("factor_data", inspect.Parameter.empty),
            ("returns", inspect.Parameter.empty),
            ("periods_before", 10),
            ("periods_after", 15),
            ("demeaned", True),
            ("group_adjust", False),
            ("by_group", False),
        ],
    )


def test_rendering_wrapper_signatures_match_plan() -> None:
    render_defaults = [
        ("backend", "html"),
        ("width", 1200),
        ("scale", 1.0),
        ("theme", "light"),
    ]
    for function, analytics in [
        (
            tears.create_summary_tear_sheet,
            [
                ("factor_data", inspect.Parameter.empty),
                ("long_short", True),
                ("group_neutral", False),
                ("turnover_periods", None),
            ],
        ),
        (
            tears.create_returns_tear_sheet,
            [
                ("factor_data", inspect.Parameter.empty),
                ("long_short", True),
                ("group_neutral", False),
                ("by_group", False),
            ],
        ),
        (
            tears.create_information_tear_sheet,
            [
                ("factor_data", inspect.Parameter.empty),
                ("group_neutral", False),
                ("by_group", False),
                ("rolling_window", 22),
            ],
        ),
        (
            tears.create_turnover_tear_sheet,
            [("factor_data", inspect.Parameter.empty), ("turnover_periods", None)],
        ),
        (
            tears.create_full_tear_sheet,
            [
                ("factor_data", inspect.Parameter.empty),
                ("long_short", True),
                ("group_neutral", False),
                ("by_group", False),
                ("turnover_periods", None),
                ("ic_rolling_window", 22),
            ],
        ),
        (
            tears.create_event_returns_tear_sheet,
            [
                ("factor_data", inspect.Parameter.empty),
                ("returns", inspect.Parameter.empty),
                ("avgretplot", (5, 15)),
                ("long_short", True),
                ("group_neutral", False),
                ("by_group", False),
            ],
        ),
        (
            tears.create_event_study_tear_sheet,
            [
                ("factor_data", inspect.Parameter.empty),
                ("returns", None),
                ("avgretplot", (5, 15)),
                ("rate_of_ret", True),
                ("n_bars", 50),
            ],
        ),
    ]:
        _assert_signature(function, [*analytics, *render_defaults])


def test_rendering_wrapper_calls_builder_and_renderer_once(monkeypatch) -> None:
    sentinel_report = object()
    sentinel_render = object()
    builder_calls = []
    render_calls = []

    def fake_builder(*args, **kwargs):
        builder_calls.append((args, kwargs))
        return sentinel_report

    def fake_render(report, **kwargs):
        render_calls.append((report, kwargs))
        return sentinel_render

    monkeypatch.setattr(tears, "create_returns_tear_sheet_data", fake_builder)
    monkeypatch.setattr(tears.plotting, "render", fake_render)

    result = tears.create_returns_tear_sheet(
        "frame",
        long_short=False,
        by_group=True,
        backend="svg",
        width=720,
        scale=2.0,
        theme="dark",
    )

    assert result is sentinel_render
    assert builder_calls == [(("frame", False, False, True), {})]
    assert render_calls == [
        (
            sentinel_report,
            {"backend": "svg", "width": 720, "scale": 2.0, "theme": "dark"},
        )
    ]


def test_full_report_exposes_tables_metadata_and_clone_on_read() -> None:
    report = tears.create_full_tear_sheet_data(
        _factor_data(),
        by_group=True,
        turnover_periods=[1, 3],
        ic_rolling_window=3,
    )

    assert isinstance(report, tears.TearSheetData)
    assert report.kind == "full"
    assert report.contract_version == "ferric-alpha.tear-sheet/v1"
    assert "information.summary" in report.table_ids()

    table = report.table("information.summary")
    assert isinstance(table, pl.DataFrame)
    changed = table.with_columns(pl.lit("changed").alias("period"))
    assert changed["period"].to_list()[0] == "changed"
    assert report.table("information.summary")["period"].to_list()[0] != "changed"

    metadata = report.table_metadata("information.summary")
    assert metadata["id"] == "information.summary"
    assert metadata["title"] == "Information Analysis"
    assert metadata["row_count"] == report.table("information.summary").height
    assert metadata["sort_by"] == ["period"]
    assert metadata["columns"][0]["name"] == "period"
    assert metadata["columns"][0]["role"] == "dimension"


def test_report_json_round_trips_for_every_kind() -> None:
    frame = _factor_data()
    event_frame = _event_factor_data()
    event_returns = _event_returns_data()
    reports = [
        tears.create_summary_tear_sheet_data(frame, turnover_periods=[1, 3]),
        tears.create_returns_tear_sheet_data(frame, by_group=True),
        tears.create_information_tear_sheet_data(
            frame, by_group=True, rolling_window=3
        ),
        tears.create_turnover_tear_sheet_data(frame, turnover_periods=[1, 3]),
        tears.create_full_tear_sheet_data(
            frame, by_group=True, turnover_periods=[1, 3]
        ),
        tears.create_event_returns_tear_sheet_data(
            event_frame, event_returns, avgretplot=(1, 1)
        ),
        tears.create_event_study_tear_sheet_data(
            event_frame, event_returns, avgretplot=(1, 1), n_bars=2
        ),
    ]

    assert [report.kind for report in reports] == [
        "summary",
        "returns",
        "information",
        "turnover",
        "full",
        "event_returns",
        "event_study",
    ]
    for report in reports:
        compact = report.to_json()
        pretty = report.to_json(pretty=True)
        decoded = tears.TearSheetData.from_json(compact)
        assert json.loads(compact) == json.loads(decoded.to_json())
        assert "\n" in pretty
        assert decoded.kind == report.kind
        assert decoded.table_ids() == report.table_ids()


@pytest.mark.parametrize(
    ("builder", "quantiles", "periods", "filter_zscore", "expected_kind"),
    [
        (tears.create_returns_tear_sheet_data, 2, (1, 5, 10), None, "returns"),
        (tears.create_returns_tear_sheet_data, 3, (2, 4, 6), 20.0, "returns"),
        (tears.create_information_tear_sheet_data, 2, (1, 5, 10), None, "information"),
        (
            tears.create_information_tear_sheet_data,
            4,
            (1, 2, 3, 7),
            20.0,
            "information",
        ),
        (tears.create_summary_tear_sheet_data, 2, (1, 5, 10), None, "summary"),
        (tears.create_summary_tear_sheet_data, 3, (1, 2, 3, 7), 20.0, "summary"),
        (tears.create_full_tear_sheet_data, 2, (1, 5, 10), None, "full"),
        (tears.create_full_tear_sheet_data, 3, (2, 4, 6), 20.0, "full"),
        (tears.create_full_tear_sheet_data, 4, (1, 8), 20.0, "full"),
        (tears.create_full_tear_sheet_data, 4, (1, 2, 3, 7), None, "full"),
    ],
)
def test_upstream_tears_parameter_matrix_builds_serializable_reports(
    builder,
    quantiles: int,
    periods: tuple[int, ...],
    filter_zscore: float | None,
    expected_kind: str,
) -> None:
    factor_data = _upstream_tears_factor_data(
        quantiles=quantiles,
        periods=periods,
        filter_zscore=filter_zscore,
    )

    report = builder(factor_data)

    assert report.kind == expected_kind
    assert report.table_ids()
    assert tears.TearSheetData.from_json(report.to_json()).kind == expected_kind


@pytest.mark.parametrize(
    ("turnover_periods", "quantiles", "periods", "filter_zscore"),
    [
        (None, 2, (2, 3, 6), 20.0),
        (None, 4, (1, 2, 3, 7), None),
        ([1, 2], 2, (2, 3, 6), 20.0),
        ([1], 4, (1, 2, 3, 7), None),
    ],
)
def test_upstream_turnover_tear_sheet_parameter_matrix_builds_report(
    turnover_periods: list[int] | None,
    quantiles: int,
    periods: tuple[int, ...],
    filter_zscore: float | None,
) -> None:
    factor_data = _upstream_tears_factor_data(
        quantiles=quantiles,
        periods=periods,
        filter_zscore=filter_zscore,
    )

    report = tears.create_turnover_tear_sheet_data(factor_data, turnover_periods)

    assert report.kind == "turnover"
    assert "turnover.mean_by_quantile" in report.table_ids()


@pytest.mark.parametrize(
    ("quantiles", "periods", "filter_zscore", "timezone"),
    [
        (2, (1, 5, 10), None, None),
        (3, (2, 4, 6), 20.0, "UTC"),
        (4, (3, 4), None, None),
        (2, (2, 3, 6, 9), 20.0, "UTC"),
    ],
)
def test_upstream_event_returns_tear_sheet_parameter_matrix_builds_report(
    quantiles: int,
    periods: tuple[int, ...],
    filter_zscore: float | None,
    timezone: str | None,
) -> None:
    factor_data = _upstream_tears_factor_data(
        quantiles=quantiles,
        periods=periods,
        filter_zscore=filter_zscore,
        timezone=timezone,
    )

    report = tears.create_event_returns_tear_sheet_data(
        factor_data,
        _upstream_tears_returns(timezone=timezone),
        avgretplot=(5, 11),
        long_short=False,
        group_neutral=False,
        by_group=False,
    )

    assert report.kind == "event_returns"
    assert "events.average_cumulative_returns" in report.table_ids()
    assert "events.coverage" in report.table_ids()


def test_event_tear_sheets_accept_alphalens_prices_input() -> None:
    factor_data = _upstream_tears_factor_data(
        quantiles=None,
        bins=1,
        periods=(1, 2),
        filter_zscore=None,
        event=True,
    )
    prices = _upstream_tears_prices()
    returns = _upstream_tears_returns()

    event_from_prices = tears.create_event_returns_tear_sheet_data(
        factor_data, prices, avgretplot=(2, 3)
    )
    event_from_returns = tears.create_event_returns_tear_sheet_data(
        factor_data, returns, avgretplot=(2, 3)
    )
    study_from_prices = tears.create_event_study_tear_sheet_data(
        factor_data, prices, avgretplot=(2, 3), n_bars=5
    )
    study_from_returns = tears.create_event_study_tear_sheet_data(
        factor_data, returns, avgretplot=(2, 3), n_bars=5
    )

    assert event_from_prices.to_json() == event_from_returns.to_json()
    assert study_from_prices.to_json() == study_from_returns.to_json()


@pytest.mark.parametrize(
    ("avgretplot", "filter_zscore", "timezone"),
    [
        ((6, 8), None, None),
        ((6, 3), 20.0, None),
        ((6, 3), 20.0, "UTC"),
        ((0, 3), None, None),
        ((3, 0), 20.0, "UTC"),
    ],
)
def test_upstream_event_study_tear_sheet_parameter_matrix_builds_report(
    avgretplot: tuple[int, int],
    filter_zscore: float | None,
    timezone: str | None,
) -> None:
    factor_data = _upstream_tears_factor_data(
        quantiles=None,
        bins=1,
        periods=(1, 2),
        filter_zscore=filter_zscore,
        timezone=timezone,
        event=True,
    )

    report = tears.create_event_study_tear_sheet_data(
        factor_data,
        _upstream_tears_returns(timezone=timezone),
        avgretplot=avgretplot,
    )

    assert report.kind == "event_study"
    assert report.table_ids()


def test_report_exceptions_map_to_python_contract() -> None:
    report = tears.create_returns_tear_sheet_data(_factor_data())

    with pytest.raises(KeyError):
        report.table("missing.table")
    with pytest.raises(KeyError):
        report.table_metadata("missing.table")
    with pytest.raises(ValueError, match="invalid JSON"):
        tears.TearSheetData.from_json("{")
    with pytest.raises(ValueError, match="unsupported report contract version"):
        tears.TearSheetData.from_json(
            report.to_json().replace(
                "ferric-alpha.tear-sheet/v1", "ferric-alpha.tear-sheet/v0"
            )
        )
    with pytest.raises(ValueError, match="missing required column"):
        tears.create_returns_tear_sheet_data(_factor_data().drop("factor"))
    corrupt = json.loads(report.to_json())
    corrupt["data"]["daily_by_quantile"]["columns"][0]["data"]["datetime"][
        "timezone"
    ] = "Not/AZone"
    with pytest.raises(RuntimeError):
        tears.TearSheetData.from_json(json.dumps(corrupt)).table(
            "returns.daily_by_quantile"
        )


def test_integer_boundaries_are_validated_in_python() -> None:
    frame = _factor_data()
    event_frame = _event_factor_data()
    event_returns = _event_returns_data()

    for value in [False, True, -1, 2**32, 10**100]:
        with pytest.raises(ValueError, match="invalid option turnover_periods"):
            tears.create_turnover_tear_sheet_data(frame, turnover_periods=[value])
        with pytest.raises(ValueError, match="invalid option rolling_window"):
            tears.create_information_tear_sheet_data(frame, rolling_window=value)
        with pytest.raises(ValueError, match="invalid option avgretplot"):
            tears.create_event_returns_tear_sheet_data(
                event_frame, event_returns, avgretplot=(value, 1)
            )
        with pytest.raises(ValueError, match="invalid option n_bars"):
            tears.create_event_study_tear_sheet_data(event_frame, n_bars=value)
        with pytest.raises(ValueError, match="invalid option periods_before"):
            performance.average_cumulative_return_by_quantile(
                event_frame, event_returns, periods_before=value
            )

    report = tears.create_information_tear_sheet_data(frame, rolling_window=2**32 - 1)
    assert report.table("information.ic_rolling").height == (
        frame["date"].n_unique() * 2
    )


def test_event_returns_rejects_missing_avgretplot_as_value_error() -> None:
    with pytest.raises(ValueError, match="invalid option avgretplot"):
        tears.create_event_returns_tear_sheet_data(
            _event_factor_data(), _event_returns_data(), avgretplot=None
        )


def test_raw_event_study_binding_rejects_short_avgretplot() -> None:
    with pytest.raises(ValueError, match="invalid option avgretplot"):
        tears._ferric_alpha.create_event_study_tear_sheet_data(
            _event_factor_data(), avgretplot=[1]
        )
