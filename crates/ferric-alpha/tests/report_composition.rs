use std::collections::HashSet;

use ferric_alpha::{
    FerricAlphaError, FullTearSheetOptions, InformationTearSheetOptions, Period, ReportColumnData,
    ReportKind, ReportTable, ReturnsTearSheetOptions, SummaryTearSheetOptions,
    TurnoverTearSheetOptions, create_full_tear_sheet_data, create_information_tear_sheet_data,
    create_returns_tear_sheet_data, create_summary_tear_sheet_data,
    create_turnover_tear_sheet_data,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};
use serde_json::Value;

const JAN_1: i64 = 1_704_067_200_000;
const DAY_MS: i64 = 86_400_000;

fn composition_factor_data(day_count: i32, include_group: bool, constant_ic: bool) -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut groups = Vec::new();
    let mut factors = Vec::new();
    let mut quantiles = Vec::new();
    let mut one_day = Vec::new();
    let mut three_day = Vec::new();

    for day in 0..day_count {
        let date = JAN_1 + i64::from(day) * DAY_MS;
        let rotation = if constant_ic {
            0
        } else {
            day.rem_euclid(4) as usize
        };
        for asset_index in 0..4 {
            dates.push(date);
            assets.push(["a", "b", "c", "d"][asset_index]);
            groups.push(if asset_index < 2 { "g1" } else { "g2" });
            factors.push(Some((asset_index + 1) as f64));
            quantiles.push((asset_index + 1) as u32);
            one_day.push(Some((rotation + asset_index + 1) as f64 * 0.01));
            three_day.push(Some((rotation + asset_index + 1) as f64 * 0.03));
        }
    }

    let mut columns = vec![
        Series::new("date".into(), dates)
            .into_datetime(
                TimeUnit::Milliseconds,
                TimeZone::opt_try_new(Some("UTC")).unwrap(),
            )
            .into_column(),
        Series::new("asset".into(), assets).into(),
    ];
    if include_group {
        columns.push(Series::new("group".into(), groups).into());
    }
    columns.extend([
        Series::new("factor".into(), factors).into(),
        Series::new("factor_quantile".into(), quantiles).into(),
        Series::new("forward_return_1D".into(), one_day).into(),
        Series::new("forward_return_3D".into(), three_day).into(),
    ]);

    DataFrame::new((day_count as usize) * 4, columns).unwrap()
}

fn summary_table_ids(report: &ferric_alpha::SummaryTearSheetData) -> Vec<&str> {
    vec![
        report.data.quantile_statistics.id(),
        report.data.returns_summary.id(),
        report.data.returns_mean_by_quantile.id(),
        report.data.information_summary.id(),
        report.data.turnover_mean_by_quantile.id(),
        report.data.turnover_mean_rank_autocorrelation.id(),
    ]
}

fn full_tables(report: &ferric_alpha::FullTearSheetData) -> Vec<&ReportTable> {
    let mut tables = vec![
        &report.data.quantile_statistics,
        &report.data.returns.summary,
        &report.data.returns.mean_by_quantile,
        &report.data.returns.daily_by_quantile,
        &report.data.returns.daily_spread,
        &report.data.returns.factor_returns,
        &report.data.information.summary,
        &report.data.information.ic_by_date,
        &report.data.information.ic_rolling,
        &report.data.information.ic_monthly,
        &report.data.information.qq_normal,
        &report.data.turnover.mean_by_quantile,
        &report.data.turnover.mean_rank_autocorrelation,
        &report.data.turnover.quantile_series,
        &report.data.turnover.rank_autocorrelation_series,
    ];
    if let Some(table) = &report.data.returns.group_mean_by_quantile {
        tables.push(table);
    }
    if let Some(table) = &report.data.returns.factor_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &report.data.returns.quantile_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &report.data.information.ic_by_group {
        tables.push(table);
    }
    tables
}

fn assert_same_table(left: &ReportTable, right: &ReportTable) {
    assert_eq!(left, right);
    assert_eq!(
        serde_json::to_value(left).unwrap(),
        serde_json::to_value(right).unwrap()
    );
}

#[test]
fn summary_contains_exact_compact_tables_without_time_series() {
    let report = create_summary_tear_sheet_data(
        &composition_factor_data(24, true, false),
        SummaryTearSheetOptions::default(),
    )
    .expect("summary report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::Summary);
    assert_eq!(
        summary_table_ids(&report),
        vec![
            "quantile.statistics",
            "returns.summary",
            "returns.mean_by_quantile",
            "information.summary",
            "turnover.mean_by_quantile",
            "turnover.mean_rank_autocorrelation",
        ]
    );
    assert!(
        summary_table_ids(&report)
            .into_iter()
            .all(|id| !id.contains("daily")
                && !id.contains("by_date")
                && !id.contains("rolling")
                && !id.contains("monthly")
                && !id.contains("series")
                && !id.contains("cumulative"))
    );
}

#[test]
fn full_contains_every_enabled_dataset_once() {
    let report = create_full_tear_sheet_data(
        &composition_factor_data(24, true, false),
        FullTearSheetOptions {
            long_short: true,
            group_neutral: false,
            by_group: true,
            turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
            ic_rolling_window: 3,
        },
    )
    .expect("full report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::Full);
    assert_eq!(report.options.ic_rolling_window, 3);

    let ids = full_tables(&report)
        .into_iter()
        .map(|table| table.id())
        .collect::<Vec<_>>();
    let unique = ids.iter().copied().collect::<HashSet<_>>();
    assert_eq!(unique.len(), ids.len());
    assert_eq!(
        ids,
        vec![
            "quantile.statistics",
            "returns.summary",
            "returns.mean_by_quantile",
            "returns.daily_by_quantile",
            "returns.daily_spread",
            "returns.factor_returns",
            "information.summary",
            "information.ic_by_date",
            "information.ic_rolling",
            "information.ic_monthly",
            "information.qq_normal",
            "turnover.mean_by_quantile",
            "turnover.mean_rank_autocorrelation",
            "turnover.quantile_series",
            "turnover.rank_autocorrelation_series",
            "returns.group_mean_by_quantile",
            "returns.factor_cumulative_1d",
            "returns.quantile_cumulative_1d",
            "information.ic_by_group",
        ]
    );
}

#[test]
fn full_tables_match_standalone_tables_for_identical_options() {
    let data = composition_factor_data(24, true, false);
    let full_options = FullTearSheetOptions {
        long_short: false,
        group_neutral: false,
        by_group: true,
        turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
        ic_rolling_window: 3,
    };
    let full = create_full_tear_sheet_data(&data, full_options.clone()).unwrap();
    let returns_options = ReturnsTearSheetOptions {
        long_short: full_options.long_short,
        group_neutral: full_options.group_neutral,
        by_group: full_options.by_group,
    };
    let information_options = InformationTearSheetOptions {
        group_neutral: full_options.group_neutral,
        by_group: full_options.by_group,
        rolling_window: full_options.ic_rolling_window,
    };
    let turnover_options = TurnoverTearSheetOptions {
        periods: full_options.turnover_periods,
    };
    let returns = create_returns_tear_sheet_data(&data, returns_options).unwrap();
    let information = create_information_tear_sheet_data(&data, information_options).unwrap();
    let turnover = create_turnover_tear_sheet_data(&data, turnover_options).unwrap();

    assert_same_table(&full.data.returns.summary, &returns.data.summary);
    assert_same_table(
        &full.data.returns.mean_by_quantile,
        &returns.data.mean_by_quantile,
    );
    assert_same_table(
        &full.data.returns.daily_by_quantile,
        &returns.data.daily_by_quantile,
    );
    assert_same_table(&full.data.returns.daily_spread, &returns.data.daily_spread);
    assert_same_table(
        &full.data.returns.factor_returns,
        &returns.data.factor_returns,
    );
    assert_same_table(
        full.data.returns.group_mean_by_quantile.as_ref().unwrap(),
        returns.data.group_mean_by_quantile.as_ref().unwrap(),
    );
    assert_same_table(
        full.data.returns.factor_cumulative_1d.as_ref().unwrap(),
        returns.data.factor_cumulative_1d.as_ref().unwrap(),
    );
    assert_same_table(
        full.data.returns.quantile_cumulative_1d.as_ref().unwrap(),
        returns.data.quantile_cumulative_1d.as_ref().unwrap(),
    );

    assert_same_table(&full.data.information.summary, &information.data.summary);
    assert_same_table(
        &full.data.information.ic_by_date,
        &information.data.ic_by_date,
    );
    assert_same_table(
        &full.data.information.ic_rolling,
        &information.data.ic_rolling,
    );
    assert_same_table(
        &full.data.information.ic_monthly,
        &information.data.ic_monthly,
    );
    assert_same_table(
        full.data.information.ic_by_group.as_ref().unwrap(),
        information.data.ic_by_group.as_ref().unwrap(),
    );
    assert_same_table(
        &full.data.information.qq_normal,
        &information.data.qq_normal,
    );

    assert_same_table(
        &full.data.turnover.mean_by_quantile,
        &turnover.data.mean_by_quantile,
    );
    assert_same_table(
        &full.data.turnover.mean_rank_autocorrelation,
        &turnover.data.mean_rank_autocorrelation,
    );
    assert_same_table(
        &full.data.turnover.quantile_series,
        &turnover.data.quantile_series,
    );
    assert_same_table(
        &full.data.turnover.rank_autocorrelation_series,
        &turnover.data.rank_autocorrelation_series,
    );
}

#[test]
fn missing_group_under_by_group_returns_original_typed_error() {
    let error = create_full_tear_sheet_data(
        &composition_factor_data(24, false, false),
        FullTearSheetOptions {
            by_group: true,
            ..FullTearSheetOptions::default()
        },
    )
    .expect_err("missing group should abort full report generation");

    assert!(matches!(error, FerricAlphaError::MissingColumn("group")));
}

#[test]
fn constant_ic_returns_complete_full_report_with_null_mathematical_fields() {
    let report = create_full_tear_sheet_data(
        &composition_factor_data(24, true, true),
        FullTearSheetOptions::default(),
    )
    .expect("constant IC should remain a complete report");

    assert_eq!(
        full_tables(&report)
            .into_iter()
            .map(|table| table.id())
            .collect::<Vec<_>>(),
        vec![
            "quantile.statistics",
            "returns.summary",
            "returns.mean_by_quantile",
            "returns.daily_by_quantile",
            "returns.daily_spread",
            "returns.factor_returns",
            "information.summary",
            "information.ic_by_date",
            "information.ic_rolling",
            "information.ic_monthly",
            "information.qq_normal",
            "turnover.mean_by_quantile",
            "turnover.mean_rank_autocorrelation",
            "turnover.quantile_series",
            "turnover.rank_autocorrelation_series",
            "returns.factor_cumulative_1d",
            "returns.quantile_cumulative_1d",
        ]
    );

    let json = serde_json::to_value(&report).expect("full report should serialize");
    assert_eq!(
        json["metadata"]["report_kind"],
        Value::String("full".to_string())
    );

    let summary = &report.data.information.summary;
    assert_eq!(summary.id(), "information.summary");
    let row_count = summary.row_count();
    for column in summary.columns() {
        match column.name() {
            "risk_adjusted_ic" | "t_stat" | "p_value" | "skew" | "excess_kurtosis" => {
                assert_eq!(
                    column.data(),
                    &ReportColumnData::Float64(vec![None; row_count])
                );
            }
            _ => {}
        }
    }
}
