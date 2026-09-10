use std::collections::BTreeMap;

use approx::assert_abs_diff_eq;
use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, FerricAlphaError, Period, ReportKind,
    TurnoverTearSheetOptions, create_turnover_tear_sheet_data,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};

const JAN_1: i64 = 1_704_067_200_000;
const DAY_MS: i64 = 86_400_000;

fn turnover_factor_data(day_count: i32) -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut factors = Vec::new();
    let mut quantiles = Vec::new();
    let mut one_day = Vec::new();
    let mut three_day = Vec::new();
    let mut five_day = Vec::new();

    for day in 0..day_count {
        let date = JAN_1 + i64::from(day) * DAY_MS;
        let rotation = day.rem_euclid(3) as usize;
        for asset_index in 0..4 {
            dates.push(date);
            assets.push(["a", "b", "c", "d"][asset_index]);
            factors.push(Some((asset_index + 1 + rotation) as f64));
            quantiles.push(Some([1_u32, 2, 3, 1][(asset_index + rotation) % 4]));
            one_day.push(Some((asset_index as f64 + 1.0) * 0.01));
            three_day.push(Some((asset_index as f64 + 1.0) * 0.03));
            five_day.push(Some((asset_index as f64 + 1.0) * 0.05));
        }
    }

    DataFrame::new(
        dates.len(),
        vec![
            Series::new("date".into(), dates)
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), assets).into(),
            Series::new("factor".into(), factors).into(),
            Series::new("factor_quantile".into(), quantiles).into(),
            Series::new("forward_return_5D".into(), five_day).into(),
            Series::new("forward_return_1D".into(), one_day).into(),
            Series::new("forward_return_3D".into(), three_day).into(),
        ],
    )
    .unwrap()
}

fn no_return_period_factor_data(day_count: i32) -> DataFrame {
    turnover_factor_data(day_count).drop_many([
        "forward_return_1D",
        "forward_return_3D",
        "forward_return_5D",
    ])
}

fn string_values(frame: &DataFrame, name: &str) -> Vec<Option<String>> {
    frame
        .column(name)
        .unwrap()
        .as_materialized_series()
        .str()
        .unwrap()
        .iter()
        .map(|value| value.map(ToOwned::to_owned))
        .collect()
}

fn u32_values(frame: &DataFrame, name: &str) -> Vec<Option<u32>> {
    frame
        .column(name)
        .unwrap()
        .as_materialized_series()
        .u32()
        .unwrap()
        .iter()
        .collect()
}

fn u64_values(frame: &DataFrame, name: &str) -> Vec<Option<u64>> {
    frame
        .column(name)
        .unwrap()
        .as_materialized_series()
        .u64()
        .unwrap()
        .iter()
        .collect()
}

fn float_values(frame: &DataFrame, name: &str) -> Vec<Option<f64>> {
    frame
        .column(name)
        .unwrap()
        .as_materialized_series()
        .f64()
        .unwrap()
        .iter()
        .collect()
}

#[test]
fn default_period_discovery_uses_sorted_forward_return_periods_and_quantiles() {
    let report = create_turnover_tear_sheet_data(
        &turnover_factor_data(7),
        TurnoverTearSheetOptions::default(),
    )
    .expect("turnover report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::Turnover);
    assert_eq!(
        report.metadata.periods(),
        &["1D".to_string(), "3D".to_string(), "5D".to_string()]
    );
    assert_eq!(report.metadata.quantiles(), &[1, 2, 3]);
    assert_eq!(report.options.periods, None);
    assert_eq!(
        report.data.mean_by_quantile.id(),
        "turnover.mean_by_quantile"
    );
    assert_eq!(
        report.data.mean_by_quantile.title(),
        "Mean Quantile Turnover"
    );
    assert_eq!(
        report.data.mean_rank_autocorrelation.id(),
        "turnover.mean_rank_autocorrelation"
    );
    assert_eq!(report.data.quantile_series.id(), "turnover.quantile_series");
    assert_eq!(
        report.data.rank_autocorrelation_series.id(),
        "turnover.rank_autocorrelation_series"
    );

    assert_eq!(
        report.data.mean_by_quantile.column_names(),
        vec!["factor_quantile", "period", "turnover", "count"]
    );
    assert_eq!(
        report.data.rank_autocorrelation_series.column_names(),
        vec!["date", "period", "autocorrelation"]
    );
    assert_eq!(
        report.data.mean_by_quantile.columns()[0].role(),
        ColumnRole::Dimension
    );
    assert_eq!(
        report.data.mean_by_quantile.columns()[2].display(),
        Some(&DisplayFormat::new(DisplayUnit::Ratio, 6))
    );
    assert_eq!(
        report.data.mean_by_quantile.columns()[3].display(),
        Some(&DisplayFormat::new(DisplayUnit::Count, 0))
    );

    let mean = report.data.mean_by_quantile.to_polars().unwrap();
    assert_eq!(
        string_values(&mean, "period"),
        vec![
            Some("1D".to_string()),
            Some("3D".to_string()),
            Some("5D".to_string()),
            Some("1D".to_string()),
            Some("3D".to_string()),
            Some("5D".to_string()),
            Some("1D".to_string()),
            Some("3D".to_string()),
            Some("5D".to_string()),
        ]
    );
    assert_eq!(
        u32_values(&mean, "factor_quantile"),
        vec![
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            Some(3),
            Some(3),
            Some(3),
        ]
    );
}

#[test]
fn explicit_periods_absent_from_forward_returns_are_accepted_and_deduplicated() {
    let periods = vec![Period::new(4).unwrap(), Period::new(2).unwrap()];
    let report = create_turnover_tear_sheet_data(
        &no_return_period_factor_data(7),
        TurnoverTearSheetOptions {
            periods: Some(periods.clone()),
        },
    )
    .expect("explicit turnover periods should not require forward-return columns");

    assert_eq!(report.options.periods, Some(periods));

    let quantile_series = report.data.quantile_series.to_polars().unwrap();
    assert_eq!(
        quantile_series
            .column("period")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .take(6)
            .collect::<Vec<_>>(),
        vec!["2D", "4D", "2D", "4D", "2D", "4D"]
    );

    let duplicate = create_turnover_tear_sheet_data(
        &turnover_factor_data(7),
        TurnoverTearSheetOptions {
            periods: Some(vec![Period::new(2).unwrap(), Period::new(2).unwrap()]),
        },
    )
    .expect_err("duplicate explicit turnover periods should fail");
    assert!(matches!(duplicate, FerricAlphaError::DuplicatePeriod));
}

#[test]
fn summaries_are_derived_from_detail_rows_and_ignore_nulls() {
    let report = create_turnover_tear_sheet_data(
        &turnover_factor_data(7),
        TurnoverTearSheetOptions {
            periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
        },
    )
    .unwrap();

    let quantile_detail = report.data.quantile_series.to_polars().unwrap();
    let quantile_summary = report.data.mean_by_quantile.to_polars().unwrap();
    let mut detail_turnover = BTreeMap::<(u32, String), Vec<f64>>::new();
    let detail_quantiles = quantile_detail
        .column("factor_quantile")
        .unwrap()
        .as_materialized_series()
        .u32()
        .unwrap();
    let detail_periods = quantile_detail
        .column("period")
        .unwrap()
        .as_materialized_series()
        .str()
        .unwrap();
    let detail_values = quantile_detail
        .column("turnover")
        .unwrap()
        .as_materialized_series()
        .f64()
        .unwrap();
    for index in 0..quantile_detail.height() {
        if let Some(value) = detail_values.get(index) {
            detail_turnover
                .entry((
                    detail_quantiles.get(index).unwrap(),
                    detail_periods.get(index).unwrap().to_string(),
                ))
                .or_default()
                .push(value);
        }
    }

    let summary_quantiles = quantile_summary
        .column("factor_quantile")
        .unwrap()
        .as_materialized_series()
        .u32()
        .unwrap();
    let summary_periods = quantile_summary
        .column("period")
        .unwrap()
        .as_materialized_series()
        .str()
        .unwrap();
    let summary_values = float_values(&quantile_summary, "turnover");
    let summary_counts = u64_values(&quantile_summary, "count");
    for index in 0..quantile_summary.height() {
        let key = (
            summary_quantiles.get(index).unwrap(),
            summary_periods.get(index).unwrap().to_string(),
        );
        let values = detail_turnover.get(&key).cloned().unwrap_or_default();
        assert_eq!(summary_counts[index], Some(values.len() as u64));
        assert_abs_diff_eq!(
            summary_values[index].unwrap(),
            values.iter().sum::<f64>() / values.len() as f64,
            epsilon = 1e-12
        );
    }

    let rank_detail = report.data.rank_autocorrelation_series.to_polars().unwrap();
    let rank_summary = report.data.mean_rank_autocorrelation.to_polars().unwrap();
    let mut detail_autocorrelation = BTreeMap::<String, Vec<f64>>::new();
    let rank_detail_periods = rank_detail
        .column("period")
        .unwrap()
        .as_materialized_series()
        .str()
        .unwrap();
    let rank_detail_values = rank_detail
        .column("autocorrelation")
        .unwrap()
        .as_materialized_series()
        .f64()
        .unwrap();
    for index in 0..rank_detail.height() {
        if let Some(value) = rank_detail_values.get(index) {
            detail_autocorrelation
                .entry(rank_detail_periods.get(index).unwrap().to_string())
                .or_default()
                .push(value);
        }
    }

    let rank_summary_periods = string_values(&rank_summary, "period");
    let rank_summary_values = float_values(&rank_summary, "autocorrelation");
    let rank_summary_counts = u64_values(&rank_summary, "count");
    for index in 0..rank_summary.height() {
        let period = rank_summary_periods[index].as_ref().unwrap();
        let values = detail_autocorrelation
            .get(period)
            .cloned()
            .unwrap_or_default();
        assert_eq!(rank_summary_counts[index], Some(values.len() as u64));
        assert_abs_diff_eq!(
            rank_summary_values[index].unwrap(),
            values.iter().sum::<f64>() / values.len() as f64,
            epsilon = 1e-12
        );
    }
}

#[test]
fn all_null_lag_summaries_are_retained_with_null_mean_and_zero_count() {
    let report = create_turnover_tear_sheet_data(
        &turnover_factor_data(3),
        TurnoverTearSheetOptions {
            periods: Some(vec![Period::new(4).unwrap()]),
        },
    )
    .unwrap();

    let mean_by_quantile = report.data.mean_by_quantile.to_polars().unwrap();
    assert_eq!(
        string_values(&mean_by_quantile, "period"),
        vec![
            Some("4D".to_string()),
            Some("4D".to_string()),
            Some("4D".to_string()),
        ]
    );
    assert_eq!(
        float_values(&mean_by_quantile, "turnover"),
        vec![None, None, None]
    );
    assert_eq!(
        u64_values(&mean_by_quantile, "count"),
        vec![Some(0), Some(0), Some(0)]
    );

    let mean_rank = report.data.mean_rank_autocorrelation.to_polars().unwrap();
    assert_eq!(
        string_values(&mean_rank, "period"),
        vec![Some("4D".to_string())]
    );
    assert_eq!(float_values(&mean_rank, "autocorrelation"), vec![None]);
    assert_eq!(u64_values(&mean_rank, "count"), vec![Some(0)]);
}
