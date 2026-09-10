use approx::assert_abs_diff_eq;
use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, EventReturnsOptions, EventStudyTearSheetOptions,
    FerricAlphaError, ReportColumnData, ReportKind, create_event_returns_tear_sheet_data,
    create_event_study_tear_sheet_data,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};
use serde_json::Value;

const JAN_1: i64 = 1_704_067_200_000;
const DAY_MS: i64 = 86_400_000;

fn factor_data() -> DataFrame {
    let jan_2 = JAN_1 + DAY_MS;
    let absent = JAN_1 + 3 * DAY_MS;
    DataFrame::new(
        5,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1, jan_2, jan_2, absent])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["A", "B", "A", "B", "A"]).into(),
            Series::new("group".into(), &["g1", "g1", "g1", "g1", "g1"]).into(),
            Series::new(
                "factor".into(),
                &[Some(-1.0), Some(1.0), Some(-0.5), Some(0.5), Some(2.0)],
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                &[Some(1_u32), Some(2), Some(1), Some(2), Some(2)],
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                &[Some(-0.01), Some(0.02), Some(-0.02), Some(0.03), Some(0.04)],
            )
            .into(),
        ],
    )
    .unwrap()
}

fn returns_data() -> DataFrame {
    let jan_2 = JAN_1 + DAY_MS;
    let jan_3 = JAN_1 + 2 * DAY_MS;
    DataFrame::new(
        5,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1, jan_2, jan_2, jan_3])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["A", "B", "A", "B", "A"]).into(),
            Series::new(
                "return".into(),
                &[Some(0.10), Some(0.20), None, Some(0.00), Some(0.10)],
            )
            .into(),
        ],
    )
    .unwrap()
}

fn single_date_factor_data() -> DataFrame {
    DataFrame::new(
        2,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["A", "B"]).into(),
            Series::new("group".into(), &["g1", "g1"]).into(),
            Series::new("factor".into(), &[Some(-1.0), Some(1.0)]).into(),
            Series::new("factor_quantile".into(), &[Some(1_u32), Some(2)]).into(),
            Series::new("forward_return_1D".into(), &[Some(-0.01), Some(0.02)]).into(),
        ],
    )
    .unwrap()
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

fn i32_values(frame: &DataFrame, name: &str) -> Vec<Option<i32>> {
    frame
        .column(name)
        .unwrap()
        .as_materialized_series()
        .i32()
        .unwrap()
        .iter()
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

fn f64_values(frame: &DataFrame, name: &str) -> Vec<Option<f64>> {
    frame
        .column(name)
        .unwrap()
        .as_materialized_series()
        .f64()
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

#[test]
fn event_returns_schema_and_coverage_are_stable() {
    let report = create_event_returns_tear_sheet_data(
        &factor_data(),
        &returns_data(),
        EventReturnsOptions {
            periods_before: 1,
            periods_after: 1,
            demeaned: false,
            group_adjust: false,
            by_group: false,
        },
    )
    .expect("event returns report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::EventReturns);
    assert!(!report.options.demeaned);
    assert_eq!(
        report.data.average_cumulative_returns.id(),
        "events.average_cumulative_returns"
    );
    assert_eq!(report.data.coverage.id(), "events.coverage");
    assert_eq!(
        report.data.average_cumulative_returns.column_names(),
        vec![
            "factor_quantile",
            "offset",
            "mean_cumulative_return",
            "std_cumulative_return",
            "count",
        ]
    );
    assert_eq!(
        report.data.average_cumulative_returns.sort_by(),
        &["factor_quantile", "offset"]
    );
    assert_eq!(
        report.data.average_cumulative_returns.columns()[0].role(),
        ColumnRole::Dimension
    );
    assert_eq!(
        report.data.average_cumulative_returns.columns()[2].display(),
        Some(&DisplayFormat::new(DisplayUnit::DecimalReturn, 6))
    );
    assert_eq!(
        report.data.average_cumulative_returns.columns()[4].display(),
        Some(&DisplayFormat::new(DisplayUnit::Count, 0))
    );

    let returns = report.data.average_cumulative_returns.to_polars().unwrap();
    assert_eq!(
        u32_values(&returns, "factor_quantile"),
        vec![Some(1), Some(1), Some(1), Some(2), Some(2), Some(2)]
    );
    assert_eq!(
        i32_values(&returns, "offset"),
        vec![Some(-1), Some(0), Some(1), Some(-1), Some(0), Some(1)]
    );

    assert_eq!(
        report.data.coverage.column_names(),
        vec![
            "coverage_row",
            "scope",
            "offset",
            "input_events",
            "used_events",
            "skipped_missing_date",
            "imputed_return_cells",
            "path_count",
        ]
    );
    assert_eq!(report.data.coverage.sort_by(), &["coverage_row"]);

    let coverage = report.data.coverage.to_polars().unwrap();
    assert_eq!(
        u32_values(&coverage, "coverage_row"),
        vec![Some(0), Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        string_values(&coverage, "scope"),
        vec![
            Some("overall".to_string()),
            Some("offset".to_string()),
            Some("offset".to_string()),
            Some("offset".to_string()),
        ]
    );
    assert_eq!(
        i32_values(&coverage, "offset"),
        vec![None, Some(-1), Some(0), Some(1)]
    );
    assert_eq!(
        u64_values(&coverage, "input_events"),
        vec![Some(5), Some(5), Some(5), Some(5)]
    );
    assert_eq!(
        u64_values(&coverage, "used_events"),
        vec![Some(4), Some(2), Some(4), Some(4)]
    );
    assert_eq!(
        u64_values(&coverage, "skipped_missing_date"),
        vec![Some(1), Some(1), Some(1), Some(1)]
    );
    assert_eq!(
        u64_values(&coverage, "imputed_return_cells"),
        vec![Some(2), Some(0), Some(1), Some(2)]
    );
    assert_eq!(
        u64_values(&coverage, "path_count"),
        vec![Some(0), Some(2), Some(4), Some(4)]
    );
}

#[test]
fn event_study_serialization_distribution_and_long_only_returns_are_stable() {
    let report = create_event_study_tear_sheet_data(
        &factor_data(),
        Some(&returns_data()),
        EventStudyTearSheetOptions {
            event_window: Some((1, 1)),
            rate_of_return: true,
            histogram_bins: 2,
        },
    )
    .expect("event study report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::EventStudy);
    assert_eq!(report.options.event_window, Some((1, 1)));
    assert!(report.options.rate_of_return);
    assert_eq!(report.options.histogram_bins, 2);
    assert_eq!(report.data.quantile_statistics.id(), "quantile.statistics");
    assert_eq!(report.data.distribution.id(), "events.distribution");
    assert_eq!(
        report.data.mean_by_quantile.id(),
        "returns.mean_by_quantile"
    );
    assert_eq!(
        report.data.daily_by_quantile.id(),
        "returns.daily_by_quantile"
    );
    assert_eq!(
        report
            .data
            .event_returns
            .as_ref()
            .unwrap()
            .average_cumulative_returns
            .id(),
        "events.average_cumulative_returns"
    );

    let json = serde_json::to_value(&report).expect("report should serialize");
    assert_eq!(
        json["metadata"]["report_kind"],
        Value::String("event_study".to_string())
    );
    assert_eq!(json["options"]["rate_of_return"], Value::Bool(true));
    assert_eq!(json["options"]["histogram_bins"], Value::Number(2.into()));
    assert!(json["data"]["event_returns"].is_object());

    assert_eq!(
        report.data.distribution.column_names(),
        vec!["bin_start", "bin_end", "event_count"]
    );
    assert_eq!(report.data.distribution.sort_by(), &["bin_start"]);
    assert_eq!(
        report.data.distribution.columns()[0].data(),
        &ReportColumnData::Datetime {
            values: vec![Some(JAN_1), Some(JAN_1 + DAY_MS + DAY_MS / 2)],
            time_unit: ferric_alpha::ReportTimeUnit::Milliseconds,
            timezone: Some("UTC".to_string()),
        }
    );
    assert_eq!(
        u64_values(
            &report.data.distribution.to_polars().unwrap(),
            "event_count"
        ),
        vec![Some(4), Some(1)]
    );

    let means = report.data.mean_by_quantile.to_polars().unwrap();
    let mean_returns = means
        .column("mean_return")
        .unwrap()
        .as_materialized_series()
        .f64()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_abs_diff_eq!(mean_returns[0].unwrap(), -0.015, epsilon = 1e-12);
    assert_abs_diff_eq!(mean_returns[1].unwrap(), 0.03, epsilon = 1e-12);
}

#[test]
fn event_study_without_returns_skips_event_window_tables_and_collapses_zero_width_bins() {
    let invalid = create_event_study_tear_sheet_data(
        &single_date_factor_data(),
        None,
        EventStudyTearSheetOptions {
            event_window: Some((1, 1)),
            rate_of_return: false,
            histogram_bins: 0,
        },
    )
    .expect_err("zero histogram bins should fail");
    assert!(matches!(
        invalid,
        FerricAlphaError::InvalidOption {
            option: "histogram_bins",
            ..
        }
    ));

    let report = create_event_study_tear_sheet_data(
        &single_date_factor_data(),
        None,
        EventStudyTearSheetOptions {
            event_window: Some((1, 1)),
            rate_of_return: false,
            histogram_bins: 4,
        },
    )
    .expect("event study without returns should build");

    assert!(report.data.event_returns.is_none());
    assert_eq!(report.options.histogram_bins, 4);

    let distribution = report.data.distribution.to_polars().unwrap();
    assert_eq!(distribution.height(), 1);
    assert_eq!(u64_values(&distribution, "event_count"), vec![Some(2)]);
}

#[test]
fn event_study_rate_of_return_false_preserves_period_returns() {
    let frame = DataFrame::new(
        4,
        vec![
            Series::new(
                "date".into(),
                &[JAN_1, JAN_1, JAN_1 + DAY_MS, JAN_1 + DAY_MS],
            )
            .into_datetime(
                TimeUnit::Milliseconds,
                TimeZone::opt_try_new(Some("UTC")).unwrap(),
            )
            .into_column(),
            Series::new("asset".into(), &["A", "B", "A", "B"]).into(),
            Series::new("group".into(), &["g1", "g1", "g1", "g1"]).into(),
            Series::new(
                "factor".into(),
                &[Some(-1.0), Some(1.0), Some(-0.5), Some(0.5)],
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                &[Some(1_u32), Some(2), Some(1), Some(2)],
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                &[Some(-0.01), Some(0.02), Some(-0.02), Some(0.03)],
            )
            .into(),
            Series::new(
                "forward_return_3D".into(),
                &[Some(-0.03), Some(0.06), Some(-0.06), Some(0.09)],
            )
            .into(),
        ],
    )
    .unwrap();

    let report = create_event_study_tear_sheet_data(
        &frame,
        None,
        EventStudyTearSheetOptions {
            event_window: None,
            rate_of_return: false,
            histogram_bins: 2,
        },
    )
    .expect("event study report should build");

    let means = report.data.mean_by_quantile.to_polars().unwrap();
    let quantiles = u32_values(&means, "factor_quantile");
    let periods = string_values(&means, "period");
    let returns = f64_values(&means, "mean_return");
    let q1_3d = quantiles
        .iter()
        .zip(periods.iter())
        .zip(returns.iter())
        .find_map(|((quantile, period), value)| {
            (*quantile == Some(1) && period.as_deref() == Some("3D")).then_some(value.unwrap())
        })
        .expect("q1 3D row should exist");
    assert_abs_diff_eq!(q1_3d, -0.045, epsilon = 1e-12);
}
