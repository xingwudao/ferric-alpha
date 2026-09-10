use approx::assert_abs_diff_eq;
use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportKind, ReturnsTearSheetOptions,
    create_returns_tear_sheet_data,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};

const JAN_1: i64 = 1_704_067_200_000;
const JAN_2: i64 = 1_704_153_600_000;
const JAN_3: i64 = 1_704_240_000_000;

fn base_factor_data() -> DataFrame {
    DataFrame::new(
        6,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1, JAN_2, JAN_2, JAN_3, JAN_3])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["a", "b", "a", "b", "a", "b"]).into(),
            Series::new("group".into(), &["g1", "g2", "g1", "g2", "g1", "g2"]).into(),
            Series::new(
                "factor".into(),
                &[
                    Some(-1.0),
                    Some(1.0),
                    Some(-1.0),
                    Some(1.0),
                    Some(-1.0),
                    Some(1.0),
                ],
            )
            .into(),
            Series::new("factor_quantile".into(), &[1_u32, 2, 1, 2, 1, 2]).into(),
            Series::new(
                "forward_return_5D".into(),
                &[
                    Some(-0.10),
                    Some(0.20),
                    Some(-0.05),
                    Some(0.15),
                    Some(-0.02),
                    Some(0.08),
                ],
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                &[
                    Some(-0.01),
                    Some(0.02),
                    Some(-0.02),
                    Some(0.03),
                    Some(-0.03),
                    Some(0.04),
                ],
            )
            .into(),
        ],
    )
    .unwrap()
}

fn no_one_day_factor_data() -> DataFrame {
    let data = base_factor_data();
    data.drop("forward_return_1D").unwrap()
}

fn ten_day_factor_data() -> DataFrame {
    let data = base_factor_data();
    let mut columns = data.columns().to_vec();
    columns.push(
        Series::new(
            "forward_return_10D".into(),
            &[
                Some(-0.12),
                Some(0.30),
                Some(-0.10),
                Some(0.25),
                Some(-0.08),
                Some(0.20),
            ],
        )
        .into(),
    );
    DataFrame::new(data.height(), columns).unwrap()
}

fn negative_base_factor_data() -> DataFrame {
    DataFrame::new(
        2,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["a", "b"]).into(),
            Series::new("group".into(), &["g1", "g1"]).into(),
            Series::new("factor".into(), &[Some(-1.0), Some(1.0)]).into(),
            Series::new("factor_quantile".into(), &[1_u32, 2]).into(),
            Series::new("forward_return_2D".into(), &[Some(-1.10), Some(0.44)]).into(),
            Series::new("forward_return_3D".into(), &[Some(-1.10), Some(0.728)]).into(),
        ],
    )
    .unwrap()
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
fn returns_report_tables_accept_two_digit_day_period_labels() {
    let report =
        create_returns_tear_sheet_data(&ten_day_factor_data(), ReturnsTearSheetOptions::default())
            .expect("returns report should build when 10D period labels are present");

    let means = report.data.mean_by_quantile.to_polars().unwrap();
    assert_eq!(
        means
            .column("period")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .take(3)
            .collect::<Vec<_>>(),
        vec!["10D", "1D", "5D"]
    );
}

#[test]
fn returns_summary_schema_and_metadata_are_stable() {
    let report =
        create_returns_tear_sheet_data(&base_factor_data(), ReturnsTearSheetOptions::default())
            .expect("returns report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::Returns);
    assert!(report.options.long_short);
    assert!(!report.options.group_neutral);
    assert!(!report.options.by_group);
    assert_eq!(report.data.summary.id(), "returns.summary");
    assert_eq!(report.data.summary.title(), "Returns Analysis");
    assert_eq!(
        report.data.summary.column_names(),
        vec!["metric", "period", "value", "count"]
    );
    assert_eq!(report.data.summary.sort_by(), &["metric", "period"]);
    assert_eq!(
        report.data.summary.columns()[0].role(),
        ColumnRole::Dimension
    );
    assert_eq!(report.data.summary.columns()[2].role(), ColumnRole::Measure);
    assert_eq!(
        report.data.summary.columns()[2].display(),
        Some(&DisplayFormat::new(DisplayUnit::DecimalReturn, 6))
    );

    let summary = report.data.summary.to_polars().unwrap();
    assert_eq!(
        summary
            .column("metric")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec![
            "alpha_annualized",
            "alpha_annualized",
            "alpha_raw",
            "alpha_raw",
            "beta",
            "beta",
            "bottom_quantile_mean",
            "bottom_quantile_mean",
            "mean_spread",
            "mean_spread",
            "top_quantile_mean",
            "top_quantile_mean",
        ]
    );
}

#[test]
fn shuffled_periods_use_shortest_numeric_period_as_conversion_base() {
    let report =
        create_returns_tear_sheet_data(&base_factor_data(), ReturnsTearSheetOptions::default())
            .unwrap();
    let means = report.data.mean_by_quantile.to_polars().unwrap();
    assert_eq!(
        report.metadata.periods(),
        &["1D".to_string(), "5D".to_string()]
    );
    assert_eq!(
        means
            .column("period")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .take(4)
            .collect::<Vec<_>>(),
        vec!["1D", "5D", "1D", "5D"]
    );

    let values = float_values(&means, "mean_return");
    assert_abs_diff_eq!(
        values[1].unwrap(),
        0.9_f64.powf(1.0 / 5.0) - 1.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        values[3].unwrap(),
        1.1_f64.powf(1.0 / 5.0) - 1.0,
        epsilon = 1e-12
    );
}

#[test]
fn negative_base_fractional_return_conversion_becomes_null() {
    let report = create_returns_tear_sheet_data(
        &negative_base_factor_data(),
        ReturnsTearSheetOptions {
            long_short: false,
            group_neutral: false,
            by_group: false,
        },
    )
    .unwrap();
    let means = report.data.mean_by_quantile.to_polars().unwrap();
    let values = float_values(&means, "mean_return");

    assert_eq!(values[0], Some(-1.10));
    assert_eq!(values[1], None);
    assert_abs_diff_eq!(values[2].unwrap(), 0.44, epsilon = 1e-12);
    assert_abs_diff_eq!(values[3].unwrap(), 0.44, epsilon = 1e-12);
}

#[test]
fn returns_detail_tables_and_optional_group_data_use_stable_contracts() {
    let report = create_returns_tear_sheet_data(
        &base_factor_data(),
        ReturnsTearSheetOptions {
            long_short: true,
            group_neutral: false,
            by_group: true,
        },
    )
    .unwrap();

    assert_eq!(
        report.data.mean_by_quantile.id(),
        "returns.mean_by_quantile"
    );
    assert_eq!(
        report.data.daily_by_quantile.id(),
        "returns.daily_by_quantile"
    );
    assert_eq!(report.data.daily_spread.id(), "returns.daily_spread");
    assert_eq!(report.data.factor_returns.id(), "returns.factor_returns");
    assert_eq!(
        report.data.group_mean_by_quantile.as_ref().unwrap().id(),
        "returns.group_mean_by_quantile"
    );
    assert_eq!(
        report.data.factor_cumulative_1d.as_ref().unwrap().id(),
        "returns.factor_cumulative_1d"
    );
    assert_eq!(
        report.data.quantile_cumulative_1d.as_ref().unwrap().id(),
        "returns.quantile_cumulative_1d"
    );

    assert_eq!(
        report.data.daily_by_quantile.column_names(),
        vec![
            "date",
            "factor_quantile",
            "period",
            "mean_return",
            "std_error",
            "count"
        ]
    );
    assert_eq!(
        report
            .data
            .group_mean_by_quantile
            .as_ref()
            .unwrap()
            .column_names(),
        vec![
            "factor_quantile",
            "group",
            "period",
            "mean_return",
            "std_error",
            "count"
        ]
    );
}

#[test]
fn cumulative_tables_are_absent_when_one_day_period_is_absent() {
    let report = create_returns_tear_sheet_data(
        &no_one_day_factor_data(),
        ReturnsTearSheetOptions::default(),
    )
    .unwrap();

    assert!(report.data.factor_cumulative_1d.is_none());
    assert!(report.data.quantile_cumulative_1d.is_none());
}

#[test]
fn quantile_cumulative_one_day_compounds_each_quantile_independently() {
    let report =
        create_returns_tear_sheet_data(&base_factor_data(), ReturnsTearSheetOptions::default())
            .unwrap();
    let cumulative = report
        .data
        .quantile_cumulative_1d
        .as_ref()
        .unwrap()
        .to_polars()
        .unwrap();
    let dates = cumulative
        .column("date")
        .unwrap()
        .as_materialized_series()
        .datetime()
        .unwrap();
    let quantiles = cumulative
        .column("factor_quantile")
        .unwrap()
        .as_materialized_series()
        .u32()
        .unwrap();
    let values = cumulative
        .column("cumulative_return")
        .unwrap()
        .as_materialized_series()
        .f64()
        .unwrap();

    assert_eq!(cumulative.height(), 6);
    assert_eq!(dates.physical().get(0), Some(JAN_1));
    assert_eq!(quantiles.get(0), Some(1));
    assert_abs_diff_eq!(values.get(0).unwrap(), 0.985, epsilon = 1e-12);
    assert_eq!(dates.physical().get(1), Some(JAN_1));
    assert_eq!(quantiles.get(1), Some(2));
    assert_abs_diff_eq!(values.get(1).unwrap(), 1.015, epsilon = 1e-12);
    assert_eq!(dates.physical().get(2), Some(JAN_2));
    assert_eq!(quantiles.get(2), Some(1));
    assert_abs_diff_eq!(values.get(2).unwrap(), 0.985 * 0.975, epsilon = 1e-12);
    assert_eq!(dates.physical().get(3), Some(JAN_2));
    assert_eq!(quantiles.get(3), Some(2));
    assert_abs_diff_eq!(values.get(3).unwrap(), 1.015 * 1.025, epsilon = 1e-12);
}
