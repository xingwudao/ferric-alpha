use approx::assert_abs_diff_eq;
use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, FerricAlphaError, InformationTearSheetOptions,
    ReportColumnData, ReportKind, biased_excess_kurtosis, biased_skew,
    create_information_tear_sheet_data, one_sample_t_test, sample_std,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};

const JAN_1_2024_UTC_NEW_YORK_DEC_31: i64 = 1_704_070_800_000;
const JAN_2_2024_UTC: i64 = 1_704_153_600_000;
const DAY_MS: i64 = 86_400_000;

fn information_factor_data() -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut groups = Vec::new();
    let mut factors = Vec::new();
    let mut quantiles = Vec::new();
    let mut one_day = Vec::new();
    let mut five_day = Vec::new();
    let mut ten_day = Vec::new();
    let mut fifteen_day = Vec::new();
    let mut twenty_day = Vec::new();

    for day in 0..23 {
        let date = JAN_2_2024_UTC + i64::from(day) * DAY_MS;
        let one_day_returns = if day % 2 == 0 {
            [0.01, 0.02, 0.03, 0.04]
        } else {
            [0.04, 0.03, 0.02, 0.01]
        };
        for asset_index in 0..4 {
            dates.push(date);
            assets.push(["a", "b", "c", "d"][asset_index]);
            groups.push(if asset_index < 2 { "g1" } else { "g2" });
            factors.push(Some((asset_index + 1) as f64));
            quantiles.push((asset_index + 1) as u32);
            one_day.push(Some(one_day_returns[asset_index]));
            five_day.push(Some((asset_index + 1) as f64 * 0.01));
            ten_day.push(if day == 0 {
                None
            } else {
                Some(one_day_returns[asset_index])
            });
            fifteen_day.push(if day == 0 {
                Some(one_day_returns[asset_index])
            } else {
                None
            });
            twenty_day.push(None::<f64>);
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
            Series::new("group".into(), groups).into(),
            Series::new("factor".into(), factors).into(),
            Series::new("factor_quantile".into(), quantiles).into(),
            Series::new("forward_return_1D".into(), one_day).into(),
            Series::new("forward_return_5D".into(), five_day).into(),
            Series::new("forward_return_10D".into(), ten_day).into(),
            Series::new("forward_return_15D".into(), fifteen_day).into(),
            Series::new("forward_return_20D".into(), twenty_day).into(),
        ],
    )
    .unwrap()
}

fn timezone_boundary_factor_data() -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut groups = Vec::new();
    let mut factors = Vec::new();
    let mut quantiles = Vec::new();
    let mut one_day = Vec::new();

    for (row_date, returns) in [
        (JAN_1_2024_UTC_NEW_YORK_DEC_31, [0.04, 0.03, 0.02, 0.01]),
        (JAN_2_2024_UTC, [0.01, 0.02, 0.03, 0.04]),
    ] {
        for asset_index in 0..4 {
            dates.push(row_date);
            assets.push(["a", "b", "c", "d"][asset_index]);
            groups.push(if asset_index < 2 { "g1" } else { "g2" });
            factors.push(Some((asset_index + 1) as f64));
            quantiles.push((asset_index + 1) as u32);
            one_day.push(Some(returns[asset_index]));
        }
    }

    DataFrame::new(
        dates.len(),
        vec![
            Series::new("date".into(), dates)
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("America/New_York")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), assets).into(),
            Series::new("group".into(), groups).into(),
            Series::new("factor".into(), factors).into(),
            Series::new("factor_quantile".into(), quantiles).into(),
            Series::new("forward_return_1D".into(), one_day).into(),
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
fn information_summary_schema_metadata_and_degenerate_periods_are_stable() {
    let report = create_information_tear_sheet_data(
        &information_factor_data(),
        InformationTearSheetOptions::default(),
    )
    .expect("information report should build");

    assert_eq!(report.metadata.report_kind(), ReportKind::Information);
    assert!(!report.options.group_neutral);
    assert!(!report.options.by_group);
    assert_eq!(report.options.rolling_window, 22);
    assert_eq!(report.data.summary.id(), "information.summary");
    assert_eq!(report.data.summary.title(), "Information Analysis");
    assert_eq!(
        report.data.summary.column_names(),
        vec![
            "period",
            "ic_mean",
            "ic_std",
            "risk_adjusted_ic",
            "t_stat",
            "p_value",
            "skew",
            "excess_kurtosis",
            "count",
        ]
    );
    assert_eq!(report.data.summary.sort_by(), &["period"]);
    assert_eq!(
        report.data.summary.columns()[0].role(),
        ColumnRole::Dimension
    );
    assert_eq!(
        report.data.summary.columns()[1].role(),
        ColumnRole::Statistic
    );
    assert_eq!(
        report.data.summary.columns()[8].display(),
        Some(&DisplayFormat::new(DisplayUnit::Count, 0))
    );

    let summary = report.data.summary.to_polars().unwrap();
    assert_eq!(
        string_values(&summary, "period"),
        vec![
            Some("10D".to_string()),
            Some("15D".to_string()),
            Some("1D".to_string()),
            Some("20D".to_string()),
            Some("5D".to_string()),
        ]
    );

    let counts = u64_values(&summary, "count");
    assert_eq!(counts, vec![Some(22), Some(1), Some(23), Some(0), Some(23)]);

    let ic_mean = float_values(&summary, "ic_mean");
    let ic_std = float_values(&summary, "ic_std");
    let risk_adjusted = float_values(&summary, "risk_adjusted_ic");
    let t_stats = float_values(&summary, "t_stat");
    let p_values = float_values(&summary, "p_value");
    let skew = float_values(&summary, "skew");
    let kurtosis = float_values(&summary, "excess_kurtosis");

    let one_day_values = (0..23)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let t_test = one_sample_t_test(&one_day_values, 0.0).unwrap();
    assert_abs_diff_eq!(ic_mean[2].unwrap(), 1.0 / 23.0, epsilon = 1e-12);
    assert_abs_diff_eq!(
        ic_std[2].unwrap(),
        sample_std(&one_day_values).unwrap().unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        risk_adjusted[2].unwrap(),
        ic_mean[2].unwrap() / ic_std[2].unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        t_stats[2].unwrap(),
        t_test.statistic.unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        p_values[2].unwrap(),
        t_test.p_value.unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        skew[2].unwrap(),
        biased_skew(&one_day_values).unwrap().unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        kurtosis[2].unwrap(),
        biased_excess_kurtosis(&one_day_values).unwrap().unwrap(),
        epsilon = 1e-12
    );

    for row in [1, 3, 4] {
        assert_eq!(risk_adjusted[row], None);
        assert_eq!(t_stats[row], None);
        assert_eq!(p_values[row], None);
        assert_eq!(skew[row], None);
        assert_eq!(kurtosis[row], None);
    }
}

#[test]
fn ic_by_date_and_rolling_tables_are_long_form_with_complete_windows() {
    let report = create_information_tear_sheet_data(
        &information_factor_data(),
        InformationTearSheetOptions::default(),
    )
    .unwrap();

    assert_eq!(report.data.ic_by_date.id(), "information.ic_by_date");
    assert_eq!(
        report.data.ic_by_date.column_names(),
        vec!["date", "period", "ic"]
    );

    let by_date = report.data.ic_by_date.to_polars().unwrap();
    assert_eq!(by_date.height(), 115);
    assert_eq!(
        float_values(&by_date, "ic")
            .into_iter()
            .take(5)
            .collect::<Vec<_>>(),
        vec![None, Some(1.0), Some(1.0), None, Some(1.0)]
    );

    assert_eq!(report.data.ic_rolling.id(), "information.ic_rolling");
    assert_eq!(
        report.data.ic_rolling.column_names(),
        vec!["date", "period", "rolling_mean_ic", "count"]
    );
    let rolling = report.data.ic_rolling.to_polars().unwrap();
    let rolling_periods = string_values(&rolling, "period");
    let rolling_values = float_values(&rolling, "rolling_mean_ic");
    let rolling_counts = u64_values(&rolling, "count");
    let one_day_rows = rolling_periods
        .iter()
        .enumerate()
        .filter_map(|(index, period)| (period.as_deref() == Some("1D")).then_some(index))
        .collect::<Vec<_>>();

    assert_eq!(one_day_rows.len(), 23);
    for (offset, row) in one_day_rows.iter().take(21).enumerate() {
        assert_eq!(rolling_values[*row], None);
        assert_eq!(rolling_counts[*row], Some((offset + 1) as u64));
    }
    assert_abs_diff_eq!(
        rolling_values[one_day_rows[21]].unwrap(),
        0.0,
        epsilon = 1e-12
    );
    assert_eq!(rolling_counts[one_day_rows[21]], Some(22));
    assert_abs_diff_eq!(
        rolling_values[one_day_rows[22]].unwrap(),
        0.0,
        epsilon = 1e-12
    );
    assert_eq!(rolling_counts[one_day_rows[22]], Some(22));
}

#[test]
fn zero_rolling_window_is_invalid_option() {
    let err = create_information_tear_sheet_data(
        &information_factor_data(),
        InformationTearSheetOptions {
            group_neutral: false,
            by_group: false,
            rolling_window: 0,
        },
    )
    .expect_err("zero rolling window should be rejected");

    assert!(matches!(
        err,
        FerricAlphaError::InvalidOption {
            option: "rolling_window",
            ..
        }
    ));
}

#[test]
fn monthly_group_and_qq_tables_use_stable_order_and_optional_grouping() {
    let report = create_information_tear_sheet_data(
        &timezone_boundary_factor_data(),
        InformationTearSheetOptions {
            group_neutral: false,
            by_group: true,
            rolling_window: 2,
        },
    )
    .unwrap();

    assert_eq!(report.data.ic_monthly.id(), "information.ic_monthly");
    assert_eq!(
        report.data.ic_monthly.column_names(),
        vec!["time_bucket", "period", "mean_ic", "count"]
    );
    let monthly = report.data.ic_monthly.to_polars().unwrap();
    assert_eq!(
        monthly
            .column("time_bucket")
            .unwrap()
            .as_materialized_series()
            .i32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(19_692), Some(19_723)]
    );
    assert_eq!(
        float_values(&monthly, "mean_ic"),
        vec![Some(-1.0), Some(1.0)]
    );

    let grouped = report
        .data
        .ic_by_group
        .as_ref()
        .expect("by_group=true should include grouped mean IC");
    assert_eq!(grouped.id(), "information.ic_by_group");
    assert_eq!(
        grouped.column_names(),
        vec!["group", "period", "mean_ic", "count"]
    );
    let grouped = grouped.to_polars().unwrap();
    assert_eq!(
        string_values(&grouped, "group"),
        vec![Some("g1".to_string()), Some("g2".to_string())]
    );
    assert_eq!(
        float_values(&grouped, "mean_ic"),
        vec![Some(0.0), Some(0.0)]
    );

    assert_eq!(report.data.qq_normal.id(), "information.qq_normal");
    assert_eq!(
        report.data.qq_normal.column_names(),
        vec!["period", "probability", "theoretical", "observed"]
    );
    let qq = report.data.qq_normal.to_polars().unwrap();
    assert_eq!(
        string_values(&qq, "period"),
        vec![Some("1D".to_string()), Some("1D".to_string())]
    );
    assert_eq!(
        float_values(&qq, "probability"),
        vec![Some(0.25), Some(0.75)]
    );
    assert!(float_values(&qq, "theoretical")[0] < float_values(&qq, "theoretical")[1]);
    assert_eq!(
        float_values(&qq, "observed"),
        vec![Some(-0.7071067811865475), Some(0.7071067811865475)]
    );

    let no_group_report = create_information_tear_sheet_data(
        &timezone_boundary_factor_data(),
        InformationTearSheetOptions {
            group_neutral: false,
            by_group: false,
            rolling_window: 2,
        },
    )
    .unwrap();
    assert!(no_group_report.data.ic_by_group.is_none());
}

#[test]
fn qq_omits_constant_one_value_and_empty_finite_periods() {
    let report = create_information_tear_sheet_data(
        &information_factor_data(),
        InformationTearSheetOptions::default(),
    )
    .unwrap();

    let qq = report.data.qq_normal.to_polars().unwrap();
    let periods = string_values(&qq, "period");
    assert!(
        periods
            .iter()
            .all(|period| matches!(period.as_deref(), Some("1D" | "10D")))
    );
    assert!(!periods.iter().any(|period| period.as_deref() == Some("5D")));
    assert!(
        !periods
            .iter()
            .any(|period| period.as_deref() == Some("15D"))
    );
    assert!(
        !periods
            .iter()
            .any(|period| period.as_deref() == Some("20D"))
    );

    let probabilities = float_values(&qq, "probability");
    for period in ["10D", "1D"] {
        let period_probabilities = periods
            .iter()
            .zip(&probabilities)
            .filter_map(|(row_period, probability)| {
                (row_period.as_deref() == Some(period)).then_some(*probability)
            })
            .collect::<Vec<_>>();
        assert!(
            period_probabilities
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
        );
    }

    let first_column_data = report.data.qq_normal.columns()[0].data();
    assert!(matches!(first_column_data, ReportColumnData::String(_)));
}
