use approx::assert_abs_diff_eq;
use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportKind, ReportMetadata, ReportTimeUnit,
    factor_quantile_statistics,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};
use serde_json::Value;

const JAN_1: i64 = 1_704_067_200_000;
const JAN_2: i64 = 1_704_153_600_000;
const JAN_3: i64 = 1_704_240_000_000;

fn timezone_factor_data() -> DataFrame {
    let timezone = TimeZone::opt_try_new(Some("America/New_York")).unwrap();
    DataFrame::new(
        5,
        vec![
            Series::new("date".into(), &[JAN_3, JAN_1, JAN_2, JAN_1, JAN_3])
                .into_datetime(TimeUnit::Milliseconds, timezone)
                .into_column(),
            Series::new("asset".into(), &["c", "b", "a", "a", "d"]).into(),
            Series::new(
                "group".into(),
                &[
                    "utilities",
                    "technology",
                    "technology",
                    "financials",
                    "utilities",
                ],
            )
            .into(),
            Series::new(
                "factor".into(),
                &[Some(4.0), Some(1.0), None, Some(2.0), Some(f64::NAN)],
            )
            .into(),
            Series::new("factor_quantile".into(), &[3_u32, 1, 2, 1, 3]).into(),
            Series::new("forward_return_10D".into(), &[0.10, 0.01, 0.02, 0.03, 0.04]).into(),
            Series::new("forward_return_1D".into(), &[0.01, 0.02, 0.03, 0.04, 0.05]).into(),
        ],
    )
    .unwrap()
}

fn column_display(table: &ferric_alpha::ReportTable, name: &str) -> Option<DisplayFormat> {
    table
        .columns()
        .iter()
        .find(|column| column.name() == name)
        .unwrap()
        .display()
        .copied()
}

#[test]
fn metadata_discovers_sorted_source_dimensions_without_generated_timestamps() {
    let metadata = ReportMetadata::from_factor_data(&timezone_factor_data(), ReportKind::Summary)
        .expect("metadata should build from valid factor data");

    assert_eq!(metadata.contract_version(), "ferric-alpha.tear-sheet/v1");
    assert_eq!(metadata.report_kind(), ReportKind::Summary);
    assert_eq!(metadata.source_rows(), 5);
    assert_eq!(metadata.date_start(), Some(JAN_1));
    assert_eq!(metadata.date_end(), Some(JAN_3));
    assert_eq!(metadata.date_time_unit(), ReportTimeUnit::Milliseconds);
    assert_eq!(metadata.date_timezone(), Some("America/New_York"));
    assert_eq!(metadata.periods(), &["1D".to_string(), "10D".to_string()]);
    assert_eq!(metadata.quantiles(), &[1, 2, 3]);
    assert_eq!(
        metadata.groups(),
        &[
            "financials".to_string(),
            "technology".to_string(),
            "utilities".to_string()
        ]
    );

    let serialized = serde_json::to_value(&metadata).unwrap();
    assert_eq!(
        serialized["report_kind"],
        Value::String("summary".to_string())
    );
    assert!(serialized.get("generated_at").is_none());
    assert!(serialized.get("created_at").is_none());
    assert!(serialized.get("timestamp").is_none());
}

#[test]
fn quantile_statistics_uses_finite_factors_and_exact_table_contract() {
    let table = factor_quantile_statistics(&timezone_factor_data())
        .expect("quantile statistics should build from valid factor data");

    assert_eq!(table.id(), "quantile.statistics");
    assert_eq!(table.title(), "Quantile Statistics");
    assert_eq!(table.sort_by(), &["factor_quantile"]);
    assert_eq!(
        table.column_names(),
        vec![
            "factor_quantile",
            "factor_min",
            "factor_max",
            "factor_mean",
            "factor_std",
            "count",
            "count_percent"
        ]
    );
    assert_eq!(table.columns()[0].role(), ColumnRole::Dimension);
    assert_eq!(table.columns()[1].role(), ColumnRole::Statistic);
    assert_eq!(
        column_display(&table, "count_percent"),
        Some(DisplayFormat::new(DisplayUnit::Percent, 4))
    );

    let frame = table.to_polars().unwrap();
    assert_eq!(frame.height(), 3);
    assert_eq!(
        frame
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        frame.column("factor_min").unwrap().dtype(),
        &polars::prelude::DataType::Float64
    );
    assert_eq!(
        frame.column("count").unwrap().dtype(),
        &polars::prelude::DataType::UInt64
    );

    let mins = frame.column("factor_min").unwrap().f64().unwrap();
    let maxes = frame.column("factor_max").unwrap().f64().unwrap();
    let means = frame.column("factor_mean").unwrap().f64().unwrap();
    let stds = frame.column("factor_std").unwrap().f64().unwrap();
    let counts = frame.column("count").unwrap().u64().unwrap();
    let percents = frame.column("count_percent").unwrap().f64().unwrap();

    assert_abs_diff_eq!(mins.get(0).unwrap(), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(maxes.get(0).unwrap(), 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(means.get(0).unwrap(), 1.5, epsilon = 1e-12);
    assert_abs_diff_eq!(
        stds.get(0).unwrap(),
        std::f64::consts::FRAC_1_SQRT_2,
        epsilon = 1e-12
    );
    assert_eq!(counts.get(0), Some(2));
    assert_abs_diff_eq!(percents.get(0).unwrap(), 2.0 / 3.0, epsilon = 1e-12);

    assert_eq!(mins.get(1), None);
    assert_eq!(maxes.get(1), None);
    assert_eq!(means.get(1), None);
    assert_eq!(stds.get(1), None);
    assert_eq!(counts.get(1), Some(0));
    assert_abs_diff_eq!(percents.get(1).unwrap(), 0.0, epsilon = 1e-12);

    assert_abs_diff_eq!(mins.get(2).unwrap(), 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(maxes.get(2).unwrap(), 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(means.get(2).unwrap(), 4.0, epsilon = 1e-12);
    assert_eq!(stds.get(2), None);
    assert_eq!(counts.get(2), Some(1));
    assert_abs_diff_eq!(percents.get(2).unwrap(), 1.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn quantile_statistics_emits_observed_positive_quantiles_for_all_invalid_factors() {
    let timezone = TimeZone::opt_try_new(Some("UTC")).unwrap();
    let data = DataFrame::new(
        3,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1, JAN_2])
                .into_datetime(TimeUnit::Milliseconds, timezone)
                .into_column(),
            Series::new("asset".into(), &["a", "b", "c"]).into(),
            Series::new(
                "factor".into(),
                &[Some(f64::NAN), None, Some(f64::INFINITY)],
            )
            .into(),
            Series::new("factor_quantile".into(), &[2_u32, 1, 2]).into(),
            Series::new("forward_return_1D".into(), &[0.01, 0.02, 0.03]).into(),
        ],
    )
    .unwrap();

    let table = factor_quantile_statistics(&data)
        .expect("all invalid factor values should produce an empty-stat table");
    let frame = table.to_polars().unwrap();

    assert_eq!(
        frame
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        frame
            .column("count")
            .unwrap()
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![0, 0]
    );
    for name in [
        "factor_min",
        "factor_max",
        "factor_mean",
        "factor_std",
        "count_percent",
    ] {
        assert_eq!(frame.column(name).unwrap().null_count(), 2);
    }
}
