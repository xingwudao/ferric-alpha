#[allow(dead_code)]
mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{EventReturnsOptions, FerricAlphaError, average_cumulative_return_by_quantile};
use polars::prelude::{DataFrame, DataType, IntoColumn, NamedFrom, Series, TimeUnit};

const DAY_MS: i64 = 86_400_000;

fn returns_frame(dates: &[i64], assets: &[&str], returns: &[Option<f64>]) -> DataFrame {
    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(dates),
            Series::new("asset".into(), assets).into(),
            Series::new("return".into(), returns).into(),
        ],
    )
    .expect("test returns should build")
}

fn factor_frame(
    dates: &[i64],
    assets: &[&str],
    groups: Option<&[&str]>,
    quantiles: &[Option<u32>],
) -> DataFrame {
    common::factor_data(
        dates,
        assets,
        groups,
        &vec![Some(1.0); dates.len()],
        quantiles,
        &vec![Some(0.0); dates.len()],
        &vec![Some(0.0); dates.len()],
        &vec![Some(0.0); dates.len()],
    )
}

fn values_for_quantile(frame: &DataFrame, quantile: u32) -> Vec<(i32, f64, Option<f64>, u64)> {
    let quantiles = frame.column("factor_quantile").unwrap().u32().unwrap();
    let offsets = frame.column("offset").unwrap().i32().unwrap();
    let means = frame
        .column("mean_cumulative_return")
        .unwrap()
        .f64()
        .unwrap();
    let stds = frame
        .column("std_cumulative_return")
        .unwrap()
        .f64()
        .unwrap();
    let counts = frame.column("count").unwrap().u64().unwrap();

    (0..frame.height())
        .filter_map(|index| {
            (quantiles.get(index) == Some(quantile)).then_some((
                offsets.get(index).unwrap(),
                means.get(index).unwrap(),
                stds.get(index),
                counts.get(index).unwrap(),
            ))
        })
        .collect()
}

#[test]
fn averages_assets_first_then_event_dates_equally() {
    let jan_3 = common::JAN_2 + DAY_MS;
    let factor = factor_frame(
        &[common::JAN_1, common::JAN_1, common::JAN_2],
        &["A", "B", "A"],
        None,
        &[Some(1), Some(1), Some(1)],
    );
    let returns = returns_frame(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            jan_3,
            jan_3,
        ],
        &["A", "B", "A", "B", "A", "B"],
        &[
            Some(0.10),
            Some(0.20),
            Some(0.10),
            Some(0.00),
            Some(0.00),
            Some(0.20),
        ],
    );

    let result = average_cumulative_return_by_quantile(
        &factor,
        &returns,
        EventReturnsOptions {
            periods_before: 1,
            periods_after: 1,
            demeaned: false,
            group_adjust: false,
            by_group: false,
        },
    )
    .unwrap();

    let values = values_for_quantile(&result, 1);
    assert_eq!(values.len(), 3);
    assert_eq!(values[0].0, -1);
    assert_abs_diff_eq!(values[0].1, -0.09090909090909094, epsilon = 1e-12);
    assert_eq!(values[0].2, None);
    assert_eq!(values[0].3, 1);
    assert_eq!(values[1].0, 0);
    assert_abs_diff_eq!(values[1].1, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(values[1].2.unwrap(), 0.0, epsilon = 1e-12);
    assert_eq!(values[1].3, 2);
    assert_eq!(values[2].0, 1);
    assert_abs_diff_eq!(values[2].1, 0.025000000000000022, epsilon = 1e-12);
    assert_abs_diff_eq!(values[2].2.unwrap(), 0.03535533905932741, epsilon = 1e-12);
    assert_eq!(values[2].3, 2);
}

#[test]
fn keeps_edge_offsets_skips_absent_event_dates_and_imputes_sparse_cells() {
    let jan_3 = common::JAN_2 + DAY_MS;
    let absent = jan_3 + DAY_MS;
    let factor = factor_frame(
        &[common::JAN_1, common::JAN_2, absent],
        &["A", "A", "A"],
        None,
        &[Some(1), Some(1), Some(1)],
    );
    let returns = returns_frame(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            jan_3,
        ],
        &["A", "B", "A", "B", "A"],
        &[
            Some(0.10),
            Some(f64::NAN),
            None,
            Some(f64::INFINITY),
            Some(0.10),
        ],
    );

    let result = average_cumulative_return_by_quantile(
        &factor,
        &returns,
        EventReturnsOptions {
            periods_before: 1,
            periods_after: 1,
            demeaned: false,
            group_adjust: false,
            by_group: false,
        },
    )
    .unwrap();

    let values = values_for_quantile(&result, 1);
    assert_eq!(
        values.iter().map(|row| row.0).collect::<Vec<_>>(),
        [-1, 0, 1]
    );
    assert_abs_diff_eq!(values[0].1, 0.0, epsilon = 1e-12);
    assert_eq!(values[0].3, 1);
    assert_abs_diff_eq!(values[1].1, 0.0, epsilon = 1e-12);
    assert_eq!(values[1].3, 2);
    assert_abs_diff_eq!(values[2].1, 0.050000000000000044, epsilon = 1e-12);
    assert_eq!(values[2].3, 2);
}

#[test]
fn demeans_against_same_date_universe_cumulative_wealth() {
    let factor = factor_frame(
        &[common::JAN_1, common::JAN_1, common::JAN_2, common::JAN_2],
        &["A", "B", "A", "B"],
        None,
        &[Some(1), Some(2), Some(1), Some(2)],
    );
    let returns = returns_frame(
        &[common::JAN_1, common::JAN_1, common::JAN_2, common::JAN_2],
        &["A", "B", "A", "B"],
        &[Some(0.10), Some(0.30), Some(0.10), Some(0.00)],
    );

    let result = average_cumulative_return_by_quantile(
        &factor,
        &returns,
        EventReturnsOptions {
            periods_before: 0,
            periods_after: 1,
            demeaned: true,
            group_adjust: false,
            by_group: false,
        },
    )
    .unwrap();

    let values = values_for_quantile(&result, 1);
    assert_abs_diff_eq!(values[0].1, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(values[1].1, 0.050000000000000044, epsilon = 1e-12);
}

#[test]
fn group_adjust_and_by_group_partition_output() {
    let factor = factor_frame(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            common::JAN_2,
        ],
        &["A", "B", "C", "A", "B", "C"],
        Some(&["g1", "g1", "g2", "g1", "g1", "g2"]),
        &[Some(1), Some(2), Some(1), Some(1), Some(2), Some(1)],
    );
    let returns = returns_frame(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            common::JAN_2,
        ],
        &["A", "B", "C", "A", "B", "C"],
        &[
            Some(0.10),
            Some(0.30),
            Some(0.50),
            Some(0.10),
            Some(0.00),
            Some(0.10),
        ],
    );

    let result = average_cumulative_return_by_quantile(
        &factor,
        &returns,
        EventReturnsOptions {
            periods_before: 0,
            periods_after: 0,
            demeaned: true,
            group_adjust: true,
            by_group: true,
        },
    )
    .unwrap();

    assert_eq!(
        result.get_column_names(),
        [
            "factor_quantile",
            "group",
            "offset",
            "mean_cumulative_return",
            "std_cumulative_return",
            "count",
        ]
    );

    let groups = result.column("group").unwrap().str().unwrap();
    let means = result
        .column("mean_cumulative_return")
        .unwrap()
        .f64()
        .unwrap();
    let counts = result.column("count").unwrap().u64().unwrap();
    assert_eq!(groups.get(0), Some("g1"));
    assert_abs_diff_eq!(means.get(0).unwrap(), 0.0, epsilon = 1e-12);
    assert_eq!(counts.get(0), Some(2));
    assert_eq!(groups.get(1), Some("g2"));
    assert_abs_diff_eq!(means.get(1).unwrap(), 0.0, epsilon = 1e-12);
    assert_eq!(counts.get(1), Some(2));
}

#[test]
fn group_adjust_uses_full_same_date_group_when_not_demeaned() {
    let factor = factor_frame(
        &[common::JAN_1, common::JAN_1],
        &["A", "B"],
        Some(&["g1", "g1"]),
        &[Some(1), Some(2)],
    );
    let returns = returns_frame(
        &[common::JAN_1, common::JAN_1],
        &["A", "B"],
        &[Some(0.10), Some(0.30)],
    );

    let result = average_cumulative_return_by_quantile(
        &factor,
        &returns,
        EventReturnsOptions {
            periods_before: 0,
            periods_after: 0,
            demeaned: false,
            group_adjust: true,
            by_group: false,
        },
    )
    .unwrap();

    let values = values_for_quantile(&result, 1);
    assert_eq!(values.len(), 1);
    assert_abs_diff_eq!(values[0].1, 0.0, epsilon = 1e-12);
}

#[test]
fn validates_return_schema_and_usable_events() {
    let factor = factor_frame(&[common::JAN_1], &["A"], None, &[Some(1)]);
    let duplicate_returns = returns_frame(
        &[common::JAN_1, common::JAN_1],
        &["A", "A"],
        &[Some(0.01), Some(0.02)],
    );
    assert!(matches!(
        average_cumulative_return_by_quantile(
            &factor,
            &duplicate_returns,
            EventReturnsOptions::default()
        ),
        Err(FerricAlphaError::DuplicateKey)
    ));

    let wrong_return_dtype = DataFrame::new(
        1,
        vec![
            common::datetime_column(&[common::JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("return".into(), &["bad"]).into(),
        ],
    )
    .unwrap();
    assert!(matches!(
        average_cumulative_return_by_quantile(
            &factor,
            &wrong_return_dtype,
            EventReturnsOptions::default()
        ),
        Err(FerricAlphaError::InvalidDtype {
            column: "return",
            ..
        })
    ));

    let tz_returns = DataFrame::new(
        1,
        vec![
            Series::new("date".into(), &[common::JAN_1])
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap()
                .into_column(),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("return".into(), &[Some(0.01)]).into(),
        ],
    )
    .unwrap();
    assert!(matches!(
        average_cumulative_return_by_quantile(&factor, &tz_returns, EventReturnsOptions::default()),
        Err(FerricAlphaError::TimezoneMismatch)
    ));

    let no_group = factor_frame(&[common::JAN_1], &["A"], None, &[Some(1)]);
    let returns = returns_frame(&[common::JAN_1], &["A"], &[Some(0.01)]);
    assert!(matches!(
        average_cumulative_return_by_quantile(
            &no_group,
            &returns,
            EventReturnsOptions::default().with_by_group(true)
        ),
        Err(FerricAlphaError::MissingColumn("group"))
    ));

    let absent_factor = factor_frame(&[common::JAN_2], &["A"], None, &[Some(1)]);
    assert!(matches!(
        average_cumulative_return_by_quantile(
            &absent_factor,
            &returns,
            EventReturnsOptions::default()
        ),
        Err(FerricAlphaError::InsufficientData {
            operation: "average_cumulative_return_by_quantile"
        })
    ));
}
