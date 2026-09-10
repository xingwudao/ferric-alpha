mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{
    IcOptions, MeanIcOptions, TimeBucket, factor_information_coefficient,
    mean_information_coefficient,
};
use polars::prelude::*;

const ALPHALENS_JAN_1_2015: i64 = 1_420_070_400_000;
const ALPHALENS_JAN_2_2015: i64 = 1_420_156_800_000;
const ALPHALENS_WEEK_END_2015_01_04: i32 = 16_439;

fn alphalens_ic_fixture(forward_returns: &[f64; 8]) -> DataFrame {
    DataFrame::new(
        8,
        vec![
            common::datetime_column(&[
                ALPHALENS_JAN_1_2015,
                ALPHALENS_JAN_1_2015,
                ALPHALENS_JAN_1_2015,
                ALPHALENS_JAN_1_2015,
                ALPHALENS_JAN_2_2015,
                ALPHALENS_JAN_2_2015,
                ALPHALENS_JAN_2_2015,
                ALPHALENS_JAN_2_2015,
            ]),
            Series::new("asset".into(), &["A", "B", "C", "D", "A", "B", "C", "D"]).into(),
            Series::new("group".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]).into(),
            Series::new(
                "factor".into(),
                &[
                    Some(1.0),
                    Some(2.0),
                    Some(3.0),
                    Some(4.0),
                    Some(4.0),
                    Some(3.0),
                    Some(2.0),
                    Some(1.0),
                ],
            )
            .into(),
            Series::new("forward_return_1D".into(), forward_returns).into(),
        ],
    )
    .unwrap()
}

#[test]
fn computes_positive_and_negative_ic() {
    let frame = common::simple_factor_data();
    let result = factor_information_coefficient(&frame, IcOptions::default()).unwrap();
    let values: Vec<_> = result.column("ic").unwrap().f64().unwrap().iter().collect();

    assert_abs_diff_eq!(values[0].unwrap(), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(values[3].unwrap(), -1.0, epsilon = 1e-12);
}

#[test]
fn alphalens_information_coefficient_parameterized_fixture_matches_public_tests() {
    let negative = alphalens_ic_fixture(&[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]);
    let positive = alphalens_ic_fixture(&[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]);

    let negative_ic = factor_information_coefficient(&negative, IcOptions::default()).unwrap();
    let positive_ic = factor_information_coefficient(&positive, IcOptions::default()).unwrap();
    let by_group =
        factor_information_coefficient(&positive, IcOptions::default().with_by_group(true))
            .unwrap();
    let group_adjusted = factor_information_coefficient(
        &positive,
        IcOptions::default()
            .with_group_adjust(true)
            .with_by_group(true),
    )
    .unwrap();

    let negative_values: Vec<_> = negative_ic
        .column("ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let positive_values: Vec<_> = positive_ic
        .column("ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let by_group_values: Vec<_> = by_group
        .column("ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let group_adjusted_values: Vec<_> = group_adjusted
        .column("ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    assert_eq!(negative_values, vec![-1.0, -1.0]);
    assert_eq!(positive_values, vec![1.0, 1.0]);
    assert_eq!(by_group_values, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(group_adjusted_values, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn alphalens_mean_information_coefficient_parameterized_fixture_matches_public_tests() {
    let negative = alphalens_ic_fixture(&[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0]);
    let positive = alphalens_ic_fixture(&[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]);

    let daily_negative = mean_information_coefficient(
        &negative,
        MeanIcOptions::default().with_by_time(Some(TimeBucket::Daily)),
    )
    .unwrap();
    let weekly_positive = mean_information_coefficient(
        &positive,
        MeanIcOptions::default().with_by_time(Some(TimeBucket::Weekly)),
    )
    .unwrap();
    let grouped_positive =
        mean_information_coefficient(&positive, MeanIcOptions::default().with_by_group(true))
            .unwrap();
    let weekly_grouped_positive = mean_information_coefficient(
        &positive,
        MeanIcOptions::default()
            .with_by_group(true)
            .with_by_time(Some(TimeBucket::Weekly)),
    )
    .unwrap();

    let daily_values: Vec<_> = daily_negative
        .column("mean_ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let weekly_values: Vec<_> = weekly_positive
        .column("mean_ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let weekly_bucket = weekly_positive
        .column("time_bucket")
        .unwrap()
        .date()
        .unwrap()
        .phys
        .get(0);
    let grouped_values: Vec<_> = grouped_positive
        .column("mean_ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let weekly_grouped_values: Vec<_> = weekly_grouped_positive
        .column("mean_ic")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    assert_eq!(daily_values, vec![-1.0, -1.0]);
    assert_eq!(weekly_values, vec![1.0]);
    assert_eq!(weekly_bucket, Some(ALPHALENS_WEEK_END_2015_01_04));
    assert_eq!(grouped_values, vec![1.0, 1.0]);
    assert_eq!(weekly_grouped_values, vec![1.0, 1.0]);
}

#[test]
fn computes_ic_by_group_and_mean_buckets() {
    let frame = common::simple_factor_data();
    let by_group =
        factor_information_coefficient(&frame, IcOptions::default().with_by_group(true)).unwrap();
    let mean = mean_information_coefficient(
        &frame,
        MeanIcOptions::default()
            .with_by_group(true)
            .with_by_time(Some(TimeBucket::Weekly)),
    )
    .unwrap();

    assert!(by_group.column("group").is_ok());
    assert!(mean.column("time_bucket").is_ok());
    assert!(mean.column("mean_ic").is_ok());
}

#[test]
fn time_bucket_uses_datetime_timezone_local_date() {
    let date = 1_704_067_200_000_i64;
    let frame = DataFrame::new(
        2,
        vec![
            Series::new("date".into(), &[date, date])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("America/New_York")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["A", "B"]).into(),
            Series::new("factor".into(), &[Some(1.0), Some(2.0)]).into(),
            Series::new("forward_return_1D".into(), &[Some(0.1), Some(0.2)]).into(),
        ],
    )
    .unwrap();

    let result = mean_information_coefficient(
        &frame,
        MeanIcOptions::default().with_by_time(Some(TimeBucket::Daily)),
    )
    .unwrap();
    let bucket = result
        .column("time_bucket")
        .unwrap()
        .date()
        .unwrap()
        .phys
        .get(0);

    assert_eq!(bucket, Some(19_722));
}
