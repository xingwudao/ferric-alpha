mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{
    FerricAlphaError, MeanReturnByQuantileOptions, compute_mean_returns_spread,
    mean_return_by_quantile,
};
use polars::prelude::*;

const ALPHALENS_JAN_11_2015: i64 = 1_420_934_400_000;
const ALPHALENS_JAN_12_2015: i64 = 1_421_020_800_000;
const ALPHALENS_JAN_13_2015: i64 = 1_421_107_200_000;

fn alphalens_mean_return_fixture(
    daily_returns: [f64; 6],
    quantile_rows: [[u32; 6]; 3],
) -> DataFrame {
    let dates = [
        ALPHALENS_JAN_11_2015,
        ALPHALENS_JAN_11_2015,
        ALPHALENS_JAN_11_2015,
        ALPHALENS_JAN_11_2015,
        ALPHALENS_JAN_11_2015,
        ALPHALENS_JAN_11_2015,
        ALPHALENS_JAN_12_2015,
        ALPHALENS_JAN_12_2015,
        ALPHALENS_JAN_12_2015,
        ALPHALENS_JAN_12_2015,
        ALPHALENS_JAN_12_2015,
        ALPHALENS_JAN_12_2015,
        ALPHALENS_JAN_13_2015,
        ALPHALENS_JAN_13_2015,
        ALPHALENS_JAN_13_2015,
        ALPHALENS_JAN_13_2015,
        ALPHALENS_JAN_13_2015,
        ALPHALENS_JAN_13_2015,
    ];
    let assets = [
        "A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F",
    ];
    let groups = [
        "1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2", "2", "1", "1", "1", "2", "2", "2",
    ];
    let quantiles = [
        quantile_rows[0][0],
        quantile_rows[0][1],
        quantile_rows[0][2],
        quantile_rows[0][3],
        quantile_rows[0][4],
        quantile_rows[0][5],
        quantile_rows[1][0],
        quantile_rows[1][1],
        quantile_rows[1][2],
        quantile_rows[1][3],
        quantile_rows[1][4],
        quantile_rows[1][5],
        quantile_rows[2][0],
        quantile_rows[2][1],
        quantile_rows[2][2],
        quantile_rows[2][3],
        quantile_rows[2][4],
        quantile_rows[2][5],
    ];
    let returns = [
        daily_returns[0] - 1.0,
        daily_returns[1] - 1.0,
        daily_returns[2] - 1.0,
        daily_returns[3] - 1.0,
        daily_returns[4] - 1.0,
        daily_returns[5] - 1.0,
        daily_returns[0] - 1.0,
        daily_returns[1] - 1.0,
        daily_returns[2] - 1.0,
        daily_returns[3] - 1.0,
        daily_returns[4] - 1.0,
        daily_returns[5] - 1.0,
        daily_returns[0] - 1.0,
        daily_returns[1] - 1.0,
        daily_returns[2] - 1.0,
        daily_returns[3] - 1.0,
        daily_returns[4] - 1.0,
        daily_returns[5] - 1.0,
    ];

    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), assets).into(),
            Series::new("group".into(), groups).into(),
            Series::new("factor".into(), vec![Some(1.0); dates.len()]).into(),
            Series::new("factor_quantile".into(), quantiles).into(),
            Series::new("forward_return_1D".into(), returns).into(),
        ],
    )
    .unwrap()
}

#[test]
fn by_date_returns_quantile_date_period_means() {
    let result = mean_return_by_quantile(
        &common::simple_factor_data(),
        MeanReturnByQuantileOptions::default().with_by_date(true),
    )
    .unwrap();

    assert!(result.column("date").is_ok());
    assert!(result.column("mean_return").is_ok());
    assert!(result.column("std_error").is_ok());
}

#[test]
fn alphalens_mean_return_by_quantile_fixture_matches_public_tests() {
    let cases = vec![
        (
            [1.1, 1.2, 1.1, 1.2, 1.1, 1.2],
            [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
            false,
            vec![0.1, 0.2],
        ),
        (
            [1.1, 1.2, 1.1, 1.2, 1.1, 1.2],
            [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
            true,
            vec![0.1, 0.1, 0.2, 0.2],
        ),
        (
            [1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
            [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]],
            false,
            vec![0.15, 0.15, 0.15],
        ),
        (
            [1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
            [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]],
            true,
            vec![0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        ),
        (
            [1.5, 1.5, 1.2, 1.0, 1.0, 1.0],
            [[1, 1, 2, 2, 2, 2], [2, 2, 1, 2, 2, 2], [2, 2, 1, 2, 2, 2]],
            false,
            vec![0.3, 0.15],
        ),
        (
            [1.5, 1.5, 1.2, 1.0, 1.0, 1.0],
            [[1, 1, 3, 2, 2, 2], [3, 3, 1, 2, 2, 2], [3, 3, 1, 2, 2, 2]],
            false,
            vec![0.3, 0.0, 0.4],
        ),
        (
            [1.6, 1.6, 1.0, 1.0, 1.0, 1.0],
            [[1, 1, 2, 2, 2, 2], [2, 2, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1]],
            false,
            vec![0.2, 0.4],
        ),
        (
            [1.6, 1.6, 1.0, 1.6, 1.6, 1.0],
            [[1, 1, 2, 1, 1, 2], [2, 2, 1, 2, 2, 1], [2, 2, 1, 2, 2, 1]],
            true,
            vec![0.2, 0.2, 0.4, 0.4],
        ),
    ];

    for (daily_returns, quantile_rows, by_group, expected) in cases {
        let frame = alphalens_mean_return_fixture(daily_returns, quantile_rows);
        let result = mean_return_by_quantile(
            &frame,
            MeanReturnByQuantileOptions::default()
                .with_by_date(false)
                .with_by_group(by_group)
                .with_demeaned(false),
        )
        .unwrap();
        let values: Vec<_> = result
            .column("mean_return")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected) {
            assert_abs_diff_eq!(actual, &expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn by_date_false_uses_two_stage_daily_means() {
    let frame = common::factor_data(
        &[common::JAN_1, common::JAN_2, common::JAN_2, common::JAN_2],
        &["A", "A", "B", "C"],
        None,
        &[Some(1.0), Some(1.0), Some(2.0), Some(3.0)],
        &[Some(1), Some(1), Some(1), Some(1)],
        &[Some(0.0), Some(1.0), Some(1.0), Some(1.0)],
        &[Some(0.0), Some(1.0), Some(1.0), Some(1.0)],
        &[Some(0.0), Some(1.0), Some(1.0), Some(1.0)],
    );
    let result = mean_return_by_quantile(
        &frame,
        MeanReturnByQuantileOptions::default()
            .with_by_date(false)
            .with_demeaned(false),
    )
    .unwrap();
    let mean = result.column("mean_return").unwrap().f64().unwrap().get(0);

    assert_abs_diff_eq!(mean.unwrap(), 0.5, epsilon = 1e-12);
}

#[test]
fn spread_aligns_upper_and_lower_quantiles() {
    let means = mean_return_by_quantile(
        &common::simple_factor_data(),
        MeanReturnByQuantileOptions::default()
            .with_by_date(true)
            .with_demeaned(false),
    )
    .unwrap();
    let spread = compute_mean_returns_spread(&means, 5, 1, None).unwrap();

    assert!(spread.column("mean_return_difference").is_ok());
    assert!(spread.column("joint_std_error").is_ok());
}

#[test]
fn spread_rejects_incompatible_std_error_grouping_schema() {
    let grouped = DataFrame::new(
        2,
        vec![
            Series::new("factor_quantile".into(), &[1_u32, 5]).into(),
            Series::new("group".into(), &["g", "g"]).into(),
            Series::new("period".into(), &["1D", "1D"]).into(),
            Series::new("mean_return".into(), &[0.1, 0.2]).into(),
            Series::new("std_error".into(), &[0.01, 0.02]).into(),
        ],
    )
    .unwrap();
    let ungrouped_error = DataFrame::new(
        2,
        vec![
            Series::new("factor_quantile".into(), &[1_u32, 5]).into(),
            Series::new("period".into(), &["1D", "1D"]).into(),
            Series::new("std_error".into(), &[0.01, 0.02]).into(),
        ],
    )
    .unwrap();

    let error = compute_mean_returns_spread(&grouped, 5, 1, Some(&ungrouped_error)).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}
