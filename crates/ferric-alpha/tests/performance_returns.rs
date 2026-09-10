mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{ReturnOptions, WeightOptions, factor_returns};
use polars::prelude::*;

const ALPHALENS_JAN_1_2015: i64 = 1_420_070_400_000;
const ALPHALENS_JAN_2_2015: i64 = 1_420_156_800_000;

fn alphalens_factor_returns_fixture(factors: &[f64; 8], forward_returns: &[f64; 8]) -> DataFrame {
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
            Series::new("factor".into(), factors).into(),
            Series::new("forward_return_1D".into(), forward_returns).into(),
        ],
    )
    .unwrap()
}

fn assert_factor_returns(actual: &DataFrame, expected: &[Option<f64>]) {
    let actual = actual
        .column("factor_return")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
            (None, None) => {}
            _ => panic!("expected {expected:?}, got {actual:?}"),
        }
    }
}

#[test]
fn computes_hand_checked_factor_returns() {
    let result = factor_returns(&common::simple_factor_data(), ReturnOptions::default()).unwrap();
    let values: Vec<_> = result
        .column("factor_return")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();

    assert_abs_diff_eq!(values[0].unwrap(), 0.01, epsilon = 1e-12);
}

#[test]
fn by_asset_preserves_asset_period_rows() {
    let result = factor_returns(
        &common::simple_factor_data(),
        ReturnOptions::default().with_by_asset(true),
    )
    .unwrap();

    assert_eq!(result.height(), 18);
    assert!(result.column("weighted_return").is_ok());
}

#[test]
fn alphalens_factor_returns_fixture_matches_public_tests() {
    let negative_raw = alphalens_factor_returns_fixture(
        &[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    );
    let constant_raw = alphalens_factor_returns_fixture(
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    );
    let group_adjusted_negative = alphalens_factor_returns_fixture(
        &[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    );
    let group_adjusted_mixed = alphalens_factor_returns_fixture(
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        &[1.0, 4.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0],
    );
    let group_adjusted_constant = alphalens_factor_returns_fixture(
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    );

    let raw = factor_returns(&negative_raw, ReturnOptions::default()).unwrap();
    let constant = factor_returns(&constant_raw, ReturnOptions::default()).unwrap();
    let group_negative = factor_returns(
        &group_adjusted_negative,
        ReturnOptions::default()
            .with_weight_options(WeightOptions::default().with_group_adjust(true)),
    )
    .unwrap();
    let group_mixed = factor_returns(
        &group_adjusted_mixed,
        ReturnOptions::default()
            .with_weight_options(WeightOptions::default().with_group_adjust(true)),
    )
    .unwrap();
    let group_constant = factor_returns(
        &group_adjusted_constant,
        ReturnOptions::default()
            .with_weight_options(WeightOptions::default().with_group_adjust(true)),
    )
    .unwrap();

    assert_factor_returns(&raw, &[Some(-1.25), Some(-1.25)]);
    assert_factor_returns(&constant, &[None, None]);
    assert_factor_returns(&group_negative, &[Some(-0.5), Some(-0.5)]);
    assert_factor_returns(&group_mixed, &[Some(1.0), Some(0.0)]);
    assert_factor_returns(&group_constant, &[None, None]);
}
