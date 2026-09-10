mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{WeightOptions, factor_weights};
use polars::prelude::*;

const ALPHALENS_JAN_12_2000: i64 = 947_635_200_000;
fn alphalens_weight_fixture(values: &[Option<f64>]) -> DataFrame {
    alphalens_weight_fixture_from_rows(&[&values[0..5], &values[5..10], &values[10..15]])
}

fn alphalens_weight_fixture_from_rows(rows: &[&[Option<f64>]]) -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut groups = Vec::new();
    let mut factors = Vec::new();
    for (day, row) in rows.iter().enumerate() {
        for (asset_index, value) in row.iter().enumerate() {
            dates.push(ALPHALENS_JAN_12_2000 + day as i64 * 86_400_000);
            assets.push(["A", "B", "C", "D", "E"][asset_index]);
            groups.push(["Group1", "Group2", "Group1", "Group2", "Group1"][asset_index]);
            factors.push(*value);
        }
    }

    DataFrame::new(
        factors.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), assets).into(),
            Series::new("group".into(), groups).into(),
            Series::new("factor".into(), factors).into(),
        ],
    )
    .unwrap()
}

fn assert_optional_weights(actual: &DataFrame, expected: &[Option<f64>]) {
    let actual = actual
        .column("weight")
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
fn factor_weighted_demeaned_weights_are_dollar_neutral() {
    let result = factor_weights(&common::simple_factor_data(), WeightOptions::default()).unwrap();
    let dates = result.column("date").unwrap().datetime().unwrap();
    let weights = result.column("weight").unwrap().f64().unwrap();
    let first_date = dates.physical().get(0).unwrap();
    let total = (0..result.height())
        .filter(|index| dates.physical().get(*index) == Some(first_date))
        .filter_map(|index| weights.get(index))
        .sum::<f64>();

    assert_abs_diff_eq!(total, 0.0, epsilon = 1e-12);
}

#[test]
fn equal_weight_median_ties_receive_zero() {
    let frame = common::factor_data(
        &[common::JAN_1, common::JAN_1, common::JAN_1],
        &["A", "B", "C"],
        None,
        &[Some(-1.0), Some(0.0), Some(1.0)],
        &[Some(1), Some(3), Some(5)],
        &[Some(0.0), Some(0.0), Some(0.0)],
        &[Some(0.0), Some(0.0), Some(0.0)],
        &[Some(0.0), Some(0.0), Some(0.0)],
    );
    let result = factor_weights(&frame, WeightOptions::default().with_equal_weight(true)).unwrap();
    let weights: Vec<_> = result
        .column("weight")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();

    assert_eq!(weights[1], Some(0.0));
}

#[test]
fn alphalens_factor_weights_fixture_matches_public_factor_weighted_cases() {
    let frame = alphalens_weight_fixture(&[
        Some(3.0),
        Some(4.0),
        Some(2.0),
        Some(1.0),
        None,
        Some(3.0),
        Some(4.0),
        Some(-2.0),
        Some(-1.0),
        None,
        Some(3.0),
        None,
        None,
        Some(1.0),
        Some(4.0),
    ]);

    let raw = factor_weights(&frame, WeightOptions::default().with_demeaned(false)).unwrap();
    let demeaned = factor_weights(&frame, WeightOptions::default()).unwrap();

    let raw_weights: Vec<_> = raw
        .column("weight")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let demeaned_weights: Vec<_> = demeaned
        .column("weight")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    for (actual, expected) in raw_weights.iter().zip([
        0.30, 0.40, 0.20, 0.10, 0.30, 0.40, -0.20, -0.10, 0.375, 0.125, 0.50,
    ]) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 1e-12);
    }
    for (actual, expected) in demeaned_weights.iter().zip([
        0.125, 0.375, -0.125, -0.375, 0.20, 0.30, -0.30, -0.20, 0.10, -0.50, 0.40,
    ]) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 1e-12);
    }
}

#[test]
fn alphalens_factor_weights_fixture_matches_public_group_adjust_cases() {
    let group_weighted = alphalens_weight_fixture(&[
        Some(3.0),
        Some(4.0),
        Some(2.0),
        Some(1.0),
        None,
        Some(-3.0),
        Some(4.0),
        Some(-2.0),
        Some(1.0),
        None,
        Some(2.0),
        Some(2.0),
        Some(2.0),
        Some(3.0),
        Some(1.0),
    ]);
    let group_demeaned = alphalens_weight_fixture(&[
        Some(3.0),
        Some(4.0),
        Some(2.0),
        Some(1.0),
        None,
        Some(3.0),
        Some(4.0),
        Some(-2.0),
        Some(-1.0),
        None,
        Some(3.0),
        None,
        None,
        Some(1.0),
        Some(4.0),
    ]);

    let raw_group_adjusted = factor_weights(
        &group_weighted,
        WeightOptions::default()
            .with_demeaned(false)
            .with_group_adjust(true),
    )
    .unwrap();
    let demeaned_group_adjusted = factor_weights(
        &group_demeaned,
        WeightOptions::default().with_group_adjust(true),
    )
    .unwrap();

    assert_optional_weights(
        &raw_group_adjusted,
        &[
            Some(0.30),
            Some(0.40),
            Some(0.20),
            Some(0.10),
            Some(-0.30),
            Some(0.40),
            Some(-0.20),
            Some(0.10),
            Some(0.20),
            Some(0.20),
            Some(0.20),
            Some(0.30),
            Some(0.10),
        ],
    );
    assert_optional_weights(
        &demeaned_group_adjusted,
        &[
            Some(0.25),
            Some(0.25),
            Some(-0.25),
            Some(-0.25),
            Some(0.25),
            Some(0.25),
            Some(-0.25),
            Some(-0.25),
            Some(-0.50),
            None,
            Some(0.50),
        ],
    );
}

#[test]
fn alphalens_factor_weights_fixture_matches_public_equal_weight_cases() {
    let raw_equal_weight = alphalens_weight_fixture(&[
        Some(3.0),
        Some(4.0),
        Some(2.0),
        Some(1.0),
        Some(5.0),
        Some(3.0),
        Some(4.0),
        Some(-2.0),
        Some(-1.0),
        Some(5.0),
        Some(3.0),
        None,
        None,
        Some(1.0),
        None,
    ]);
    let demeaned_equal_weight = alphalens_weight_fixture(&[
        Some(1.0),
        Some(4.0),
        Some(2.0),
        Some(3.0),
        None,
        Some(1.0),
        Some(4.0),
        Some(-2.0),
        Some(-3.0),
        None,
        Some(3.0),
        None,
        None,
        Some(2.0),
        Some(7.0),
    ]);
    let group_equal_weight = alphalens_weight_fixture_from_rows(&[
        &[Some(3.0), Some(4.0), Some(2.0), Some(1.0), None],
        &[Some(-3.0), Some(4.0), Some(-2.0), Some(1.0), None],
        &[Some(3.0), None, None, Some(1.0), Some(4.0)],
        &[Some(3.0), None, None, Some(-1.0), Some(4.0)],
        &[Some(3.0), None, None, Some(1.0), Some(-4.0)],
    ]);
    let group_demeaned_equal_weight = alphalens_weight_fixture_from_rows(&[
        &[Some(1.0), Some(4.0), Some(2.0), Some(3.0), None],
        &[Some(3.0), Some(4.0), Some(-2.0), Some(-1.0), None],
        &[Some(3.0), None, None, Some(2.0), Some(7.0)],
        &[Some(3.0), None, None, Some(2.0), Some(-7.0)],
    ]);

    let raw = factor_weights(
        &raw_equal_weight,
        WeightOptions::default()
            .with_demeaned(false)
            .with_equal_weight(true),
    )
    .unwrap();
    let demeaned = factor_weights(
        &demeaned_equal_weight,
        WeightOptions::default().with_equal_weight(true),
    )
    .unwrap();
    let group = factor_weights(
        &group_equal_weight,
        WeightOptions::default()
            .with_demeaned(false)
            .with_group_adjust(true)
            .with_equal_weight(true),
    )
    .unwrap();
    let group_demeaned = factor_weights(
        &group_demeaned_equal_weight,
        WeightOptions::default()
            .with_group_adjust(true)
            .with_equal_weight(true),
    )
    .unwrap();

    assert_optional_weights(
        &raw,
        &[
            Some(0.20),
            Some(0.20),
            Some(0.20),
            Some(0.20),
            Some(0.20),
            Some(0.20),
            Some(0.20),
            Some(-0.20),
            Some(-0.20),
            Some(0.20),
            Some(0.50),
            Some(0.50),
        ],
    );
    assert_optional_weights(
        &demeaned,
        &[
            Some(-0.25),
            Some(0.25),
            Some(-0.25),
            Some(0.25),
            Some(0.25),
            Some(0.25),
            Some(-0.25),
            Some(-0.25),
            Some(0.0),
            Some(-0.50),
            Some(0.50),
        ],
    );
    assert_optional_weights(
        &group,
        &[
            Some(0.25),
            Some(0.25),
            Some(0.25),
            Some(0.25),
            Some(-0.25),
            Some(0.25),
            Some(-0.25),
            Some(0.25),
            Some(0.25),
            Some(0.50),
            Some(0.25),
            Some(0.25),
            Some(-0.50),
            Some(0.25),
            Some(0.25),
            Some(0.50),
            Some(-0.25),
        ],
    );
    assert_optional_weights(
        &group_demeaned,
        &[
            Some(-0.25),
            Some(0.25),
            Some(0.25),
            Some(-0.25),
            Some(0.25),
            Some(0.25),
            Some(-0.25),
            Some(-0.25),
            Some(-0.50),
            None,
            Some(0.50),
            Some(0.50),
            None,
            Some(-0.50),
        ],
    );
}
