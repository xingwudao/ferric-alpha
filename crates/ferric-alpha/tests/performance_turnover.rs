mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{Period, factor_rank_autocorrelation, quantile_turnover};
use polars::prelude::*;
use proptest::prelude::*;

const ALPHALENS_JAN_1_2015: i64 = 1_420_070_400_000;

fn alphalens_rank_fixture(rows: &[[f64; 4]]) -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut factors = Vec::new();
    for (day, row) in rows.iter().enumerate() {
        for (asset_index, value) in row.iter().enumerate() {
            dates.push(ALPHALENS_JAN_1_2015 + day as i64 * 86_400_000);
            assets.push(["A", "B", "C", "D"][asset_index]);
            factors.push(*value);
        }
    }

    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), assets).into(),
            Series::new("factor".into(), factors).into(),
        ],
    )
    .unwrap()
}

fn alphalens_quantile_fixture(rows: &[[u32; 4]]) -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut factors = Vec::new();
    let mut quantiles = Vec::new();
    for (day, row) in rows.iter().enumerate() {
        for (asset_index, quantile) in row.iter().enumerate() {
            dates.push(ALPHALENS_JAN_1_2015 + day as i64 * 86_400_000);
            assets.push(["A", "B", "C", "D"][asset_index]);
            factors.push(Some(asset_index as f64 + 1.0));
            quantiles.push(Some(*quantile));
        }
    }

    common::factor_data(
        &dates,
        &assets,
        None,
        &factors,
        &quantiles,
        &vec![Some(0.0); dates.len()],
        &vec![Some(0.0); dates.len()],
        &vec![Some(0.0); dates.len()],
    )
}

fn assert_optional_metric(actual: &DataFrame, column: &str, expected: &[Option<f64>]) {
    let actual = actual
        .column(column)
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
fn quantile_turnover_counts_new_current_members() {
    let day_3 = common::JAN_2 + 86_400_000;
    let frame = common::factor_data(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            common::JAN_2,
            common::JAN_2,
            day_3,
            day_3,
            day_3,
            day_3,
        ],
        &["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"],
        None,
        &[
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(4.0),
            Some(3.0),
            Some(2.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
        ],
        &[
            Some(5),
            Some(5),
            Some(1),
            Some(1),
            Some(5),
            Some(1),
            Some(5),
            Some(1),
            Some(1),
            Some(1),
            Some(5),
            Some(5),
        ],
        &[Some(0.0); 12],
        &[Some(0.0); 12],
        &[Some(0.0); 12],
    );

    let result = quantile_turnover(&frame, 5, Period::new(1).unwrap()).unwrap();
    let values = result.column("turnover").unwrap().f64().unwrap();

    assert_eq!(values.get(0), None);
    assert_abs_diff_eq!(values.get(1).unwrap(), 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(values.get(2).unwrap(), 0.5, epsilon = 1e-12);
}

#[test]
fn rank_autocorrelation_aligns_assets_and_uses_average_ranks() {
    let frame = common::simple_factor_data();

    let result = factor_rank_autocorrelation(&frame, Period::new(1).unwrap()).unwrap();
    let values = result.column("autocorrelation").unwrap().f64().unwrap();

    assert_eq!(values.get(0), None);
    assert_abs_diff_eq!(values.get(1).unwrap(), -1.0, epsilon = 1e-12);
}

#[test]
fn alphalens_rank_autocorrelation_fixture_matches_public_tests() {
    let stable = alphalens_rank_fixture(&[
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    let alternating = alphalens_rank_fixture(&[
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    let multi_lag = alphalens_rank_fixture(&[
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 4.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 4.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
        [2.0, 1.0, 4.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 4.0, 3.0],
        [2.0, 1.0, 4.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
    ]);

    let stable_result = factor_rank_autocorrelation(&stable, Period::new(1).unwrap()).unwrap();
    let alternating_result =
        factor_rank_autocorrelation(&alternating, Period::new(1).unwrap()).unwrap();
    let multi_lag_result =
        factor_rank_autocorrelation(&multi_lag, Period::new(3).unwrap()).unwrap();

    assert_optional_metric(
        &stable_result,
        "autocorrelation",
        &[None, Some(1.0), Some(1.0), Some(1.0)],
    );
    assert_optional_metric(
        &alternating_result,
        "autocorrelation",
        &[None, Some(-1.0), Some(-1.0), Some(-1.0)],
    );
    assert_optional_metric(
        &multi_lag_result,
        "autocorrelation",
        &[
            None,
            None,
            None,
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(0.6),
            Some(-0.6),
            Some(-1.0),
            Some(1.0),
            Some(-0.6),
            Some(-1.0),
        ],
    );
}

#[test]
fn alphalens_quantile_turnover_fixture_matches_public_short_window_tests() {
    let switching_four =
        alphalens_quantile_fixture(&[[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]]);
    let stable_three =
        alphalens_quantile_fixture(&[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]);
    let alternating_two =
        alphalens_quantile_fixture(&[[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1]]);

    assert_optional_metric(
        &quantile_turnover(&switching_four, 4, Period::new(1).unwrap()).unwrap(),
        "turnover",
        &[None, Some(1.0), Some(1.0), Some(0.0)],
    );
    assert_optional_metric(
        &quantile_turnover(&switching_four, 4, Period::new(2).unwrap()).unwrap(),
        "turnover",
        &[None, None, Some(0.0), Some(1.0)],
    );
    assert_optional_metric(
        &quantile_turnover(&switching_four, 4, Period::new(3).unwrap()).unwrap(),
        "turnover",
        &[None, None, None, Some(0.0)],
    );
    assert_optional_metric(
        &quantile_turnover(&stable_three, 3, Period::new(1).unwrap()).unwrap(),
        "turnover",
        &[None, Some(0.0), Some(0.0), Some(0.0)],
    );
    assert_optional_metric(
        &quantile_turnover(&stable_three, 3, Period::new(2).unwrap()).unwrap(),
        "turnover",
        &[None, None, Some(0.0), Some(0.0)],
    );
    assert_optional_metric(
        &quantile_turnover(&stable_three, 3, Period::new(3).unwrap()).unwrap(),
        "turnover",
        &[None, None, None, Some(0.0)],
    );
    assert_optional_metric(
        &quantile_turnover(&alternating_two, 2, Period::new(1).unwrap()).unwrap(),
        "turnover",
        &[None, Some(1.0), Some(1.0), Some(1.0)],
    );
}

#[test]
fn alphalens_quantile_turnover_fixture_matches_public_long_window_tests() {
    let four_day_period = alphalens_quantile_fixture(&[
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
    ]);
    let ten_day_period = alphalens_quantile_fixture(&[
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]);

    assert_optional_metric(
        &quantile_turnover(&four_day_period, 3, Period::new(4).unwrap()).unwrap(),
        "turnover",
        &[
            None,
            None,
            None,
            None,
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
        ],
    );
    assert_optional_metric(
        &quantile_turnover(&ten_day_period, 3, Period::new(10).unwrap()).unwrap(),
        "turnover",
        &[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(0.0),
            Some(1.0),
        ],
    );
}

#[test]
fn turnover_is_null_when_current_quantile_is_empty() {
    let frame = common::factor_data(
        &[common::JAN_1, common::JAN_2],
        &["A", "A"],
        None,
        &[Some(1.0), Some(1.0)],
        &[Some(5), Some(1)],
        &[Some(0.0); 2],
        &[Some(0.0); 2],
        &[Some(0.0); 2],
    );

    let result = quantile_turnover(&frame, 5, Period::new(1).unwrap()).unwrap();

    assert_eq!(
        result.column("turnover").unwrap().f64().unwrap().get(1),
        None
    );
}

#[test]
fn rank_autocorrelation_handles_ties_and_constant_paired_ranks() {
    let frame = common::factor_data(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            common::JAN_2,
        ],
        &["A", "B", "C", "A", "B", "C"],
        None,
        &[
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(2.0),
            Some(1.0),
        ],
        &[Some(1); 6],
        &[Some(0.0); 6],
        &[Some(0.0); 6],
        &[Some(0.0); 6],
    );

    let result = factor_rank_autocorrelation(&frame, Period::new(1).unwrap()).unwrap();

    assert_abs_diff_eq!(
        result
            .column("autocorrelation")
            .unwrap()
            .f64()
            .unwrap()
            .get(1)
            .unwrap(),
        -1.0,
        epsilon = 1e-12,
    );
}

#[test]
fn turnover_and_rank_autocorrelation_are_input_order_independent() {
    let frame = common::simple_factor_data();
    let reversed = frame.reverse();

    let ordered_turnover = quantile_turnover(&frame, 5, Period::new(1).unwrap()).unwrap();
    let reversed_turnover = quantile_turnover(&reversed, 5, Period::new(1).unwrap()).unwrap();
    assert_eq!(
        ordered_turnover
            .column("turnover")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        reversed_turnover
            .column("turnover")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    );

    let ordered_rank = factor_rank_autocorrelation(&frame, Period::new(1).unwrap()).unwrap();
    let reversed_rank = factor_rank_autocorrelation(&reversed, Period::new(1).unwrap()).unwrap();
    assert_eq!(
        ordered_rank
            .column("autocorrelation")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        reversed_rank
            .column("autocorrelation")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    );
}

#[test]
fn rank_autocorrelation_handles_changing_universe_non_finite_values_and_multi_lag() {
    let day_3 = common::JAN_2 + 86_400_000;
    let frame = common::factor_data(
        &[
            common::JAN_1,
            common::JAN_1,
            common::JAN_2,
            common::JAN_2,
            day_3,
            day_3,
        ],
        &["A", "B", "A", "C", "A", "B"],
        None,
        &[
            Some(1.0),
            Some(2.0),
            Some(f64::NAN),
            Some(1.0),
            Some(2.0),
            Some(1.0),
        ],
        &[Some(1); 6],
        &[Some(0.0); 6],
        &[Some(0.0); 6],
        &[Some(0.0); 6],
    );

    let result = factor_rank_autocorrelation(&frame, Period::new(2).unwrap()).unwrap();
    let values = result.column("autocorrelation").unwrap().f64().unwrap();

    assert_eq!(values.get(0), None);
    assert_eq!(values.get(1), None);
    assert_abs_diff_eq!(values.get(2).unwrap(), -1.0, epsilon = 1e-12);
}

proptest! {
    #[test]
    fn turnover_is_bounded_for_arbitrary_membership(previous in any::<u8>(), current in any::<u8>()) {
        let mut dates = Vec::new();
        let mut assets = Vec::new();
        let mut factors = Vec::new();
        let mut quantiles = Vec::new();
        for (date, bits) in [(common::JAN_1, previous), (common::JAN_2, current)] {
            for index in 0..4 {
                dates.push(date);
                assets.push(["A", "B", "C", "D"][index]);
                factors.push(Some(index as f64));
                quantiles.push(Some(if bits & (1 << index) != 0 { 5 } else { 1 }));
            }
        }
        let frame = common::factor_data(
            &dates,
            &assets,
            None,
            &factors,
            &quantiles,
            &[Some(0.0); 8],
            &[Some(0.0); 8],
            &[Some(0.0); 8],
        );

        let result = quantile_turnover(&frame, 5, Period::new(1).unwrap()).unwrap();
        if let Some(value) = result.column("turnover").unwrap().f64().unwrap().get(1) {
            prop_assert!((0.0..=1.0).contains(&value));
        }
    }
}
