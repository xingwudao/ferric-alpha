#[allow(dead_code)]
mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{FerricAlphaError, Period, cumulative_returns, positions};
use polars::prelude::*;

#[test]
fn cumulative_returns_compounds_simple_returns() {
    let frame = DataFrame::new(
        3,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_2, common::JAN_2 + 86_400_000]),
            Series::new("return".into(), &[0.1, -0.1, 0.2]).into(),
        ],
    )
    .unwrap();

    let result = cumulative_returns(&frame).unwrap();
    let wealth = result.column("cumulative_return").unwrap().f64().unwrap();

    assert_abs_diff_eq!(wealth.get(0).unwrap(), 1.1, epsilon = 1e-12);
    assert_abs_diff_eq!(wealth.get(1).unwrap(), 0.99, epsilon = 1e-12);
    assert_abs_diff_eq!(wealth.get(2).unwrap(), 1.188, epsilon = 1e-12);
}

#[test]
fn alphalens_cumulative_returns_fixture_matches_public_tests() {
    let cases = [
        ([1.0, 0.5, 1.0, 0.5, 0.5], [2.0, 3.0, 6.0, 9.0, 13.5]),
        (
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [1.1, 1.21, 1.331, 1.4641, 1.61051],
        ),
        (
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            [0.9, 0.81, 0.729, 0.6561, 0.59049],
        ),
    ];

    for (returns, expected) in cases {
        let frame = DataFrame::new(
            5,
            vec![
                common::datetime_column(&[
                    common::JAN_1,
                    common::JAN_2,
                    common::JAN_2 + 86_400_000,
                    common::JAN_2 + 2 * 86_400_000,
                    common::JAN_2 + 3 * 86_400_000,
                ]),
                Series::new("return".into(), returns).into(),
            ],
        )
        .unwrap();

        let result = cumulative_returns(&frame).unwrap();
        let wealth = result.column("cumulative_return").unwrap().f64().unwrap();
        for (actual, expected) in wealth.iter().zip(expected) {
            assert_abs_diff_eq!(actual.unwrap(), expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn cumulative_returns_groups_periods_and_treats_invalid_values_as_zero() {
    let frame = DataFrame::new(
        4,
        vec![
            common::datetime_column(&[common::JAN_2, common::JAN_1, common::JAN_2, common::JAN_1]),
            Series::new("period".into(), &["1D", "1D", "5D", "5D"]).into(),
            Series::new(
                "return".into(),
                &[Some(0.1), None, Some(f64::NAN), Some(0.2)],
            )
            .into(),
        ],
    )
    .unwrap();

    let result = cumulative_returns(&frame).unwrap();
    let periods = result.column("period").unwrap().str().unwrap();
    let wealth = result.column("cumulative_return").unwrap().f64().unwrap();

    assert_eq!(periods.get(0), Some("1D"));
    assert_abs_diff_eq!(wealth.get(0).unwrap(), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(wealth.get(1).unwrap(), 1.1, epsilon = 1e-12);
    assert_eq!(periods.get(2), Some("5D"));
    assert_abs_diff_eq!(wealth.get(2).unwrap(), 1.2, epsilon = 1e-12);
    assert_abs_diff_eq!(wealth.get(3).unwrap(), 1.2, epsilon = 1e-12);
}

#[test]
fn cumulative_returns_rejects_duplicate_series_keys() {
    let frame = DataFrame::new(
        2,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_1]),
            Series::new("return".into(), &[0.1, 0.2]).into(),
        ],
    )
    .unwrap();

    let error = cumulative_returns(&frame).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}

#[test]
fn cumulative_returns_rejects_non_finite_wealth() {
    let frame = DataFrame::new(
        2,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_2]),
            Series::new("return".into(), &[f64::MAX, f64::MAX]).into(),
        ],
    )
    .unwrap();

    let error = cumulative_returns(&frame).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}

#[test]
fn positions_combine_live_sleeves_and_emit_balancing_cash() {
    let day_3 = common::JAN_2 + 86_400_000;
    let day_4 = day_3 + 86_400_000;
    let weights = DataFrame::new(
        4,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_1, common::JAN_2, common::JAN_2]),
            Series::new("asset".into(), &["A", "B", "A", "B"]).into(),
            Series::new("weight".into(), &[0.75, -0.25, 0.25, -0.75]).into(),
        ],
    )
    .unwrap();
    let sessions = DataFrame::new(
        4,
        vec![common::datetime_column(&[
            common::JAN_1,
            common::JAN_2,
            day_3,
            day_4,
        ])],
    )
    .unwrap();

    let result = positions(&weights, Period::new(2).unwrap(), Some(&sessions)).unwrap();
    let positions = result.column("position").unwrap().f64().unwrap();

    assert_eq!(result.height(), 12);
    let expected = [
        0.75, -0.25, 0.5, 0.5, -0.5, 1.0, 0.25, -0.75, 1.5, 0.0, 0.0, 1.0,
    ];
    for (actual, expected) in positions.iter().zip(expected) {
        assert_abs_diff_eq!(actual.unwrap(), expected, epsilon = 1e-12);
    }
}

#[test]
fn positions_rejects_finite_weight_accumulation_overflow() {
    let weights = DataFrame::new(
        2,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_1]),
            Series::new("asset".into(), &["A", "B"]).into(),
            Series::new("weight".into(), &[f64::MAX, f64::MAX]).into(),
        ],
    )
    .unwrap();

    let error = positions(&weights, Period::new(1).unwrap(), None).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}

#[test]
fn positions_rejects_reserved_cash_asset() {
    let weights = DataFrame::new(
        1,
        vec![
            common::datetime_column(&[common::JAN_1]),
            Series::new("asset".into(), &["cash"]).into(),
            Series::new("weight".into(), &[1.0]).into(),
        ],
    )
    .unwrap();

    let error = positions(&weights, Period::new(1).unwrap(), None).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}

#[test]
fn positions_zero_gross_cancellation_emits_all_cash() {
    let weights = DataFrame::new(
        2,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_2]),
            Series::new("asset".into(), &["A", "A"]).into(),
            Series::new("weight".into(), &[1.0, -1.0]).into(),
        ],
    )
    .unwrap();

    let result = positions(&weights, Period::new(2).unwrap(), None).unwrap();
    let values = result.column("position").unwrap().f64().unwrap();

    assert_eq!(values.get(2), Some(0.0));
    assert_eq!(values.get(3), Some(1.0));
}

#[test]
fn positions_rejects_trade_date_missing_from_explicit_sessions() {
    let weights = DataFrame::new(
        1,
        vec![
            common::datetime_column(&[common::JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("weight".into(), &[1.0]).into(),
        ],
    )
    .unwrap();
    let sessions = DataFrame::new(1, vec![common::datetime_column(&[common::JAN_2])]).unwrap();

    let error = positions(&weights, Period::new(1).unwrap(), Some(&sessions)).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}

#[test]
fn positions_rejects_non_datetime_or_noncanonical_session_frames() {
    let weights = DataFrame::new(
        1,
        vec![
            common::datetime_column(&[common::JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("weight".into(), &[1.0]).into(),
        ],
    )
    .unwrap();
    let wrong_dtype =
        DataFrame::new(1, vec![Series::new("date".into(), &[common::JAN_1]).into()]).unwrap();
    let extra_column = DataFrame::new(
        1,
        vec![
            common::datetime_column(&[common::JAN_1]),
            Series::new("unexpected".into(), &[1]).into(),
        ],
    )
    .unwrap();

    assert!(matches!(
        positions(&weights, Period::new(1).unwrap(), Some(&wrong_dtype)).unwrap_err(),
        FerricAlphaError::InvalidDtype { .. }
    ));
    assert!(matches!(
        positions(&weights, Period::new(1).unwrap(), Some(&extra_column)).unwrap_err(),
        FerricAlphaError::InvalidInput { .. }
    ));
}

#[test]
fn cumulative_returns_preserves_timezone_for_empty_typed_input() {
    let timezone = TimeZone::opt_try_new(Some("America/New_York")).unwrap();
    let frame = DataFrame::new(
        0,
        vec![
            Series::new("date".into(), Vec::<i64>::new())
                .into_datetime(TimeUnit::Milliseconds, timezone.clone())
                .into_column(),
            Series::new("return".into(), Vec::<f64>::new()).into(),
        ],
    )
    .unwrap();

    let result = cumulative_returns(&frame).unwrap();

    assert_eq!(
        result.column("date").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Milliseconds, timezone)
    );
    assert_eq!(
        result.column("cumulative_return").unwrap().dtype(),
        &DataType::Float64
    );
    assert_eq!(result.height(), 0);
}

#[test]
fn positions_preserves_timezone_and_keeps_sleeve_past_short_calendar() {
    let timezone = TimeZone::opt_try_new(Some("America/New_York")).unwrap();
    let date = Series::new("date".into(), &[common::JAN_1])
        .into_datetime(TimeUnit::Milliseconds, timezone.clone())
        .into_column();
    let weights = DataFrame::new(
        1,
        vec![
            date.clone(),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("weight".into(), &[1.0]).into(),
        ],
    )
    .unwrap();
    let sessions = DataFrame::new(1, vec![date]).unwrap();

    let result = positions(&weights, Period::new(5).unwrap(), Some(&sessions)).unwrap();

    assert_eq!(
        result.column("date").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Milliseconds, timezone)
    );
    assert_eq!(
        result
            .column("position")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1.0), Some(0.0)]
    );
}

#[test]
fn positions_long_only_assets_sum_to_one_without_cash_exposure() {
    let weights = DataFrame::new(
        2,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_1]),
            Series::new("asset".into(), &["A", "B"]).into(),
            Series::new("weight".into(), &[1.0, 3.0]).into(),
        ],
    )
    .unwrap();

    let result = positions(&weights, Period::new(1).unwrap(), None).unwrap();
    let values = result.column("position").unwrap().f64().unwrap();

    assert_abs_diff_eq!(values.get(0).unwrap(), 0.25, epsilon = 1e-12);
    assert_abs_diff_eq!(values.get(1).unwrap(), 0.75, epsilon = 1e-12);
    assert_abs_diff_eq!(values.get(2).unwrap(), 0.0, epsilon = 1e-12);
}
