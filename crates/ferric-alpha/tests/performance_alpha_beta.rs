mod common;

use ferric_alpha::{AlphaBetaOptions, FerricAlphaError, factor_alpha_beta};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit};

#[test]
fn alphalens_factor_alpha_beta_fixture_matches_public_tests() {
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
        ],
        &["A", "B", "C", "D", "A", "B", "C", "D"],
        Some(&["1", "1", "2", "2", "1", "1", "2", "2"]),
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
        &[
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(4),
            Some(3),
            Some(2),
            Some(1),
        ],
        &[
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
        ],
        &[
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
        ],
        &[
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
        ],
    );

    let result = factor_alpha_beta(&frame, None, AlphaBetaOptions::default()).unwrap();
    let periods = result.column("period").unwrap().str().unwrap();
    let annualized_alpha = result.column("annualized_alpha").unwrap().f64().unwrap();
    let beta = result.column("beta").unwrap().f64().unwrap();

    assert_eq!(result.height(), 3);
    for index in 0..result.height() {
        assert!(matches!(
            periods.get(index),
            Some("1D") | Some("5D") | Some("10D")
        ));
        approx::assert_abs_diff_eq!(annualized_alpha.get(index).unwrap(), -1.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(beta.get(index).unwrap(), 5.0 / 6.0, epsilon = 1e-12);
    }
}

#[test]
fn alpha_beta_emits_raw_annualized_beta_and_count() {
    let result = factor_alpha_beta(
        &common::simple_factor_data(),
        None,
        AlphaBetaOptions::default(),
    )
    .unwrap();

    assert_eq!(result.height(), 3);
    assert!(result.column("alpha").is_ok());
    assert!(result.column("annualized_alpha").is_ok());
    assert!(result.column("beta").is_ok());
    assert!(result.column("count").is_ok());
}

#[test]
fn alpha_beta_rejects_custom_returns_with_mismatched_datetime_unit() {
    let returns = DataFrame::new(
        2,
        vec![
            Series::new(
                "date".into(),
                &[common::JAN_1 * 1_000, common::JAN_2 * 1_000],
            )
            .cast(&polars::prelude::DataType::Datetime(
                TimeUnit::Microseconds,
                None,
            ))
            .unwrap()
            .into_column(),
            Series::new("period".into(), &["1D", "1D"]).into_column(),
            Series::new("factor_return".into(), &[Some(0.01), Some(0.02)]).into_column(),
        ],
    )
    .unwrap();

    let error = factor_alpha_beta(
        &common::simple_factor_data(),
        Some(&returns),
        AlphaBetaOptions::default(),
    )
    .expect_err("datetime unit mismatch should fail instead of producing an empty fit");

    assert!(matches!(error, FerricAlphaError::TimezoneMismatch));
}
