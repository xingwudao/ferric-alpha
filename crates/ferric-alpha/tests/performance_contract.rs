mod common;

use ferric_alpha::{
    FerricAlphaError, IcOptions, MeanIcOptions, TimeBucket, WeightOptions,
    factor_information_coefficient, factor_weights,
};
use polars::prelude::*;

#[test]
fn rejects_malformed_forward_periods() {
    let mut frame = common::simple_factor_data();
    frame
        .rename("forward_return_1D", "forward_return_01D".into())
        .unwrap();

    let error = factor_information_coefficient(&frame, IcOptions::default()).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}

#[test]
fn rejects_missing_group_when_group_option_enabled() {
    let frame = common::factor_data(
        &[common::JAN_1],
        &["A"],
        None,
        &[Some(1.0)],
        &[Some(1)],
        &[Some(0.1)],
        &[Some(0.1)],
        &[Some(0.1)],
    );

    let error = factor_weights(&frame, WeightOptions::default().with_group_adjust(true))
        .expect_err("missing group should fail");

    assert!(matches!(error, FerricAlphaError::MissingColumn("group")));
}

#[test]
fn option_builders_set_expected_modes() {
    let frame = common::simple_factor_data();
    let ic = ferric_alpha::mean_information_coefficient(
        &frame,
        MeanIcOptions::default().with_by_time(Some(TimeBucket::Daily)),
    )
    .unwrap();

    assert!(ic.column("time_bucket").is_ok());
}

#[test]
fn rejects_wrong_forward_return_dtype() {
    let mut frame = common::simple_factor_data();
    frame
        .with_column(Series::new("forward_return_badD".into(), &["x"]).into())
        .unwrap();

    let error = factor_information_coefficient(&frame, IcOptions::default()).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidInput { .. }));
}
