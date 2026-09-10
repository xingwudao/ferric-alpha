mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{
    FerricAlphaError, Period, PortfolioOptions, ReturnOptions, WeightOptions, cumulative_returns,
    factor_cumulative_returns, factor_positions, factor_returns,
};
use polars::prelude::*;

#[test]
fn factor_cumulative_returns_compounds_selected_factor_period() {
    let frame = common::simple_factor_data();

    let result =
        factor_cumulative_returns(&frame, Period::new(1).unwrap(), PortfolioOptions::default())
            .unwrap();
    let wealth = result.column("cumulative_return").unwrap().f64().unwrap();

    assert_abs_diff_eq!(wealth.get(0).unwrap(), 1.01, epsilon = 1e-12);
    assert_abs_diff_eq!(wealth.get(1).unwrap(), 0.9898, epsilon = 1e-12);
}

#[test]
fn factor_positions_reuses_filtered_weight_universe_and_adds_cash() {
    let frame = common::simple_factor_data();
    let options = PortfolioOptions::default()
        .with_quantiles(Some(vec![1, 3, 5]))
        .with_groups(Some(vec!["g1".to_owned()]));

    let result = factor_positions(&frame, Period::new(1).unwrap(), options, None).unwrap();
    let assets = result.column("asset").unwrap().str().unwrap();

    assert_eq!(result.height(), 6);
    assert!(assets.iter().any(|asset| asset == Some("cash")));
    assert!(!assets.iter().any(|asset| asset == Some("C")));
}

#[test]
fn factor_cumulative_returns_rejects_a_missing_period() {
    let frame = common::simple_factor_data();

    let error =
        factor_cumulative_returns(&frame, Period::new(3).unwrap(), PortfolioOptions::default())
            .unwrap_err();

    assert!(matches!(error, FerricAlphaError::InvalidOption { .. }));
}

#[test]
fn portfolio_filters_reject_duplicates_and_empty_values() {
    let frame = common::simple_factor_data();
    let duplicate_quantile = PortfolioOptions::default().with_quantiles(Some(vec![1, 1]));
    let empty_group = PortfolioOptions::default().with_groups(Some(vec![String::new()]));

    assert!(matches!(
        factor_cumulative_returns(&frame, Period::new(1).unwrap(), duplicate_quantile).unwrap_err(),
        FerricAlphaError::InvalidOption { .. }
    ));
    assert!(matches!(
        factor_cumulative_returns(&frame, Period::new(1).unwrap(), empty_group).unwrap_err(),
        FerricAlphaError::InvalidOption { .. }
    ));
}

#[test]
fn portfolio_filter_rejects_an_empty_selected_universe() {
    let frame = common::simple_factor_data();
    let options = PortfolioOptions::default().with_quantiles(Some(vec![4]));

    let error = factor_positions(&frame, Period::new(1).unwrap(), options, None).unwrap_err();

    assert!(matches!(error, FerricAlphaError::InsufficientData { .. }));
}

#[test]
fn group_neutral_portfolio_requires_a_group_column() {
    let frame = common::factor_data(
        &[common::JAN_1, common::JAN_1],
        &["A", "B"],
        None,
        &[Some(-1.0), Some(1.0)],
        &[Some(1), Some(5)],
        &[Some(-0.01), Some(0.01)],
        &[Some(-0.05), Some(0.05)],
        &[Some(-0.1), Some(0.1)],
    );
    let options = PortfolioOptions::default()
        .with_weight_options(WeightOptions::default().with_group_adjust(true));

    let error = factor_positions(&frame, Period::new(1).unwrap(), options, None).unwrap_err();

    assert!(matches!(error, FerricAlphaError::MissingColumn { .. }));
}

#[test]
fn factor_cumulative_returns_matches_direct_weighted_return_composition() {
    let frame = common::simple_factor_data();
    let factor_wealth =
        factor_cumulative_returns(&frame, Period::new(1).unwrap(), PortfolioOptions::default())
            .unwrap();
    let returns = factor_returns(&frame, ReturnOptions::default()).unwrap();
    let periods = returns.column("period").unwrap().str().unwrap();
    let selected = returns
        .filter(&BooleanChunked::from_iter_values(
            "period_filter".into(),
            periods.iter().map(|value| value == Some("1D")),
        ))
        .unwrap();
    let direct_input = DataFrame::new(
        selected.height(),
        vec![
            selected.column("date").unwrap().clone(),
            selected
                .column("factor_return")
                .unwrap()
                .clone()
                .with_name("return".into()),
        ],
    )
    .unwrap();
    let direct_wealth = cumulative_returns(&direct_input).unwrap();

    assert_eq!(
        factor_wealth
            .column("cumulative_return")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        direct_wealth
            .column("cumulative_return")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    );
}
