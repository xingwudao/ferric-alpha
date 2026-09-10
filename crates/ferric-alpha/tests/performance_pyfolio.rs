mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{
    FerricAlphaError, Period, PortfolioOptions, WeightOptions, create_pyfolio_input,
};
use polars::prelude::{DataFrame, NamedFrom, Series};

#[test]
fn pyfolio_input_emits_returns_positions_and_benchmark() {
    let frame = common::simple_factor_data();

    let result = create_pyfolio_input(
        &frame,
        Period::new(1).unwrap(),
        None,
        PortfolioOptions::default(),
        Some(Period::new(1).unwrap()),
        None,
    )
    .unwrap();
    let returns = result.returns.column("return").unwrap().f64().unwrap();

    assert_eq!(returns.get(0), Some(0.0));
    assert_abs_diff_eq!(returns.get(1).unwrap(), -0.02, epsilon = 1e-12);
    assert_eq!(result.positions.height(), 8);
    assert!(result.benchmark.is_some());
}

#[test]
fn pyfolio_capital_positions_sum_to_aligned_portfolio_wealth() {
    let frame = common::simple_factor_data();

    let result = create_pyfolio_input(
        &frame,
        Period::new(1).unwrap(),
        Some(1_000.0),
        PortfolioOptions::default(),
        None,
        None,
    )
    .unwrap();
    let dates = result.positions.column("date").unwrap().datetime().unwrap();
    let values = result.positions.column("position").unwrap().f64().unwrap();
    let first_equity = dates
        .physical()
        .iter()
        .zip(values.iter())
        .filter_map(|(date, value)| (date == Some(common::JAN_1)).then_some(value.unwrap()))
        .sum::<f64>();
    let second_equity = dates
        .physical()
        .iter()
        .zip(values.iter())
        .filter_map(|(date, value)| (date == Some(common::JAN_2)).then_some(value.unwrap()))
        .sum::<f64>();

    assert_abs_diff_eq!(first_equity, 1_010.0, epsilon = 1e-10);
    assert_abs_diff_eq!(second_equity, 989.8, epsilon = 1e-10);
}

#[test]
fn pyfolio_rejects_invalid_capital_and_omits_absent_benchmark() {
    let frame = common::simple_factor_data();
    let error = create_pyfolio_input(
        &frame,
        Period::new(1).unwrap(),
        Some(0.0),
        PortfolioOptions::default(),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, FerricAlphaError::InvalidOption { .. }));

    let result = create_pyfolio_input(
        &frame,
        Period::new(1).unwrap(),
        None,
        PortfolioOptions::default(),
        Some(Period::new(3).unwrap()),
        None,
    )
    .unwrap();
    assert!(result.benchmark.is_none());
}

#[test]
fn pyfolio_capital_forward_fills_wealth_on_no_trade_session() {
    let frame = common::simple_factor_data();
    let day_3 = common::JAN_2 + 86_400_000;
    let sessions = polars::prelude::DataFrame::new(
        3,
        vec![common::datetime_column(&[
            common::JAN_1,
            common::JAN_2,
            day_3,
        ])],
    )
    .unwrap();

    let result = create_pyfolio_input(
        &frame,
        Period::new(5).unwrap(),
        Some(1_000.0),
        PortfolioOptions::default(),
        None,
        Some(&sessions),
    )
    .unwrap();
    let dates = result.positions.column("date").unwrap().datetime().unwrap();
    let values = result.positions.column("position").unwrap().f64().unwrap();
    let equity = dates
        .physical()
        .iter()
        .zip(values.iter())
        .filter_map(|(date, value)| (date == Some(day_3)).then_some(value.unwrap()))
        .sum::<f64>();

    assert_abs_diff_eq!(equity, 1004.2287210547987, epsilon = 1e-10);
}

#[test]
fn pyfolio_multiday_returns_use_overlapping_daily_sleeves() {
    let jan_3 = common::JAN_2 + 86_400_000;
    let frame = DataFrame::new(
        2,
        vec![
            common::datetime_column(&[common::JAN_1, common::JAN_2]),
            Series::new("asset".into(), &["A", "A"]).into(),
            Series::new("factor".into(), &[Some(1.0), Some(1.0)]).into(),
            Series::new("factor_quantile".into(), &[Some(1_u32), Some(1)]).into(),
            Series::new("forward_return_2D".into(), &[Some(0.21), Some(0.21)]).into(),
        ],
    )
    .unwrap();
    let sessions = DataFrame::new(
        3,
        vec![common::datetime_column(&[
            common::JAN_1,
            common::JAN_2,
            jan_3,
        ])],
    )
    .unwrap();

    let result = create_pyfolio_input(
        &frame,
        Period::new(2).unwrap(),
        Some(1_000.0),
        PortfolioOptions::default()
            .with_weight_options(WeightOptions::default().with_demeaned(false)),
        None,
        Some(&sessions),
    )
    .unwrap();
    let returns = result.returns.column("return").unwrap().f64().unwrap();

    assert_eq!(returns.get(0), Some(0.0));
    assert_abs_diff_eq!(returns.get(1).unwrap(), 0.10, epsilon = 1e-12);
    assert_abs_diff_eq!(returns.get(2).unwrap(), 0.10, epsilon = 1e-12);

    let dates = result.positions.column("date").unwrap().datetime().unwrap();
    let values = result.positions.column("position").unwrap().f64().unwrap();
    let day_3_equity = dates
        .physical()
        .iter()
        .zip(values.iter())
        .filter_map(|(date, value)| (date == Some(jan_3)).then_some(value.unwrap()))
        .sum::<f64>();
    assert_abs_diff_eq!(day_3_equity, 1_210.0, epsilon = 1e-10);
}
