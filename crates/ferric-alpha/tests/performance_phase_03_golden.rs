use std::collections::BTreeSet;

use approx::assert_relative_eq;
use chrono::NaiveDate;
use ferric_alpha::{
    Period, PortfolioOptions, cumulative_returns, factor_positions, factor_rank_autocorrelation,
    positions, quantile_turnover,
};
use polars::prelude::*;
use serde::Deserialize;

const EPSILON: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct Input {
    session_dates: Vec<String>,
    simple_returns: Vec<ReturnRow>,
    weights: Vec<WeightRow>,
    rows: Vec<FactorRow>,
}

#[derive(Debug, Deserialize)]
struct ReturnRow {
    date: String,
    #[serde(rename = "return")]
    value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct WeightRow {
    date: String,
    asset: String,
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct FactorRow {
    date: String,
    asset: String,
    group: String,
    factor: f64,
    factor_quantile: u32,
    #[serde(rename = "forward_return_1D")]
    one_day: f64,
    #[serde(rename = "forward_return_3D")]
    three_day: f64,
}

#[derive(Debug, Deserialize)]
struct GoldenRecord {
    date: String,
    asset: Option<String>,
    turnover: Option<f64>,
    autocorrelation: Option<f64>,
    cumulative_return: Option<f64>,
    position: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Defect {
    exception: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct Defects {
    factor_cumulative_returns: Defect,
    create_pyfolio_input: Defect,
}

#[test]
fn phase_03_matches_callable_alphalens_golden_outputs() {
    let input = load_input();
    assert!(input.rows.iter().any(|left| {
        input.rows.iter().any(|right| {
            left.date == right.date && left.asset != right.asset && left.factor == right.factor
        })
    }));
    let factor_data = factor_frame(&input);

    let turnover = quantile_turnover(&factor_data, 5, Period::new(2).unwrap()).unwrap();
    assert_schema(
        &turnover,
        &[
            ("date", DataType::Datetime(TimeUnit::Milliseconds, None)),
            ("factor_quantile", DataType::UInt32),
            ("period", DataType::String),
            ("turnover", DataType::Float64),
        ],
    );
    assert!(
        turnover
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .all(|value| value == Some(5))
    );
    assert!(
        turnover
            .column("period")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .all(|value| value == Some("2D"))
    );
    assert_series_golden(
        &turnover,
        "turnover",
        load_records(include_str!("../../../tests/golden/phase-03/turnover.json")),
        |record| record.turnover,
    );

    let autocorrelation =
        factor_rank_autocorrelation(&factor_data, Period::new(1).unwrap()).unwrap();
    assert_schema(
        &autocorrelation,
        &[
            ("date", DataType::Datetime(TimeUnit::Milliseconds, None)),
            ("period", DataType::String),
            ("autocorrelation", DataType::Float64),
        ],
    );
    assert!(
        autocorrelation
            .column("period")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .all(|value| value == Some("1D"))
    );
    assert_series_golden(
        &autocorrelation,
        "autocorrelation",
        load_records(include_str!(
            "../../../tests/golden/phase-03/rank_autocorrelation.json"
        )),
        |record| record.autocorrelation,
    );

    let cumulative = cumulative_returns(&return_frame(&input)).unwrap();
    assert_schema(
        &cumulative,
        &[
            ("date", DataType::Datetime(TimeUnit::Milliseconds, None)),
            ("cumulative_return", DataType::Float64),
        ],
    );
    assert_series_golden(
        &cumulative,
        "cumulative_return",
        load_records(include_str!(
            "../../../tests/golden/phase-03/cumulative_returns.json"
        )),
        |record| record.cumulative_return,
    );

    let sessions = session_frame(&input);
    let raw_positions = positions(
        &weight_frame(&input),
        Period::new(2).unwrap(),
        Some(&sessions),
    )
    .unwrap();
    assert_schema(
        &raw_positions,
        &[
            ("date", DataType::Datetime(TimeUnit::Milliseconds, None)),
            ("asset", DataType::String),
            ("position", DataType::Float64),
        ],
    );
    assert_upstream_position_projection(
        &raw_positions,
        load_records(include_str!(
            "../../../tests/golden/phase-03/positions.json"
        )),
    );
    assert_exact_position_contract(
        &raw_positions,
        load_records(include_str!(
            "../../../tests/golden/phase-03/ferric_positions_contract.json"
        )),
    );

    let simulated = factor_positions(
        &factor_data,
        Period::new(1).unwrap(),
        PortfolioOptions::default(),
        Some(&sessions),
    )
    .unwrap();
    assert_schema(
        &simulated,
        &[
            ("date", DataType::Datetime(TimeUnit::Milliseconds, None)),
            ("asset", DataType::String),
            ("position", DataType::Float64),
        ],
    );
    assert_upstream_position_projection(
        &simulated,
        load_records(include_str!(
            "../../../tests/golden/phase-03/factor_positions.json"
        )),
    );

    let defects: Defects = serde_json::from_str(include_str!(
        "../../../tests/golden/phase-03/upstream_defects.json"
    ))
    .unwrap();
    for defect in [
        defects.factor_cumulative_returns,
        defects.create_pyfolio_input,
    ] {
        assert_eq!(defect.exception, "TypeError");
        assert!(defect.message.contains("cumulative_returns"));
    }
}

fn load_input() -> Input {
    serde_json::from_str(include_str!("../../../tests/golden/phase-03/input.json")).unwrap()
}

fn load_records(json: &str) -> Vec<GoldenRecord> {
    serde_json::from_str(json).unwrap()
}

fn datetime_column(values: Vec<i64>) -> Column {
    Series::new("date".into(), values)
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .unwrap()
        .into()
}

fn factor_frame(input: &Input) -> DataFrame {
    DataFrame::new(
        input.rows.len(),
        vec![
            datetime_column(input.rows.iter().map(|row| date_ms(&row.date)).collect()),
            Series::new(
                "asset".into(),
                input
                    .rows
                    .iter()
                    .map(|row| row.asset.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "group".into(),
                input
                    .rows
                    .iter()
                    .map(|row| row.group.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor".into(),
                input.rows.iter().map(|row| row.factor).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                input
                    .rows
                    .iter()
                    .map(|row| row.factor_quantile)
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                input.rows.iter().map(|row| row.one_day).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_3D".into(),
                input
                    .rows
                    .iter()
                    .map(|row| row.three_day)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn return_frame(input: &Input) -> DataFrame {
    DataFrame::new(
        input.simple_returns.len(),
        vec![
            datetime_column(
                input
                    .simple_returns
                    .iter()
                    .map(|row| date_ms(&row.date))
                    .collect(),
            ),
            Series::new(
                "return".into(),
                input
                    .simple_returns
                    .iter()
                    .map(|row| row.value)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn weight_frame(input: &Input) -> DataFrame {
    DataFrame::new(
        input.weights.len(),
        vec![
            datetime_column(input.weights.iter().map(|row| date_ms(&row.date)).collect()),
            Series::new(
                "asset".into(),
                input
                    .weights
                    .iter()
                    .map(|row| row.asset.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "weight".into(),
                input
                    .weights
                    .iter()
                    .map(|row| row.weight)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn session_frame(input: &Input) -> DataFrame {
    DataFrame::new(
        input.session_dates.len(),
        vec![datetime_column(
            input
                .session_dates
                .iter()
                .map(|date| date_ms(date))
                .collect(),
        )],
    )
    .unwrap()
}

fn assert_series_golden(
    actual: &DataFrame,
    value_name: &str,
    expected: Vec<GoldenRecord>,
    expected_value: impl Fn(&GoldenRecord) -> Option<f64>,
) {
    let dates = actual.column("date").unwrap().datetime().unwrap();
    let values = actual.column(value_name).unwrap().f64().unwrap();
    assert_eq!(actual.height(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(
            date_string(dates.physical().get(index).unwrap()),
            expected.date
        );
        assert_optional_close(values.get(index), expected_value(expected));
    }
}

fn assert_schema(actual: &DataFrame, expected: &[(&str, DataType)]) {
    assert_eq!(actual.width(), expected.len());
    for (column, (expected_name, expected_dtype)) in actual.columns().iter().zip(expected) {
        assert_eq!(column.name().as_str(), *expected_name);
        assert_eq!(column.dtype(), expected_dtype);
    }
}

// Alphalens omits synthetic cash and derives a different BDay calendar. This
// projection compares only the common upstream keys; the full Ferric contract
// is asserted separately below.
fn assert_upstream_position_projection(actual: &DataFrame, expected: Vec<GoldenRecord>) {
    let expected_keys = expected
        .iter()
        .map(|record| (record.date.clone(), record.asset.clone().unwrap()))
        .collect::<BTreeSet<_>>();
    let dates = actual.column("date").unwrap().datetime().unwrap();
    let assets = actual.column("asset").unwrap().str().unwrap();
    let values = actual.column("position").unwrap().f64().unwrap();
    let filtered = (0..actual.height())
        .filter_map(|index| {
            let key = (
                date_string(dates.physical().get(index).unwrap()),
                assets.get(index).unwrap().to_owned(),
            );
            expected_keys
                .contains(&key)
                .then_some((key, values.get(index)))
        })
        .collect::<Vec<_>>();
    assert_eq!(filtered.len(), expected.len());
    for ((actual_key, actual_value), expected) in filtered.into_iter().zip(expected) {
        assert_eq!(actual_key, (expected.date, expected.asset.unwrap()));
        assert_optional_close(actual_value, expected.position);
    }
}

fn assert_exact_position_contract(actual: &DataFrame, expected: Vec<GoldenRecord>) {
    let dates = actual.column("date").unwrap().datetime().unwrap();
    let assets = actual.column("asset").unwrap().str().unwrap();
    let values = actual.column("position").unwrap().f64().unwrap();
    assert_eq!(actual.height(), expected.len());
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            date_string(dates.physical().get(index).unwrap()),
            expected.date
        );
        assert_eq!(assets.get(index), expected.asset.as_deref());
        assert_optional_close(values.get(index), expected.position);
    }
}

fn assert_optional_close(actual: Option<f64>, expected: Option<f64>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            assert_relative_eq!(actual, expected, epsilon = EPSILON, max_relative = EPSILON)
        }
        (None, None) => {}
        mismatch => panic!("numeric mismatch: {mismatch:?}"),
    }
}

fn date_ms(value: &str) -> i64 {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_millis()
}

fn date_string(value: i64) -> String {
    chrono::DateTime::from_timestamp_millis(value)
        .unwrap()
        .date_naive()
        .to_string()
}
