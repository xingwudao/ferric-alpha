use std::collections::BTreeMap;

use approx::assert_relative_eq;
use chrono::NaiveDate;
use ferric_alpha::{
    AlphaBetaOptions, IcOptions, MeanIcOptions, MeanReturnByQuantileOptions, ReturnOptions,
    WeightOptions, compute_mean_returns_spread, factor_alpha_beta, factor_information_coefficient,
    factor_returns, factor_weights, mean_information_coefficient, mean_return_by_quantile,
};
use polars::prelude::*;
use serde::Deserialize;

const ABS: f64 = 1e-12;
const REL: f64 = 1e-10;

#[derive(Debug, Deserialize)]
struct Input {
    rows: Vec<InputRow>,
}

#[derive(Debug, Deserialize)]
struct InputRow {
    date: String,
    asset: String,
    group: String,
    factor: f64,
    factor_quantile: u32,
    #[serde(rename = "forward_return_1D")]
    forward_return_1d: Option<f64>,
    #[serde(rename = "forward_return_5D")]
    forward_return_5d: Option<f64>,
    #[serde(rename = "forward_return_10D")]
    forward_return_10d: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ValueRecord {
    date: Option<String>,
    asset: Option<String>,
    period: Option<String>,
    metric: Option<String>,
    factor_quantile: Option<u32>,
    weight: Option<f64>,
    ic: Option<f64>,
    mean_ic: Option<f64>,
    mean_return: Option<f64>,
    factor_return: Option<f64>,
    mean_return_difference: Option<f64>,
    std_error: Option<f64>,
    value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct QuantileGolden {
    mean_return: Vec<ValueRecord>,
    std_error: Vec<ValueRecord>,
}

#[test]
fn phase_02_matches_alphalens_golden_outputs() {
    let input: Input =
        serde_json::from_str(include_str!("../../../tests/golden/phase-02/input.json")).unwrap();
    let frame = input_frame(&input);

    let ic = factor_information_coefficient(&frame, IcOptions::default()).unwrap();
    assert_schema(&ic, &["date", "period", "ic"]);
    assert_ordered_records(
        ordered_records(&ic, "ic"),
        load_ordered_records(include_str!("../../../tests/golden/phase-02/ic.json"), "ic"),
    );
    let mean_ic = mean_information_coefficient(&frame, MeanIcOptions::default()).unwrap();
    assert_schema(&mean_ic, &["period", "mean_ic", "count"]);
    assert_ordered_records(
        ordered_mean_ic_records(&mean_ic),
        load_ordered_records(
            include_str!("../../../tests/golden/phase-02/mean_ic.json"),
            "mean_ic",
        ),
    );
    let weights = factor_weights(&frame, WeightOptions::default()).unwrap();
    assert_schema(&weights, &["date", "asset", "group", "weight"]);
    assert_ordered_records(
        ordered_weight_records(&weights),
        load_ordered_records(
            include_str!("../../../tests/golden/phase-02/weights.json"),
            "weight",
        ),
    );
    let returns = factor_returns(&frame, ReturnOptions::default()).unwrap();
    assert_schema(&returns, &["date", "period", "factor_return"]);
    assert_ordered_records(
        ordered_records(&returns, "factor_return"),
        load_ordered_records(
            include_str!("../../../tests/golden/phase-02/factor_returns.json"),
            "factor_return",
        ),
    );

    let mean_returns = mean_return_by_quantile(
        &frame,
        MeanReturnByQuantileOptions::default().with_by_date(true),
    )
    .unwrap();
    assert_schema(
        &mean_returns,
        &[
            "factor_quantile",
            "date",
            "period",
            "mean_return",
            "std_error",
            "count",
        ],
    );
    let quantile_golden: QuantileGolden = serde_json::from_str(include_str!(
        "../../../tests/golden/phase-02/mean_return_by_quantile.json"
    ))
    .unwrap();
    assert_ordered_records(
        ordered_quantile_records(&mean_returns, "mean_return"),
        ordered_quantile_golden(quantile_golden.mean_return, "mean_return"),
    );
    assert_ordered_records(
        ordered_quantile_records(&mean_returns, "std_error"),
        ordered_quantile_golden(quantile_golden.std_error, "std_error"),
    );
    let spread = compute_mean_returns_spread(&mean_returns, 5, 1, None).unwrap();
    assert_schema(
        &spread,
        &[
            "date",
            "period",
            "mean_return_difference",
            "joint_std_error",
        ],
    );
    assert_ordered_records(
        ordered_records(&spread, "mean_return_difference"),
        load_ordered_records(
            include_str!("../../../tests/golden/phase-02/spread.json"),
            "mean_return_difference",
        ),
    );

    let alpha_beta = factor_alpha_beta(&frame, None, AlphaBetaOptions::default()).unwrap();
    assert_alpha_beta(&alpha_beta);
}

fn input_frame(input: &Input) -> DataFrame {
    let dates = input
        .rows
        .iter()
        .map(|row| date_to_ms(&row.date))
        .collect::<Vec<_>>();
    DataFrame::new(
        input.rows.len(),
        vec![
            Series::new("date".into(), dates)
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
                .into(),
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
                input
                    .rows
                    .iter()
                    .map(|row| row.forward_return_1d)
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_5D".into(),
                input
                    .rows
                    .iter()
                    .map(|row| row.forward_return_5d)
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_10D".into(),
                input
                    .rows
                    .iter()
                    .map(|row| row.forward_return_10d)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn date_to_ms(value: &str) -> i64 {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_millis()
}

fn ms_to_date(value: i64) -> String {
    chrono::DateTime::from_timestamp_millis(value)
        .unwrap()
        .date_naive()
        .to_string()
}

fn load_ordered_records(json: &str, value_name: &str) -> Vec<(String, Option<f64>)> {
    serde_json::from_str::<Vec<ValueRecord>>(json)
        .unwrap()
        .into_iter()
        .map(|record| (key(&record), value(&record, value_name)))
        .collect()
}

fn key(record: &ValueRecord) -> String {
    [
        record.date.clone(),
        record.asset.clone(),
        record.factor_quantile.map(|value| value.to_string()),
        record.period.clone(),
        record.metric.clone(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("|")
}

fn value(record: &ValueRecord, value_name: &str) -> Option<f64> {
    match value_name {
        "ic" => record.ic,
        "mean_ic" => record.mean_ic,
        "mean_return" => record.mean_return,
        "std_error" => record.std_error,
        "weight" => record.weight,
        "factor_return" => record.factor_return,
        "mean_return_difference" => record.mean_return_difference.or(record.value),
        _ => unreachable!(),
    }
}

fn ordered_records(frame: &DataFrame, value_name: &str) -> Vec<(String, Option<f64>)> {
    let dates = frame.column("date").unwrap().datetime().unwrap();
    let periods = frame.column("period").unwrap().str().unwrap();
    let values = frame.column(value_name).unwrap().f64().unwrap();
    (0..frame.height())
        .map(|index| {
            (
                format!(
                    "{}|{}",
                    ms_to_date(dates.physical().get(index).unwrap()),
                    periods.get(index).unwrap()
                ),
                values.get(index),
            )
        })
        .collect()
}

fn ordered_mean_ic_records(frame: &DataFrame) -> Vec<(String, Option<f64>)> {
    let periods = frame.column("period").unwrap().str().unwrap();
    let values = frame.column("mean_ic").unwrap().f64().unwrap();
    (0..frame.height())
        .map(|index| (periods.get(index).unwrap().to_owned(), values.get(index)))
        .collect()
}

fn ordered_weight_records(frame: &DataFrame) -> Vec<(String, Option<f64>)> {
    let dates = frame.column("date").unwrap().datetime().unwrap();
    let assets = frame.column("asset").unwrap().str().unwrap();
    let values = frame.column("weight").unwrap().f64().unwrap();
    (0..frame.height())
        .map(|index| {
            (
                format!(
                    "{}|{}",
                    ms_to_date(dates.physical().get(index).unwrap()),
                    assets.get(index).unwrap()
                ),
                values.get(index),
            )
        })
        .collect()
}

fn ordered_quantile_records(frame: &DataFrame, value_name: &str) -> Vec<(String, Option<f64>)> {
    let quantiles = frame.column("factor_quantile").unwrap().u32().unwrap();
    let dates = frame.column("date").unwrap().datetime().unwrap();
    let periods = frame.column("period").unwrap().str().unwrap();
    let values = frame.column(value_name).unwrap().f64().unwrap();
    (0..frame.height())
        .map(|index| {
            (
                format!(
                    "{}|{}|{}",
                    ms_to_date(dates.physical().get(index).unwrap()),
                    quantiles.get(index).unwrap(),
                    periods.get(index).unwrap()
                ),
                values.get(index),
            )
        })
        .collect()
}

fn ordered_quantile_golden(
    records: Vec<ValueRecord>,
    value_name: &str,
) -> Vec<(String, Option<f64>)> {
    records
        .into_iter()
        .map(|record| (key(&record), value(&record, value_name)))
        .collect()
}

fn assert_ordered_records(
    actual: Vec<(String, Option<f64>)>,
    expected: Vec<(String, Option<f64>)>,
) {
    assert_eq!(
        actual.iter().map(|record| &record.0).collect::<Vec<_>>(),
        expected.iter().map(|record| &record.0).collect::<Vec<_>>()
    );
    for ((key, actual), (_, expected)) in actual.into_iter().zip(expected) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                assert_relative_eq!(actual, expected, epsilon = ABS, max_relative = REL)
            }
            (None, None) => {}
            mismatch => panic!("{key}: value mismatch {mismatch:?}"),
        }
    }
}

fn assert_schema(frame: &DataFrame, expected_names: &[&str]) {
    assert_eq!(
        frame
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>(),
        expected_names
    );
    for name in expected_names {
        match *name {
            "date" => assert!(matches!(
                frame.column(name).unwrap().dtype(),
                DataType::Datetime(_, _)
            )),
            "period" | "asset" | "group" => {
                assert_eq!(frame.column(name).unwrap().dtype(), &DataType::String)
            }
            "factor_quantile" | "count" => {
                assert_eq!(frame.column(name).unwrap().dtype(), &DataType::UInt32)
            }
            _ => assert_eq!(frame.column(name).unwrap().dtype(), &DataType::Float64),
        }
    }
}

fn assert_alpha_beta(frame: &DataFrame) {
    let golden = load_alpha_beta();
    let periods = frame.column("period").unwrap().str().unwrap();
    let annual = frame.column("annualized_alpha").unwrap().f64().unwrap();
    let beta = frame.column("beta").unwrap().f64().unwrap();
    for index in 0..frame.height() {
        let period = periods.get(index).unwrap();
        for (metric, value) in [
            ("annualized_alpha", annual.get(index)),
            ("beta", beta.get(index)),
        ] {
            let expected = golden[&format!("{period}|{metric}")];
            match (value, expected) {
                (Some(actual), Some(expected)) => {
                    assert_relative_eq!(actual, expected, epsilon = ABS, max_relative = REL)
                }
                (None, None) => {}
                mismatch => panic!("{period}|{metric}: value mismatch {mismatch:?}"),
            }
        }
    }
}

fn load_alpha_beta() -> BTreeMap<String, Option<f64>> {
    serde_json::from_str::<Vec<ValueRecord>>(include_str!(
        "../../../tests/golden/phase-02/alpha_beta.json"
    ))
    .unwrap()
    .into_iter()
    .map(|record| {
        (
            format!(
                "{}|{}",
                record.period.expect("period"),
                record.metric.expect("metric")
            ),
            record.value,
        )
    })
    .collect()
}
