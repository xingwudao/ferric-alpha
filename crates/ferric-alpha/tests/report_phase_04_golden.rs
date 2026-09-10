use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use approx::assert_abs_diff_eq;
use chrono::DateTime;
use ferric_alpha::{
    EventReturnsOptions, EventStudyTearSheetOptions, FullTearSheetOptions,
    InformationTearSheetOptions, Period, ReturnsTearSheetOptions, TearSheetData,
    TurnoverTearSheetOptions, average_cumulative_return_by_quantile,
    create_event_returns_tear_sheet_data, create_event_study_tear_sheet_data,
    create_full_tear_sheet_data, create_information_tear_sheet_data,
    create_returns_tear_sheet_data, create_turnover_tear_sheet_data, factor_quantile_statistics,
};
use polars::prelude::{DataFrame, DataType, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};
use serde::Deserialize;
use serde_json::{Map, Value, json};

const EPSILON: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct Input {
    timezone: String,
    rows: Vec<FactorRow>,
    constant_ic_rows: Vec<FactorRow>,
    event_returns: Vec<EventReturnRow>,
}

#[derive(Debug, Deserialize)]
struct FactorRow {
    date: String,
    asset: String,
    group: String,
    factor: Option<f64>,
    factor_quantile: u32,
    #[serde(rename = "forward_return_5D")]
    forward_return_5d: Option<f64>,
    #[serde(rename = "forward_return_1D")]
    forward_return_1d: Option<f64>,
    #[serde(rename = "forward_return_3D")]
    forward_return_3d: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct EventReturnRow {
    date: String,
    asset: String,
    #[serde(rename = "return")]
    value: ReturnValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ReturnValue {
    Number(f64),
    Text(String),
    Null,
}

#[test]
fn phase_04_matches_ferric_numeric_baseline() {
    let input = load_input();
    assert_synthetic_input_shape(&input);

    let factor_data = factor_frame(&input.rows, &input.timezone);
    let constant_ic_data = factor_frame(&input.constant_ic_rows, &input.timezone);
    let event_returns = event_returns_frame(&input.event_returns, &input.timezone);

    let quantile_statistics = factor_quantile_statistics(&factor_data).unwrap();
    assert_table_matches_records(
        &quantile_statistics.to_polars().unwrap(),
        &load_json("quantile_statistics.json")["ferric_projection"],
        &["factor_quantile"],
        &[
            "factor_min",
            "factor_max",
            "factor_mean",
            "factor_std",
            "count_percent",
        ],
    );

    let returns = create_returns_tear_sheet_data(
        &factor_data,
        ReturnsTearSheetOptions {
            by_group: true,
            ..ReturnsTearSheetOptions::default()
        },
    )
    .unwrap();
    let returns_capture = load_json("returns_capture.json");
    assert_table_matches_records(
        &returns.data.mean_by_quantile.to_polars().unwrap(),
        &returns_capture["mean_by_quantile"],
        &["factor_quantile", "period"],
        &["mean_return", "std_error"],
    );
    assert_table_matches_records(
        &returns.data.factor_returns.to_polars().unwrap(),
        &returns_capture["factor_returns"],
        &["date", "period"],
        &["factor_return"],
    );

    let information = create_information_tear_sheet_data(
        &constant_ic_data,
        InformationTearSheetOptions {
            by_group: true,
            rolling_window: 3,
            ..InformationTearSheetOptions::default()
        },
    )
    .unwrap();
    let information_capture = load_json("information_capture.json");
    assert_table_matches_records(
        &information.data.ic_by_date.to_polars().unwrap(),
        &information_capture["constant_ic_by_date"],
        &["date", "period"],
        &["ic"],
    );
    assert_table_matches_records(
        &information.data.summary.to_polars().unwrap(),
        &information_capture["constant_summary"],
        &["period"],
        &["ic_mean"],
    );

    let turnover = create_turnover_tear_sheet_data(
        &factor_data,
        TurnoverTearSheetOptions {
            periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
        },
    )
    .unwrap();
    let turnover_capture = load_json("turnover_capture.json");
    assert_table_matches_records(
        &turnover.data.quantile_series.to_polars().unwrap(),
        &turnover_capture["quantile_turnover"],
        &["date", "factor_quantile", "period"],
        &["turnover"],
    );
    assert_table_matches_records(
        &turnover
            .data
            .rank_autocorrelation_series
            .to_polars()
            .unwrap(),
        &turnover_capture["rank_autocorrelation"],
        &["date", "period"],
        &["autocorrelation"],
    );

    let event_capture = load_json("event_returns.json");
    let event_options = EventReturnsOptions {
        periods_before: 2,
        periods_after: 2,
        demeaned: false,
        group_adjust: false,
        by_group: false,
    };
    let event_projection =
        average_cumulative_return_by_quantile(&factor_data, &event_returns, event_options).unwrap();
    assert_table_matches_records(
        &event_projection,
        &event_capture["ferric_projection"],
        &["factor_quantile", "offset"],
        &["mean_cumulative_return", "std_cumulative_return"],
    );
}

#[test]
fn phase_04_ferric_contract_matches_complete_fixture() {
    let input = load_input();
    let actual = build_contract(&input);
    let expected = load_json("ferric_contract.json");
    assert_eq!(actual, expected);

    for report in actual["reports"].as_array().unwrap() {
        let decoded = TearSheetData::from_json(report["canonical_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            decoded.to_json().unwrap(),
            report["canonical_json"].as_str().unwrap()
        );
    }
}

#[test]
#[ignore = "exports the Ferric Phase 04 contract for local fixture maintenance"]
fn export_phase_04_ferric_contract() {
    let Some(output) = env::var_os("FERRIC_ALPHA_PHASE04_CONTRACT_OUT") else {
        return;
    };
    let input = load_input();
    let content = serde_json::to_string_pretty(&build_contract(&input)).unwrap() + "\n";
    fs::write(output, content).unwrap();
}

fn assert_synthetic_input_shape(input: &Input) {
    let sessions = input
        .rows
        .iter()
        .map(|row| &row.date)
        .collect::<BTreeSet<_>>();
    let assets = input
        .rows
        .iter()
        .map(|row| &row.asset)
        .collect::<BTreeSet<_>>();
    let groups = input
        .rows
        .iter()
        .map(|row| &row.group)
        .collect::<BTreeSet<_>>();
    let quantiles = input
        .rows
        .iter()
        .map(|row| row.factor_quantile)
        .collect::<BTreeSet<_>>();
    assert!(sessions.len() >= 8);
    assert_eq!(assets.len(), 5);
    assert_eq!(groups.len(), 2);
    assert_eq!(quantiles, BTreeSet::from([1, 2, 3]));
    assert!(input.rows.iter().any(|row| row.factor.is_none()));
    assert!(input.rows.iter().any(|left| {
        input.rows.iter().any(|right| {
            left.date == right.date
                && left.asset != right.asset
                && left.factor.is_some()
                && left.factor == right.factor
        })
    }));
    assert!(
        sessions
            .iter()
            .any(|date| input.rows.iter().filter(|row| &row.date == *date).count() < 5)
    );
    assert!(
        input
            .event_returns
            .iter()
            .any(|row| matches!(row.value, ReturnValue::Null))
    );
    assert!(input.event_returns.iter().any(
        |row| matches!(&row.value, ReturnValue::Text(text) if text == "NaN" || text == "Infinity")
    ));
}

fn build_contract(input: &Input) -> Value {
    let factor_data = factor_frame(&input.rows, &input.timezone);
    let event_returns = event_returns_frame(&input.event_returns, &input.timezone);
    let reports = vec![
        TearSheetData::from(
            create_full_tear_sheet_data(
                &factor_data,
                FullTearSheetOptions {
                    by_group: true,
                    turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                    ic_rolling_window: 3,
                    ..FullTearSheetOptions::default()
                },
            )
            .unwrap(),
        ),
        TearSheetData::from(
            create_event_returns_tear_sheet_data(
                &factor_data,
                &event_returns,
                EventReturnsOptions {
                    periods_before: 2,
                    periods_after: 2,
                    demeaned: false,
                    group_adjust: false,
                    by_group: false,
                },
            )
            .unwrap(),
        ),
        TearSheetData::from(
            create_event_study_tear_sheet_data(
                &factor_data,
                Some(&event_returns),
                EventStudyTearSheetOptions {
                    event_window: Some((2, 2)),
                    rate_of_return: true,
                    histogram_bins: 4,
                },
            )
            .unwrap(),
        ),
    ];

    json!({
        "contract_version": "ferric-alpha.tear-sheet/v1",
        "input": {
            "timezone": input.timezone,
            "factor_rows": input.rows.len(),
            "constant_ic_rows": input.constant_ic_rows.len(),
            "event_return_rows": input.event_returns.len(),
        },
        "reports": reports.into_iter().map(report_contract).collect::<Vec<_>>(),
    })
}

fn report_contract(report: TearSheetData) -> Value {
    let canonical_json = report.to_json().unwrap();
    let body: Value = serde_json::from_str(&canonical_json).unwrap();
    let table_ids = report
        .table_ids()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let tables = table_ids
        .iter()
        .map(|id| table_contract(report.table(id).unwrap()))
        .collect::<Vec<_>>();

    json!({
        "kind": body["metadata"]["report_kind"],
        "metadata": body["metadata"],
        "options": body["options"],
        "table_ids": table_ids,
        "tables": tables,
        "canonical_json": canonical_json,
    })
}

fn table_contract(table: &ferric_alpha::ReportTable) -> Value {
    let table_json = serde_json::to_value(table).unwrap();
    let schema = table
        .columns()
        .iter()
        .map(|column| {
            json!({
                "name": column.name(),
                "label": column.label(),
                "role": table_json["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|value| value["name"] == column.name())
                    .unwrap()["role"],
                "dtype": column_data_type(column.data()),
                "display": table_json["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|value| value["name"] == column.name())
                    .unwrap()["display"],
            })
        })
        .collect::<Vec<_>>();
    let null_counts = table
        .columns()
        .iter()
        .map(|column| (column.name().to_owned(), json!(null_count(column.data()))))
        .collect::<Map<_, _>>();
    json!({
        "id": table.id(),
        "title": table.title(),
        "row_count": table.row_count(),
        "sort_by": table.sort_by(),
        "schema": schema,
        "null_counts": null_counts,
        "json": table_json,
    })
}

fn assert_table_matches_records(
    frame: &DataFrame,
    expected: &Value,
    keys: &[&str],
    measures: &[&str],
) {
    let actual = frame_records(frame, keys, measures);
    let expected = expected_records(expected, keys, measures);
    assert_eq!(actual.len(), expected.len(), "record count mismatch");
    for (key, expected_values) in expected {
        let actual_values = actual.get(&key).unwrap_or_else(|| {
            panic!("missing actual record for key {key}");
        });
        for measure in measures {
            let left = actual_values.get(*measure).unwrap();
            let right = expected_values.get(*measure).unwrap();
            assert_option_float(left.as_f64(), right.as_f64(), &format!("{key}:{measure}"));
        }
    }
}

fn frame_records(
    frame: &DataFrame,
    keys: &[&str],
    measures: &[&str],
) -> BTreeMap<String, Map<String, Value>> {
    let mut out = BTreeMap::new();
    for row in 0..frame.height() {
        let mut values = Map::new();
        let key = keys
            .iter()
            .map(|name| scalar_key(frame, name, row))
            .collect::<Vec<_>>()
            .join("|");
        for measure in measures {
            values.insert((*measure).to_string(), scalar_value(frame, measure, row));
        }
        out.insert(key, values);
    }
    out
}

fn expected_records(
    expected: &Value,
    keys: &[&str],
    measures: &[&str],
) -> BTreeMap<String, Map<String, Value>> {
    expected
        .as_array()
        .unwrap()
        .iter()
        .map(|record| {
            let key = keys
                .iter()
                .map(|name| record[*name].to_string().trim_matches('"').to_string())
                .collect::<Vec<_>>()
                .join("|");
            let mut values = Map::new();
            for measure in measures {
                values.insert((*measure).to_string(), record[*measure].clone());
            }
            (key, values)
        })
        .collect()
}

fn assert_option_float(actual: Option<f64>, expected: Option<f64>, label: &str) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            assert_abs_diff_eq!(actual, expected, epsilon = EPSILON);
        }
        (None, None) => {}
        other => panic!("null mismatch at {label}: {other:?}"),
    }
}

fn scalar_key(frame: &DataFrame, name: &str, row: usize) -> String {
    let column = frame.column(name).unwrap();
    match column.dtype() {
        DataType::Datetime(_, _) => ms_to_date(
            column
                .as_materialized_series()
                .datetime()
                .unwrap()
                .physical()
                .get(row)
                .unwrap(),
        ),
        DataType::String => column
            .as_materialized_series()
            .str()
            .unwrap()
            .get(row)
            .unwrap()
            .to_owned(),
        DataType::UInt32 => column
            .as_materialized_series()
            .u32()
            .unwrap()
            .get(row)
            .unwrap()
            .to_string(),
        DataType::Int32 => column
            .as_materialized_series()
            .i32()
            .unwrap()
            .get(row)
            .unwrap()
            .to_string(),
        dtype => panic!("unsupported key dtype for {name}: {dtype:?}"),
    }
}

fn scalar_value(frame: &DataFrame, name: &str, row: usize) -> Value {
    let column = frame.column(name).unwrap();
    match column.dtype() {
        DataType::Float64 => match column.as_materialized_series().f64().unwrap().get(row) {
            Some(value) => json!(value),
            None => Value::Null,
        },
        DataType::UInt64 => match column.as_materialized_series().u64().unwrap().get(row) {
            Some(value) => json!(value),
            None => Value::Null,
        },
        dtype => panic!("unsupported measure dtype for {name}: {dtype:?}"),
    }
}

fn load_input() -> Input {
    serde_json::from_str(&fs::read_to_string(golden_path("input.json")).unwrap()).unwrap()
}

fn load_json(filename: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(golden_path(filename)).unwrap()).unwrap()
}

fn golden_path(filename: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/phase-04")
        .join(filename)
}

fn factor_frame(rows: &[FactorRow], timezone: &str) -> DataFrame {
    let dates = rows
        .iter()
        .map(|row| timestamp_ms(&row.date))
        .collect::<Vec<_>>();
    let timezone = TimeZone::opt_try_new(Some(timezone)).unwrap();
    DataFrame::new(
        rows.len(),
        vec![
            Series::new("date".into(), dates)
                .into_datetime(TimeUnit::Milliseconds, timezone)
                .into_column(),
            Series::new(
                "asset".into(),
                rows.iter()
                    .map(|row| row.asset.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "group".into(),
                rows.iter()
                    .map(|row| row.group.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor".into(),
                rows.iter().map(|row| row.factor).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                rows.iter()
                    .map(|row| Some(row.factor_quantile))
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_5D".into(),
                rows.iter()
                    .map(|row| row.forward_return_5d)
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                rows.iter()
                    .map(|row| row.forward_return_1d)
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_3D".into(),
                rows.iter()
                    .map(|row| row.forward_return_3d)
                    .collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn event_returns_frame(rows: &[EventReturnRow], timezone: &str) -> DataFrame {
    let timezone = TimeZone::opt_try_new(Some(timezone)).unwrap();
    DataFrame::new(
        rows.len(),
        vec![
            Series::new(
                "date".into(),
                rows.iter()
                    .map(|row| timestamp_ms(&row.date))
                    .collect::<Vec<_>>(),
            )
            .into_datetime(TimeUnit::Milliseconds, timezone)
            .into_column(),
            Series::new(
                "asset".into(),
                rows.iter()
                    .map(|row| row.asset.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "return".into(),
                rows.iter()
                    .map(|row| match &row.value {
                        ReturnValue::Number(value) => Some(*value),
                        ReturnValue::Text(text) if text == "NaN" => Some(f64::NAN),
                        ReturnValue::Text(text) if text == "Infinity" => Some(f64::INFINITY),
                        ReturnValue::Text(text) if text == "-Infinity" => Some(f64::NEG_INFINITY),
                        ReturnValue::Text(_) | ReturnValue::Null => None,
                    })
                    .collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn timestamp_ms(value: &str) -> i64 {
    DateTime::parse_from_rfc3339(value)
        .unwrap()
        .timestamp_millis()
}

fn ms_to_date(value: i64) -> String {
    DateTime::from_timestamp_millis(value)
        .unwrap()
        .date_naive()
        .to_string()
}

fn column_data_type(data: &ferric_alpha::ReportColumnData) -> &'static str {
    match data {
        ferric_alpha::ReportColumnData::Boolean(_) => "boolean",
        ferric_alpha::ReportColumnData::Int32(_) => "int32",
        ferric_alpha::ReportColumnData::Int64(_) => "int64",
        ferric_alpha::ReportColumnData::UInt32(_) => "uint32",
        ferric_alpha::ReportColumnData::UInt64(_) => "uint64",
        ferric_alpha::ReportColumnData::Float64(_) => "float64",
        ferric_alpha::ReportColumnData::String(_) => "string",
        ferric_alpha::ReportColumnData::Datetime { .. } => "datetime",
    }
}

fn null_count(data: &ferric_alpha::ReportColumnData) -> usize {
    match data {
        ferric_alpha::ReportColumnData::Boolean(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::Int32(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::Int64(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::UInt32(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::UInt64(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::Float64(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::String(values) => {
            values.iter().filter(|value| value.is_none()).count()
        }
        ferric_alpha::ReportColumnData::Datetime { values, .. } => {
            values.iter().filter(|value| value.is_none()).count()
        }
    }
}
