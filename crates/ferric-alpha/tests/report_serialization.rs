use ferric_alpha::{
    EventReturnsOptions, EventStudyTearSheetOptions, FerricAlphaError, FullTearSheetOptions,
    InformationTearSheetOptions, Period, ReportKind, ReturnsTearSheetOptions,
    SummaryTearSheetOptions, TearSheetData, TurnoverTearSheetOptions,
    create_event_returns_tear_sheet_data, create_event_study_tear_sheet_data,
    create_full_tear_sheet_data, create_information_tear_sheet_data,
    create_returns_tear_sheet_data, create_summary_tear_sheet_data,
    create_turnover_tear_sheet_data,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};
use serde_json::Value;

const JAN_1: i64 = 1_704_067_200_000;
const DAY_MS: i64 = 86_400_000;

fn factor_data(reversed: bool) -> DataFrame {
    let mut rows = Vec::new();
    for day in 0..24_i64 {
        let date = JAN_1 + day * DAY_MS;
        let rotation = day.rem_euclid(4) as usize;
        for asset_index in 0..4 {
            rows.push((
                date,
                ["a", "b", "c", "d"][asset_index],
                if asset_index < 2 { "g1" } else { "g2" },
                Some((asset_index + 1) as f64),
                Some((asset_index + 1) as u32),
                Some((rotation + asset_index + 1) as f64 * 0.01),
                Some((rotation + asset_index + 1) as f64 * 0.03),
            ));
        }
    }
    if reversed {
        rows.reverse();
    }

    DataFrame::new(
        rows.len(),
        vec![
            Series::new(
                "date".into(),
                rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            )
            .into_datetime(
                TimeUnit::Milliseconds,
                TimeZone::opt_try_new(Some("UTC")).unwrap(),
            )
            .into_column(),
            Series::new(
                "asset".into(),
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "group".into(),
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor".into(),
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                rows.iter().map(|row| row.5).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_3D".into(),
                rows.iter().map(|row| row.6).collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap()
}

fn event_factor_data() -> DataFrame {
    let jan_2 = JAN_1 + DAY_MS;
    let absent = JAN_1 + 3 * DAY_MS;
    DataFrame::new(
        5,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1, jan_2, jan_2, absent])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["A", "B", "A", "B", "A"]).into(),
            Series::new("group".into(), &["g1", "g1", "g1", "g1", "g1"]).into(),
            Series::new(
                "factor".into(),
                &[Some(-1.0), Some(1.0), Some(-0.5), Some(0.5), Some(2.0)],
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                &[Some(1_u32), Some(2), Some(1), Some(2), Some(2)],
            )
            .into(),
            Series::new(
                "forward_return_1D".into(),
                &[Some(-0.01), Some(0.02), Some(-0.02), Some(0.03), Some(0.04)],
            )
            .into(),
        ],
    )
    .unwrap()
}

fn event_returns_data() -> DataFrame {
    let jan_2 = JAN_1 + DAY_MS;
    let jan_3 = JAN_1 + 2 * DAY_MS;
    DataFrame::new(
        5,
        vec![
            Series::new("date".into(), &[JAN_1, JAN_1, jan_2, jan_2, jan_3])
                .into_datetime(
                    TimeUnit::Milliseconds,
                    TimeZone::opt_try_new(Some("UTC")).unwrap(),
                )
                .into_column(),
            Series::new("asset".into(), &["A", "B", "A", "B", "A"]).into(),
            Series::new(
                "return".into(),
                &[Some(0.10), Some(0.20), None, Some(0.00), Some(0.10)],
            )
            .into(),
        ],
    )
    .unwrap()
}

fn reports() -> Vec<TearSheetData> {
    let data = factor_data(false);
    let event_factor = event_factor_data();
    let event_returns = event_returns_data();
    vec![
        TearSheetData::from(
            create_summary_tear_sheet_data(&data, SummaryTearSheetOptions::default()).unwrap(),
        ),
        TearSheetData::from(
            create_returns_tear_sheet_data(
                &data,
                ReturnsTearSheetOptions {
                    by_group: true,
                    ..ReturnsTearSheetOptions::default()
                },
            )
            .unwrap(),
        ),
        TearSheetData::from(
            create_information_tear_sheet_data(
                &data,
                InformationTearSheetOptions {
                    by_group: true,
                    rolling_window: 3,
                    ..InformationTearSheetOptions::default()
                },
            )
            .unwrap(),
        ),
        TearSheetData::from(
            create_turnover_tear_sheet_data(
                &data,
                TurnoverTearSheetOptions {
                    periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                },
            )
            .unwrap(),
        ),
        TearSheetData::from(
            create_full_tear_sheet_data(
                &data,
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
                &event_factor,
                &event_returns,
                EventReturnsOptions {
                    periods_before: 1,
                    periods_after: 1,
                    demeaned: false,
                    group_adjust: false,
                    by_group: false,
                },
            )
            .unwrap(),
        ),
        TearSheetData::from(
            create_event_study_tear_sheet_data(
                &event_factor,
                Some(&event_returns),
                EventStudyTearSheetOptions {
                    event_window: Some((1, 1)),
                    rate_of_return: true,
                    histogram_bins: 2,
                },
            )
            .unwrap(),
        ),
    ]
}

fn reports_without_1d() -> Vec<TearSheetData> {
    let mut rows = Vec::new();
    for day in 0..24_i64 {
        let date = JAN_1 + day * DAY_MS;
        let rotation = day.rem_euclid(4) as usize;
        for asset_index in 0..4 {
            rows.push((
                date,
                ["a", "b", "c", "d"][asset_index],
                ["g1", "g1", "g2", "g2"][asset_index],
                Some((asset_index + 1) as f64),
                Some((asset_index + 1) as u32),
                Some((rotation + asset_index + 1) as f64 * 0.03),
            ));
        }
    }

    let data = DataFrame::new(
        rows.len(),
        vec![
            Series::new(
                "date".into(),
                rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            )
            .into_datetime(
                TimeUnit::Milliseconds,
                TimeZone::opt_try_new(Some("UTC")).unwrap(),
            )
            .into_column(),
            Series::new(
                "asset".into(),
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "group".into(),
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor".into(),
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "factor_quantile".into(),
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "forward_return_3D".into(),
                rows.iter().map(|row| row.5).collect::<Vec<_>>(),
            )
            .into(),
        ],
    )
    .unwrap();

    vec![TearSheetData::from(
        create_returns_tear_sheet_data(
            &data,
            ReturnsTearSheetOptions {
                by_group: true,
                ..ReturnsTearSheetOptions::default()
            },
        )
        .unwrap(),
    )]
}

fn expect_invalid_report(json: &str, expected_reason: &str) {
    let error = TearSheetData::from_json(json).expect_err("corrupt report should be rejected");
    match error {
        FerricAlphaError::InvalidReport { operation, reason } => {
            assert_eq!(operation, "TearSheetData::from_json");
            assert_eq!(reason, expected_reason);
        }
        other => panic!("expected InvalidReport, got {other:?}"),
    }
}

fn first_table_mut(value: &mut Value) -> &mut Value {
    value
        .pointer_mut("/data/quantile_statistics")
        .expect("fixture should contain quantile statistics")
}

fn first_column_data_array_mut(table: &mut Value) -> &mut Vec<Value> {
    let column = table["columns"][0].as_object_mut().unwrap();
    let data = column.get_mut("data").unwrap().as_object_mut().unwrap();
    let payload = data.values_mut().next().unwrap();
    if payload.is_array() {
        payload.as_array_mut().unwrap()
    } else {
        payload["values"].as_array_mut().unwrap()
    }
}

#[test]
fn every_report_kind_round_trips_through_generic_enum_with_stable_compact_json() {
    for report in reports() {
        let compact = report.to_json().expect("compact JSON should serialize");
        assert!(!compact.ends_with('\n'));
        assert!(!compact.contains("NaN"));
        assert!(!compact.contains("Infinity"));

        let decoded = TearSheetData::from_json(&compact).expect("compact JSON should deserialize");
        assert_eq!(decoded, report);
        assert_eq!(
            decoded
                .to_json()
                .expect("decoded report should reserialize"),
            compact
        );

        let pretty = report
            .to_pretty_json()
            .expect("pretty JSON should serialize");
        assert_eq!(
            TearSheetData::from_json(&pretty).expect("pretty JSON should deserialize"),
            report
        );
    }
}

#[test]
fn canonical_contract_reports_expose_metadata() {
    let fixture = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/golden/phase-04/ferric_contract.json"),
    )
    .unwrap();
    let value = serde_json::from_str::<Value>(&fixture).unwrap();
    let expected = [
        ReportKind::Full,
        ReportKind::EventReturns,
        ReportKind::EventStudy,
    ];

    for (report, expected_kind) in value["reports"].as_array().unwrap().iter().zip(expected) {
        let canonical_json = report["canonical_json"].as_str().unwrap();
        let decoded = TearSheetData::from_json(canonical_json).unwrap();
        assert_eq!(decoded.metadata().report_kind(), expected_kind);
        assert_eq!(
            decoded.metadata().contract_version(),
            "ferric-alpha.tear-sheet/v1"
        );
    }
}

#[test]
fn compact_json_is_deterministic_for_equivalent_normalized_input_order() {
    let sorted = TearSheetData::from(
        create_returns_tear_sheet_data(
            &factor_data(false),
            ReturnsTearSheetOptions {
                by_group: true,
                ..ReturnsTearSheetOptions::default()
            },
        )
        .unwrap(),
    );
    let reversed = TearSheetData::from(
        create_returns_tear_sheet_data(
            &factor_data(true),
            ReturnsTearSheetOptions {
                by_group: true,
                ..ReturnsTearSheetOptions::default()
            },
        )
        .unwrap(),
    );

    assert_eq!(sorted, reversed);
    assert_eq!(sorted.to_json().unwrap(), reversed.to_json().unwrap());
}

#[test]
fn from_json_rejects_major_contract_version_mismatch() {
    let report = reports().remove(0);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    value["metadata"]["contract_version"] = Value::String("ferric-alpha.tear-sheet/v2".to_string());
    let json = serde_json::to_string(&value).unwrap();

    let error = TearSheetData::from_json(&json).expect_err("v2 report should be rejected");
    match error {
        FerricAlphaError::ReportVersion { expected, actual } => {
            assert_eq!(expected, "ferric-alpha.tear-sheet/v1");
            assert_eq!(actual, "ferric-alpha.tear-sheet/v2");
        }
        other => panic!("expected ReportVersion, got {other:?}"),
    }
}

#[test]
fn from_json_rejects_report_kind_body_mismatch() {
    let report = reports().remove(0);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    value["metadata"]["report_kind"] = Value::String("returns".to_string());
    let json = serde_json::to_string(&value).unwrap();

    expect_invalid_report(&json, "report kind/body mismatch");
}

#[test]
fn from_json_rejects_corrupt_table_row_count() {
    let report = reports().remove(0);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    first_table_mut(&mut value)["row_count"] = Value::Number(99.into());
    let json = serde_json::to_string(&value).unwrap();

    expect_invalid_report(&json, "row count does not match column data");
}

#[test]
fn from_json_rejects_corrupt_table_column_lengths() {
    let report = reports().remove(0);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let table = first_table_mut(&mut value);
    first_column_data_array_mut(table).pop();
    table["row_count"] = Value::Number(3.into());
    let json = serde_json::to_string(&value).unwrap();

    expect_invalid_report(&json, "column lengths differ");
}

#[test]
fn from_json_rejects_corrupt_table_sort_order() {
    let report = reports().remove(0);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    first_column_data_array_mut(first_table_mut(&mut value)).swap(0, 1);
    let json = serde_json::to_string(&value).unwrap();

    expect_invalid_report(&json, "rows are not sorted by sort_by");
}

#[test]
fn from_json_rejects_out_of_range_json_float() {
    let report = reports().remove(0);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let table = first_table_mut(&mut value);
    let column = table["columns"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|column| column["data"].get("float64").is_some())
        .unwrap();
    column["data"]["float64"][0] = Value::String("__OUT_OF_RANGE_FLOAT__".to_string());
    let json = serde_json::to_string(&value)
        .unwrap()
        .replace("\"__OUT_OF_RANGE_FLOAT__\"", "1e309");

    expect_invalid_report(&json, "invalid JSON");
}

#[test]
fn table_ids_are_semantic_order_and_unknown_table_returns_none() {
    let full = TearSheetData::from(
        create_full_tear_sheet_data(
            &factor_data(false),
            FullTearSheetOptions {
                by_group: true,
                turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                ic_rolling_window: 3,
                ..FullTearSheetOptions::default()
            },
        )
        .unwrap(),
    );

    let ids = full.table_ids();
    assert_eq!(
        ids,
        vec![
            "quantile.statistics",
            "returns.summary",
            "returns.mean_by_quantile",
            "returns.daily_by_quantile",
            "returns.daily_spread",
            "returns.factor_returns",
            "information.summary",
            "information.ic_by_date",
            "information.ic_rolling",
            "information.ic_monthly",
            "information.qq_normal",
            "turnover.mean_by_quantile",
            "turnover.mean_rank_autocorrelation",
            "turnover.quantile_series",
            "turnover.rank_autocorrelation_series",
            "returns.group_mean_by_quantile",
            "returns.factor_cumulative_1d",
            "returns.quantile_cumulative_1d",
            "information.ic_by_group",
        ]
    );
    for id in ids {
        let table = full.table(id).expect("known table id should resolve");
        assert_eq!(table.id(), id);
    }
    assert!(full.table("missing.table").is_none());
}

#[test]
fn from_json_rejects_duplicate_table_ids_in_full_reports() {
    let full = reports().remove(4);
    let mut value = serde_json::from_str::<Value>(&full.to_json().unwrap()).unwrap();
    value["data"]["returns"]["summary"]["id"] = Value::String("quantile.statistics".to_string());
    let json = serde_json::to_string(&value).unwrap();

    expect_invalid_report(&json, "unexpected table id");
}

#[test]
fn from_json_rejects_table_id_renamed_inside_known_slot() {
    let report = reports().remove(1);
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    value["data"]["summary"]["id"] = Value::String("returns.renamed_summary".to_string());
    let json = serde_json::to_string(&value).unwrap();

    expect_invalid_report(&json, "unexpected table id");
}

#[test]
#[ignore = "exports a Phase 05 no-1D report fixture for the golden generator"]
fn export_phase_05_report_cases() {
    let out = std::env::var("FERRIC_ALPHA_PHASE05_NO1D_OUT")
        .expect("FERRIC_ALPHA_PHASE05_NO1D_OUT must point to an output file");
    let report = reports_without_1d().remove(0);
    assert!(
        !report
            .metadata()
            .periods()
            .iter()
            .any(|period| period == "1D")
    );
    match &report {
        TearSheetData::Returns(returns) => {
            assert!(returns.data.factor_cumulative_1d.is_none());
            assert!(returns.data.quantile_cumulative_1d.is_none());
        }
        other => panic!("expected returns report, got {other:?}"),
    }
    std::fs::write(out, report.to_json().unwrap()).unwrap();
}
