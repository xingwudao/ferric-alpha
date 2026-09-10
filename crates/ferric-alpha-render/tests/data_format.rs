mod common;

use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportTable,
    ReportTimeUnit,
};
use ferric_alpha_render::{
    DatetimeColumn, RenderError, bools, datetimes, float64, format_datetime, format_number,
    format_optional, int32, int64, require_table, strings, uint32, uint64,
};

#[test]
fn checked_accessors_preserve_order_and_nulls() {
    let table = all_types_table();

    assert_eq!(
        bools(&table, "flag").unwrap(),
        &[Some(true), None, Some(false)]
    );
    assert_eq!(int32(&table, "i32").unwrap(), &[Some(1), None, Some(-3)]);
    assert_eq!(int64(&table, "i64").unwrap(), &[Some(10), None, Some(-30)]);
    assert_eq!(uint32(&table, "u32").unwrap(), &[Some(1), None, Some(3)]);
    assert_eq!(uint64(&table, "u64").unwrap(), &[Some(10), None, Some(30)]);
    assert_eq!(
        float64(&table, "f64").unwrap(),
        &[Some(1.25), None, Some(-3.5)]
    );
    assert_eq!(
        strings(&table, "label").unwrap(),
        &[Some("a".to_string()), None, Some("c".to_string())]
    );
    assert_eq!(
        datetimes(&table, "date").unwrap(),
        DatetimeColumn {
            values: &[Some(1_704_067_200_000), None, Some(1_704_153_600_000)],
            time_unit: ReportTimeUnit::Milliseconds,
            timezone: Some("UTC"),
        }
    );
}

#[test]
fn checked_accessors_report_missing_and_wrong_types() {
    let report = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "full")
        .unwrap()
        .1;
    assert_missing_table(require_table(&report, "missing.table").unwrap_err());

    let table = all_types_table();
    assert_missing_column(float64(&table, "missing").unwrap_err());
    assert_wrong_dtype(float64(&table, "label").unwrap_err());
}

#[test]
fn report_case_fixture_decodes_all_projected_cases() {
    let names = common::phase_05_report_cases()
        .into_iter()
        .map(|(name, report)| {
            assert_eq!(
                report.metadata().contract_version(),
                "ferric-alpha.tear-sheet/v1"
            );
            name
        })
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "summary",
            "returns",
            "returns_without_group",
            "returns_without_1d_cumulative",
            "information",
            "information_without_group",
            "turnover",
            "full",
            "event_returns",
            "event_study",
            "event_study_without_event_returns",
        ]
    );
}

#[test]
fn display_units_format_stably() {
    assert_eq!(
        format_number(0.01234, DisplayUnit::DecimalReturn, 2),
        "1.23%"
    );
    assert_eq!(
        format_number(0.0012, DisplayUnit::BasisPoints, 3),
        "12.000 bps"
    );
    assert_eq!(format_number(0.125, DisplayUnit::Percent, 1), "12.5%");
    assert_eq!(format_number(-0.0, DisplayUnit::Ratio, 3), "0.000");
    assert_eq!(format_optional(None, DisplayUnit::Raw, 2), "\u{2014}");
    assert_eq!(format_optional(Some(7.0), DisplayUnit::Count, 0), "7");
}

#[test]
fn datetimes_format_by_unit_and_timezone() {
    assert_eq!(
        format_datetime(1_704_067_200_000_000_000, ReportTimeUnit::Nanoseconds, None).unwrap(),
        "2024-01-01"
    );
    assert_eq!(
        format_datetime(
            1_704_067_200_000_000,
            ReportTimeUnit::Microseconds,
            Some("UTC")
        )
        .unwrap(),
        "2024-01-01"
    );
    assert_eq!(
        format_datetime(
            1_704_124_800_000,
            ReportTimeUnit::Milliseconds,
            Some("Asia/Shanghai")
        )
        .unwrap(),
        "2024-01-02"
    );

    match format_datetime(
        1_704_067_200_000,
        ReportTimeUnit::Milliseconds,
        Some("Bad/Zone"),
    )
    .unwrap_err()
    {
        RenderError::InvalidContract { reason } => {
            assert_eq!(reason, "unknown timezone Bad/Zone");
        }
        other => panic!("expected InvalidContract, got {other:?}"),
    }
}

fn all_types_table() -> ReportTable {
    ReportTable::new(
        "fixture.all_types",
        "All Types",
        vec!["row".to_string()],
        vec![
            ReportColumn::new(
                "row",
                "Row",
                ColumnRole::Dimension,
                None,
                ReportColumnData::UInt32(vec![Some(1), Some(2), Some(3)]),
            ),
            ReportColumn::new(
                "flag",
                "Flag",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Boolean(vec![Some(true), None, Some(false)]),
            ),
            ReportColumn::new(
                "i32",
                "I32",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Int32(vec![Some(1), None, Some(-3)]),
            ),
            ReportColumn::new(
                "i64",
                "I64",
                ColumnRole::Measure,
                None,
                ReportColumnData::Int64(vec![Some(10), None, Some(-30)]),
            ),
            ReportColumn::new(
                "u32",
                "U32",
                ColumnRole::Measure,
                None,
                ReportColumnData::UInt32(vec![Some(1), None, Some(3)]),
            ),
            ReportColumn::new(
                "u64",
                "U64",
                ColumnRole::Measure,
                None,
                ReportColumnData::UInt64(vec![Some(10), None, Some(30)]),
            ),
            ReportColumn::new(
                "f64",
                "F64",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::Ratio, 2)),
                ReportColumnData::Float64(vec![Some(1.25), None, Some(-3.5)]),
            ),
            ReportColumn::new(
                "label",
                "Label",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(vec![Some("a".to_string()), None, Some("c".to_string())]),
            ),
            ReportColumn::new(
                "date",
                "Date",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: vec![Some(1_704_067_200_000), None, Some(1_704_153_600_000)],
                    time_unit: ReportTimeUnit::Milliseconds,
                    timezone: Some("UTC".to_string()),
                },
            ),
        ],
    )
    .unwrap()
}

fn assert_missing_table(error: RenderError) {
    match error {
        RenderError::MissingTable { id } => assert_eq!(id, "missing.table"),
        other => panic!("expected MissingTable, got {other:?}"),
    }
}

fn assert_missing_column(error: RenderError) {
    match error {
        RenderError::MissingColumn { table, column } => {
            assert_eq!(table, "fixture.all_types");
            assert_eq!(column, "missing");
        }
        other => panic!("expected MissingColumn, got {other:?}"),
    }
}

fn assert_wrong_dtype(error: RenderError) {
    match error {
        RenderError::InvalidColumnType {
            table,
            column,
            expected,
        } => {
            assert_eq!(table, "fixture.all_types");
            assert_eq!(column, "label");
            assert_eq!(expected, "float64");
        }
        other => panic!("expected InvalidColumnType, got {other:?}"),
    }
}
