use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, FerricAlphaError, ReportColumn, ReportColumnData,
    ReportTable, ReportTimeUnit,
};
use polars::prelude::{DataType, TimeUnit, TimeZone};

fn display(unit: DisplayUnit, decimals: u8) -> Option<DisplayFormat> {
    Some(DisplayFormat::new(unit, decimals))
}

fn all_dtype_columns() -> Vec<ReportColumn> {
    vec![
        ReportColumn::new(
            "date",
            "Date",
            ColumnRole::Dimension,
            None,
            ReportColumnData::Datetime {
                values: vec![Some(1_704_067_200_000), Some(1_704_153_600_000)],
                time_unit: ReportTimeUnit::Milliseconds,
                timezone: Some("UTC".to_string()),
            },
        ),
        ReportColumn::new(
            "group",
            "Group",
            ColumnRole::Dimension,
            None,
            ReportColumnData::String(vec![Some("growth".to_string()), None]),
        ),
        ReportColumn::new(
            "flag",
            "Flag",
            ColumnRole::Dimension,
            None,
            ReportColumnData::Boolean(vec![Some(true), None]),
        ),
        ReportColumn::new(
            "bucket_i32",
            "Bucket I32",
            ColumnRole::Dimension,
            display(DisplayUnit::Count, 0),
            ReportColumnData::Int32(vec![Some(-1), None]),
        ),
        ReportColumn::new(
            "score_i64",
            "Score I64",
            ColumnRole::Statistic,
            display(DisplayUnit::Raw, 0),
            ReportColumnData::Int64(vec![Some(-10), None]),
        ),
        ReportColumn::new(
            "quantile",
            "Quantile",
            ColumnRole::Dimension,
            display(DisplayUnit::Count, 0),
            ReportColumnData::UInt32(vec![Some(1), None]),
        ),
        ReportColumn::new(
            "volume",
            "Volume",
            ColumnRole::Measure,
            display(DisplayUnit::Count, 0),
            ReportColumnData::UInt64(vec![Some(100), None]),
        ),
        ReportColumn::new(
            "return",
            "Return",
            ColumnRole::Measure,
            display(DisplayUnit::DecimalReturn, 6),
            ReportColumnData::Float64(vec![Some(0.01), None]),
        ),
    ]
}

fn invalid_report_reason(result: ferric_alpha::Result<ReportTable>) -> &'static str {
    match result {
        Err(FerricAlphaError::InvalidReport { operation, reason }) => {
            assert_eq!(operation, "ReportTable::new");
            reason
        }
        other => panic!("expected InvalidReport, got {other:?}"),
    }
}

#[test]
fn constructs_table_with_every_dtype_and_converts_to_polars() {
    let table = ReportTable::new(
        "returns.by_quantile",
        "Returns By Quantile",
        vec!["date".to_string()],
        all_dtype_columns(),
    )
    .unwrap();

    assert_eq!(table.id(), "returns.by_quantile");
    assert_eq!(table.title(), "Returns By Quantile");
    assert_eq!(table.row_count(), 2);
    assert_eq!(table.sort_by(), &["date"]);
    assert_eq!(
        table.column_names(),
        vec![
            "date",
            "group",
            "flag",
            "bucket_i32",
            "score_i64",
            "quantile",
            "volume",
            "return"
        ]
    );
    assert_eq!(table.columns()[0].role(), ColumnRole::Dimension);
    assert_eq!(table.columns()[7].role(), ColumnRole::Measure);
    assert_eq!(
        table.columns()[7].display(),
        Some(&DisplayFormat::new(DisplayUnit::DecimalReturn, 6))
    );

    let frame = table.to_polars().unwrap();
    assert_eq!(frame.height(), 2);
    let frame_column_names = frame
        .get_column_names()
        .into_iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        frame_column_names,
        vec![
            "date",
            "group",
            "flag",
            "bucket_i32",
            "score_i64",
            "quantile",
            "volume",
            "return"
        ]
    );
    assert_eq!(
        frame.column("date").unwrap().dtype(),
        &DataType::Datetime(
            TimeUnit::Milliseconds,
            TimeZone::opt_try_new(Some("UTC")).unwrap()
        )
    );
    assert_eq!(frame.column("group").unwrap().dtype(), &DataType::String);
    assert_eq!(frame.column("flag").unwrap().dtype(), &DataType::Boolean);
    assert_eq!(
        frame.column("bucket_i32").unwrap().dtype(),
        &DataType::Int32
    );
    assert_eq!(frame.column("score_i64").unwrap().dtype(), &DataType::Int64);
    assert_eq!(frame.column("quantile").unwrap().dtype(), &DataType::UInt32);
    assert_eq!(frame.column("volume").unwrap().dtype(), &DataType::UInt64);
    assert_eq!(frame.column("return").unwrap().dtype(), &DataType::Float64);

    let date = frame.column("date").unwrap().datetime().unwrap();
    assert_eq!(date.phys.get(0), Some(1_704_067_200_000));
    assert_eq!(date.phys.get(1), Some(1_704_153_600_000));
    assert_eq!(frame.column("group").unwrap().str().unwrap().get(1), None);
    assert_eq!(frame.column("flag").unwrap().bool().unwrap().get(1), None);
    assert_eq!(
        frame.column("bucket_i32").unwrap().i32().unwrap().get(1),
        None
    );
    assert_eq!(
        frame.column("score_i64").unwrap().i64().unwrap().get(1),
        None
    );
    assert_eq!(
        frame.column("quantile").unwrap().u32().unwrap().get(1),
        None
    );
    assert_eq!(frame.column("volume").unwrap().u64().unwrap().get(1), None);
    assert_eq!(frame.column("return").unwrap().f64().unwrap().get(1), None);
}

#[test]
fn rejects_duplicate_column_names() {
    let mut columns = all_dtype_columns();
    columns.push(ReportColumn::new(
        "date",
        "Duplicate Date",
        ColumnRole::Dimension,
        None,
        ReportColumnData::Int64(vec![Some(1), Some(2)]),
    ));

    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["date".to_string()],
            columns,
        )),
        "duplicate column name"
    );
}

#[test]
fn rejects_unequal_column_lengths() {
    let mut columns = all_dtype_columns();
    columns[1] = ReportColumn::new(
        "group",
        "Group",
        ColumnRole::Dimension,
        None,
        ReportColumnData::String(vec![Some("growth".to_string())]),
    );

    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["date".to_string()],
            columns,
        )),
        "column lengths differ"
    );
}

#[test]
fn rejects_missing_sort_key() {
    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["missing".to_string()],
            all_dtype_columns(),
        )),
        "sort key missing"
    );
}

#[test]
fn rejects_null_sort_key() {
    let mut columns = all_dtype_columns();
    columns[1] = ReportColumn::new(
        "group",
        "Group",
        ColumnRole::Dimension,
        None,
        ReportColumnData::String(vec![Some("growth".to_string()), None]),
    );

    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["group".to_string()],
            columns,
        )),
        "sort key contains null"
    );
}

#[test]
fn rejects_rows_not_ordered_by_typed_sort_keys() {
    let mut columns = all_dtype_columns();
    columns[5] = ReportColumn::new(
        "quantile",
        "Quantile",
        ColumnRole::Dimension,
        display(DisplayUnit::Count, 0),
        ReportColumnData::UInt32(vec![Some(2), Some(1)]),
    );

    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["quantile".to_string()],
            columns,
        )),
        "rows are not sorted by sort_by"
    );
}

#[test]
fn rejects_invalid_id() {
    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "Returns.ByQuantile",
            "Returns By Quantile",
            vec!["date".to_string()],
            all_dtype_columns(),
        )),
        "invalid table id"
    );
}

#[test]
fn rejects_nan_float_values() {
    let mut columns = all_dtype_columns();
    columns[7] = ReportColumn::new(
        "return",
        "Return",
        ColumnRole::Measure,
        display(DisplayUnit::DecimalReturn, 6),
        ReportColumnData::Float64(vec![Some(f64::NAN), None]),
    );

    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["date".to_string()],
            columns,
        )),
        "float value is not finite"
    );
}

#[test]
fn rejects_infinite_float_values() {
    let mut columns = all_dtype_columns();
    columns[7] = ReportColumn::new(
        "return",
        "Return",
        ColumnRole::Measure,
        display(DisplayUnit::DecimalReturn, 6),
        ReportColumnData::Float64(vec![Some(f64::INFINITY), None]),
    );

    assert_eq!(
        invalid_report_reason(ReportTable::new(
            "returns.by_quantile",
            "Returns By Quantile",
            vec!["date".to_string()],
            columns,
        )),
        "float value is not finite"
    );
}
