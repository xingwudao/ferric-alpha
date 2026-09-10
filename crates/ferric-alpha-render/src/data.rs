use ferric_alpha::{ReportColumnData, ReportTable, ReportTimeUnit, TearSheetData};

use crate::{RenderError, RenderResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatetimeColumn<'a> {
    pub values: &'a [Option<i64>],
    pub time_unit: ReportTimeUnit,
    pub timezone: Option<&'a str>,
}

pub fn require_table<'a>(report: &'a TearSheetData, id: &str) -> RenderResult<&'a ReportTable> {
    report
        .table(id)
        .ok_or_else(|| RenderError::MissingTable { id: id.to_string() })
}

pub fn bools<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<bool>]> {
    match column_data(table, name)? {
        ReportColumnData::Boolean(values) => Ok(values),
        _ => wrong_type(table, name, "boolean"),
    }
}

pub fn int32<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<i32>]> {
    match column_data(table, name)? {
        ReportColumnData::Int32(values) => Ok(values),
        _ => wrong_type(table, name, "int32"),
    }
}

pub fn int64<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<i64>]> {
    match column_data(table, name)? {
        ReportColumnData::Int64(values) => Ok(values),
        _ => wrong_type(table, name, "int64"),
    }
}

pub fn uint32<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<u32>]> {
    match column_data(table, name)? {
        ReportColumnData::UInt32(values) => Ok(values),
        _ => wrong_type(table, name, "uint32"),
    }
}

pub fn uint64<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<u64>]> {
    match column_data(table, name)? {
        ReportColumnData::UInt64(values) => Ok(values),
        _ => wrong_type(table, name, "uint64"),
    }
}

pub fn float64<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<f64>]> {
    match column_data(table, name)? {
        ReportColumnData::Float64(values) => Ok(values),
        _ => wrong_type(table, name, "float64"),
    }
}

pub fn strings<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a [Option<String>]> {
    match column_data(table, name)? {
        ReportColumnData::String(values) => Ok(values),
        _ => wrong_type(table, name, "string"),
    }
}

pub fn datetimes<'a>(table: &'a ReportTable, name: &str) -> RenderResult<DatetimeColumn<'a>> {
    match column_data(table, name)? {
        ReportColumnData::Datetime {
            values,
            time_unit,
            timezone,
        } => Ok(DatetimeColumn {
            values,
            time_unit: *time_unit,
            timezone: timezone.as_deref(),
        }),
        _ => wrong_type(table, name, "datetime"),
    }
}

fn column_data<'a>(table: &'a ReportTable, name: &str) -> RenderResult<&'a ReportColumnData> {
    table
        .columns()
        .iter()
        .find(|column| column.name() == name)
        .map(|column| column.data())
        .ok_or_else(|| RenderError::MissingColumn {
            table: table.id().to_string(),
            column: name.to_string(),
        })
}

fn wrong_type<T>(table: &ReportTable, name: &str, expected: &'static str) -> RenderResult<T> {
    Err(RenderError::InvalidColumnType {
        table: table.id().to_string(),
        column: name.to_string(),
        expected,
    })
}
