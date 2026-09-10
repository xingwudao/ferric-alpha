use std::cmp::Ordering;
use std::collections::HashSet;

use polars::prelude::{Column, DataFrame, IntoColumn, NamedFrom, Series, TimeUnit, TimeZone};
use serde::{Deserialize, Serialize};

use crate::{FerricAlphaError, Result};

const NEW_OPERATION: &str = "ReportTable::new";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportTable {
    id: String,
    title: String,
    row_count: usize,
    sort_by: Vec<String>,
    columns: Vec<ReportColumn>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportColumn {
    name: String,
    label: String,
    role: ColumnRole,
    display: Option<DisplayFormat>,
    data: ReportColumnData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportColumnData {
    Boolean(Vec<Option<bool>>),
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    UInt32(Vec<Option<u32>>),
    UInt64(Vec<Option<u64>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Datetime {
        values: Vec<Option<i64>>,
        time_unit: ReportTimeUnit,
        timezone: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnRole {
    Dimension,
    Measure,
    Statistic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportTimeUnit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayFormat {
    unit: DisplayUnit,
    decimals: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayUnit {
    Raw,
    DecimalReturn,
    BasisPoints,
    Percent,
    Ratio,
    Count,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SortValue<'a> {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float64(f64),
    String(&'a str),
    Datetime(i64),
}

impl ReportTable {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        sort_by: Vec<String>,
        columns: Vec<ReportColumn>,
    ) -> Result<Self> {
        let id = id.into();
        if !valid_table_id(&id) {
            return invalid("invalid table id");
        }

        let mut names = HashSet::with_capacity(columns.len());
        for column in &columns {
            if !names.insert(column.name.as_str()) {
                return invalid("duplicate column name");
            }
            if matches!(&column.data, ReportColumnData::Float64(values) if values
                .iter()
                .flatten()
                .any(|value| !value.is_finite()))
            {
                return invalid("float value is not finite");
            }
        }

        let row_count = match columns.first() {
            Some(column) => column.data.len(),
            None => 0,
        };
        if columns.iter().any(|column| column.data.len() != row_count) {
            return invalid("column lengths differ");
        }

        let sort_columns = sort_by
            .iter()
            .map(|name| {
                columns.iter().find(|column| column.name == *name).ok_or(
                    FerricAlphaError::InvalidReport {
                        operation: NEW_OPERATION,
                        reason: "sort key missing",
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;

        for row in 0..row_count {
            if sort_columns
                .iter()
                .any(|column| column.data.sort_value(row).is_none())
            {
                return invalid("sort key contains null");
            }
        }

        for row in 1..row_count {
            let ordering = compare_rows(&sort_columns, row - 1, row);
            if ordering == Some(Ordering::Greater) {
                return invalid("rows are not sorted by sort_by");
            }
        }

        Ok(Self {
            id,
            title: title.into(),
            row_count,
            sort_by,
            columns,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn sort_by(&self) -> &[String] {
        &self.sort_by
    }

    pub fn columns(&self) -> &[ReportColumn] {
        &self.columns
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .map(|column| column.name.as_str())
            .collect()
    }

    pub fn to_polars(&self) -> Result<DataFrame> {
        let columns = self
            .columns
            .iter()
            .map(ReportColumn::to_polars)
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new(self.row_count, columns)?)
    }
}

impl ReportColumn {
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        role: ColumnRole,
        display: Option<DisplayFormat>,
        data: ReportColumnData,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            role,
            display,
            data,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn role(&self) -> ColumnRole {
        self.role
    }

    pub fn display(&self) -> Option<&DisplayFormat> {
        self.display.as_ref()
    }

    pub fn data(&self) -> &ReportColumnData {
        &self.data
    }

    fn to_polars(&self) -> Result<Column> {
        let column = match &self.data {
            ReportColumnData::Boolean(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::Int32(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::Int64(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::UInt32(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::UInt64(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::Float64(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::String(values) => {
                Series::new(self.name.as_str().into(), values).into_column()
            }
            ReportColumnData::Datetime {
                values,
                time_unit,
                timezone,
            } => Series::new(self.name.as_str().into(), values)
                .into_datetime(
                    (*time_unit).into(),
                    TimeZone::opt_try_new(timezone.as_deref())?,
                )
                .into_column(),
        };
        Ok(column)
    }
}

impl ReportColumnData {
    pub fn len(&self) -> usize {
        match self {
            Self::Boolean(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::UInt32(values) => values.len(),
            Self::UInt64(values) => values.len(),
            Self::Float64(values) => values.len(),
            Self::String(values) => values.len(),
            Self::Datetime { values, .. } => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn sort_value(&self, row: usize) -> Option<SortValue<'_>> {
        match self {
            Self::Boolean(values) => values.get(row).copied().flatten().map(SortValue::Boolean),
            Self::Int32(values) => values.get(row).copied().flatten().map(SortValue::Int32),
            Self::Int64(values) => values.get(row).copied().flatten().map(SortValue::Int64),
            Self::UInt32(values) => values.get(row).copied().flatten().map(SortValue::UInt32),
            Self::UInt64(values) => values.get(row).copied().flatten().map(SortValue::UInt64),
            Self::Float64(values) => values.get(row).copied().flatten().map(SortValue::Float64),
            Self::String(values) => values
                .get(row)
                .and_then(Option::as_deref)
                .map(SortValue::String),
            Self::Datetime { values, .. } => {
                values.get(row).copied().flatten().map(SortValue::Datetime)
            }
        }
    }
}

impl DisplayFormat {
    pub fn new(unit: DisplayUnit, decimals: u8) -> Self {
        Self { unit, decimals }
    }

    pub fn unit(&self) -> DisplayUnit {
        self.unit
    }

    pub fn decimals(&self) -> u8 {
        self.decimals
    }
}

impl From<ReportTimeUnit> for TimeUnit {
    fn from(value: ReportTimeUnit) -> Self {
        match value {
            ReportTimeUnit::Nanoseconds => Self::Nanoseconds,
            ReportTimeUnit::Microseconds => Self::Microseconds,
            ReportTimeUnit::Milliseconds => Self::Milliseconds,
        }
    }
}

fn invalid<T>(reason: &'static str) -> Result<T> {
    Err(FerricAlphaError::InvalidReport {
        operation: NEW_OPERATION,
        reason,
    })
}

fn compare_rows(columns: &[&ReportColumn], left: usize, right: usize) -> Option<Ordering> {
    for column in columns {
        let left = column.data.sort_value(left)?;
        let right = column.data.sort_value(right)?;
        let ordering = left.partial_cmp(&right)?;
        if ordering != Ordering::Equal {
            return Some(ordering);
        }
    }
    Some(Ordering::Equal)
}

fn valid_table_id(id: &str) -> bool {
    !id.is_empty() && id.split('.').all(valid_table_id_segment)
}

fn valid_table_id_segment(segment: &str) -> bool {
    if segment.is_empty()
        || segment.starts_with('_')
        || segment.ends_with('_')
        || segment.contains("__")
    {
        return false;
    }

    segment.chars().enumerate().all(|(index, character)| {
        character.is_ascii_lowercase()
            || (index > 0 && (character.is_ascii_digit() || character == '_'))
    })
}

impl PartialOrd for SortValue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Boolean(left), Self::Boolean(right)) => left.partial_cmp(right),
            (Self::Int32(left), Self::Int32(right)) => left.partial_cmp(right),
            (Self::Int64(left), Self::Int64(right)) => left.partial_cmp(right),
            (Self::UInt32(left), Self::UInt32(right)) => left.partial_cmp(right),
            (Self::UInt64(left), Self::UInt64(right)) => left.partial_cmp(right),
            (Self::Float64(left), Self::Float64(right)) => left.partial_cmp(right),
            (Self::String(left), Self::String(right)) => left.partial_cmp(right),
            (Self::Datetime(left), Self::Datetime(right)) => left.partial_cmp(right),
            _ => None,
        }
    }
}
