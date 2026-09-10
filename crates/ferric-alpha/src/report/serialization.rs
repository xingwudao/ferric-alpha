use std::cmp::Ordering;
use std::collections::HashSet;

use serde::Deserialize;
use serde_json::Value;

use crate::report::metadata::CONTRACT_VERSION;
use crate::report::{
    EventReturnsReportData, EventReturnsTearSheetData, EventStudyReportData,
    EventStudyTearSheetData, FullReportData, FullTearSheetData, InformationReportData,
    InformationTearSheetData, ReportColumn, ReportColumnData, ReportKind, ReportMetadata,
    ReportTable, ReturnsReportData, ReturnsTearSheetData, SummaryTearSheetData, TurnoverReportData,
    TurnoverTearSheetData,
};
use crate::{FerricAlphaError, Result};

const OPERATION: &str = "TearSheetData::from_json";

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum TearSheetData {
    Summary(SummaryTearSheetData),
    Returns(ReturnsTearSheetData),
    Information(InformationTearSheetData),
    Turnover(TurnoverTearSheetData),
    Full(FullTearSheetData),
    EventReturns(EventReturnsTearSheetData),
    EventStudy(EventStudyTearSheetData),
}

#[derive(Debug, Deserialize)]
struct Header {
    metadata: HeaderMetadata,
}

#[derive(Debug, Deserialize)]
struct HeaderMetadata {
    contract_version: String,
    report_kind: ReportKind,
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

impl TearSheetData {
    pub fn to_json(&self) -> Result<String> {
        match self {
            Self::Summary(report) => serialize_compact(report),
            Self::Returns(report) => serialize_compact(report),
            Self::Information(report) => serialize_compact(report),
            Self::Turnover(report) => serialize_compact(report),
            Self::Full(report) => serialize_compact(report),
            Self::EventReturns(report) => serialize_compact(report),
            Self::EventStudy(report) => serialize_compact(report),
        }
    }

    pub fn to_pretty_json(&self) -> Result<String> {
        match self {
            Self::Summary(report) => serialize_pretty(report),
            Self::Returns(report) => serialize_pretty(report),
            Self::Information(report) => serialize_pretty(report),
            Self::Turnover(report) => serialize_pretty(report),
            Self::Full(report) => serialize_pretty(report),
            Self::EventReturns(report) => serialize_pretty(report),
            Self::EventStudy(report) => serialize_pretty(report),
        }
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let value =
            serde_json::from_str::<Value>(json).map_err(|_| invalid_error("invalid JSON"))?;
        let header = serde_json::from_value::<Header>(value.clone())
            .map_err(|_| invalid_error("report kind/body mismatch"))?;
        if header.metadata.contract_version != CONTRACT_VERSION {
            return Err(FerricAlphaError::ReportVersion {
                expected: CONTRACT_VERSION,
                actual: header.metadata.contract_version,
            });
        }

        match header.metadata.report_kind {
            ReportKind::Summary => decode::<SummaryTearSheetData>(value, Self::Summary),
            ReportKind::Returns => decode::<ReturnsTearSheetData>(value, Self::Returns),
            ReportKind::Information => decode::<InformationTearSheetData>(value, Self::Information),
            ReportKind::Turnover => decode::<TurnoverTearSheetData>(value, Self::Turnover),
            ReportKind::Full => decode::<FullTearSheetData>(value, Self::Full),
            ReportKind::EventReturns => {
                decode::<EventReturnsTearSheetData>(value, Self::EventReturns)
            }
            ReportKind::EventStudy => decode::<EventStudyTearSheetData>(value, Self::EventStudy),
        }
    }

    pub fn table_ids(&self) -> Vec<&str> {
        self.table_refs().into_iter().map(ReportTable::id).collect()
    }

    pub fn table(&self, id: &str) -> Option<&ReportTable> {
        self.table_refs().into_iter().find(|table| table.id() == id)
    }

    pub fn metadata(&self) -> &ReportMetadata {
        match self {
            Self::Summary(report) => &report.metadata,
            Self::Returns(report) => &report.metadata,
            Self::Information(report) => &report.metadata,
            Self::Turnover(report) => &report.metadata,
            Self::Full(report) => &report.metadata,
            Self::EventReturns(report) => &report.metadata,
            Self::EventStudy(report) => &report.metadata,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Summary(report) => {
                validate_metadata(&report.metadata, ReportKind::Summary)?;
                validate_summary_ids(report)?;
                validate_tables(&summary_tables(report))?;
            }
            Self::Returns(report) => {
                validate_metadata(&report.metadata, ReportKind::Returns)?;
                validate_returns_ids(&report.data)?;
                validate_tables(&returns_tables(&report.data))?;
            }
            Self::Information(report) => {
                validate_metadata(&report.metadata, ReportKind::Information)?;
                validate_information_ids(&report.data)?;
                validate_tables(&information_tables(&report.data))?;
            }
            Self::Turnover(report) => {
                validate_metadata(&report.metadata, ReportKind::Turnover)?;
                validate_turnover_ids(&report.data)?;
                validate_tables(&turnover_tables(&report.data))?;
            }
            Self::Full(report) => {
                validate_metadata(&report.metadata, ReportKind::Full)?;
                validate_full_ids(&report.data)?;
                validate_tables(&full_tables(&report.data))?;
            }
            Self::EventReturns(report) => {
                validate_metadata(&report.metadata, ReportKind::EventReturns)?;
                validate_event_returns_ids(&report.data)?;
                validate_tables(&event_returns_tables(&report.data))?;
            }
            Self::EventStudy(report) => {
                validate_metadata(&report.metadata, ReportKind::EventStudy)?;
                validate_event_study_ids(&report.data)?;
                validate_tables(&event_study_tables(&report.data))?;
            }
        }
        Ok(())
    }

    fn table_refs(&self) -> Vec<&ReportTable> {
        match self {
            Self::Summary(report) => summary_tables(report),
            Self::Returns(report) => returns_tables(&report.data),
            Self::Information(report) => information_tables(&report.data),
            Self::Turnover(report) => turnover_tables(&report.data),
            Self::Full(report) => full_tables(&report.data),
            Self::EventReturns(report) => event_returns_tables(&report.data),
            Self::EventStudy(report) => event_study_tables(&report.data),
        }
    }
}

impl From<SummaryTearSheetData> for TearSheetData {
    fn from(value: SummaryTearSheetData) -> Self {
        Self::Summary(canonicalized(value))
    }
}

impl From<ReturnsTearSheetData> for TearSheetData {
    fn from(value: ReturnsTearSheetData) -> Self {
        Self::Returns(canonicalized(value))
    }
}

impl From<InformationTearSheetData> for TearSheetData {
    fn from(value: InformationTearSheetData) -> Self {
        Self::Information(canonicalized(value))
    }
}

impl From<TurnoverTearSheetData> for TearSheetData {
    fn from(value: TurnoverTearSheetData) -> Self {
        Self::Turnover(canonicalized(value))
    }
}

impl From<FullTearSheetData> for TearSheetData {
    fn from(value: FullTearSheetData) -> Self {
        Self::Full(canonicalized(value))
    }
}

impl From<EventReturnsTearSheetData> for TearSheetData {
    fn from(value: EventReturnsTearSheetData) -> Self {
        Self::EventReturns(canonicalized(value))
    }
}

impl From<EventStudyTearSheetData> for TearSheetData {
    fn from(value: EventStudyTearSheetData) -> Self {
        Self::EventStudy(canonicalized(value))
    }
}

fn canonicalized<T>(report: T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut json = serde_json::to_string(&report).expect("valid report should serialize");
    for _ in 0..16 {
        let next = serde_json::from_str::<T>(&json).expect("valid report JSON should deserialize");
        let next_json = serde_json::to_string(&next).expect("valid report should serialize");
        if next_json == json {
            return next;
        }
        json = next_json;
    }
    serde_json::from_str(&json).expect("valid report JSON should deserialize")
}

fn serialize_compact<T: serde::Serialize>(report: &T) -> Result<String> {
    serde_json::to_string(report).map_err(|_| invalid_error("serialization failed"))
}

fn serialize_pretty<T: serde::Serialize>(report: &T) -> Result<String> {
    serde_json::to_string_pretty(report).map_err(|_| invalid_error("serialization failed"))
}

fn decode<T>(value: Value, wrap: impl FnOnce(T) -> TearSheetData) -> Result<TearSheetData>
where
    T: serde::de::DeserializeOwned,
{
    let report = serde_json::from_value::<T>(value)
        .map_err(|_| invalid_error("report kind/body mismatch"))
        .map(wrap)?;
    report.validate()?;
    Ok(report)
}

fn validate_metadata(metadata: &ReportMetadata, expected_kind: ReportKind) -> Result<()> {
    if metadata.contract_version() != CONTRACT_VERSION {
        return Err(FerricAlphaError::ReportVersion {
            expected: CONTRACT_VERSION,
            actual: metadata.contract_version().to_string(),
        });
    }
    if metadata.report_kind() != expected_kind {
        return invalid("report kind/body mismatch");
    }
    Ok(())
}

fn validate_tables(tables: &[&ReportTable]) -> Result<()> {
    let mut seen = HashSet::with_capacity(tables.len());
    for table in tables {
        validate_table(table)?;
        if !seen.insert(table.id()) {
            return invalid("duplicate table id");
        }
    }
    Ok(())
}

fn validate_table(table: &ReportTable) -> Result<()> {
    if !valid_table_id(table.id()) {
        return invalid("invalid table id");
    }

    let mut names = HashSet::with_capacity(table.columns().len());
    for column in table.columns() {
        if !names.insert(column.name()) {
            return invalid("duplicate column name");
        }
        if matches!(column.data(), ReportColumnData::Float64(values) if values
            .iter()
            .flatten()
            .any(|value| !value.is_finite()))
        {
            return invalid("float value is not finite");
        }
    }

    let actual_row_count = table
        .columns()
        .first()
        .map_or(0, |column| column.data().len());
    if table.row_count() != actual_row_count {
        return invalid("row count does not match column data");
    }
    if table
        .columns()
        .iter()
        .any(|column| column.data().len() != actual_row_count)
    {
        return invalid("column lengths differ");
    }

    let sort_columns = table
        .sort_by()
        .iter()
        .map(|name| {
            table
                .columns()
                .iter()
                .find(|column| column.name() == name)
                .ok_or_else(|| invalid_error("sort key missing"))
        })
        .collect::<Result<Vec<_>>>()?;

    for row in 0..actual_row_count {
        if sort_columns
            .iter()
            .any(|column| sort_value(column.data(), row).is_none())
        {
            return invalid("sort key contains null");
        }
    }

    for row in 1..actual_row_count {
        if compare_rows(&sort_columns, row - 1, row) == Some(Ordering::Greater) {
            return invalid("rows are not sorted by sort_by");
        }
    }

    Ok(())
}

fn validate_summary_ids(report: &SummaryTearSheetData) -> Result<()> {
    expect_table_id(&report.data.quantile_statistics, "quantile.statistics")?;
    expect_table_id(&report.data.returns_summary, "returns.summary")?;
    expect_table_id(
        &report.data.returns_mean_by_quantile,
        "returns.mean_by_quantile",
    )?;
    expect_table_id(&report.data.information_summary, "information.summary")?;
    expect_table_id(
        &report.data.turnover_mean_by_quantile,
        "turnover.mean_by_quantile",
    )?;
    expect_table_id(
        &report.data.turnover_mean_rank_autocorrelation,
        "turnover.mean_rank_autocorrelation",
    )
}

fn validate_returns_ids(data: &ReturnsReportData) -> Result<()> {
    expect_table_id(&data.summary, "returns.summary")?;
    expect_table_id(&data.mean_by_quantile, "returns.mean_by_quantile")?;
    expect_table_id(&data.daily_by_quantile, "returns.daily_by_quantile")?;
    expect_table_id(&data.daily_spread, "returns.daily_spread")?;
    expect_table_id(&data.factor_returns, "returns.factor_returns")?;
    if let Some(table) = &data.group_mean_by_quantile {
        expect_table_id(table, "returns.group_mean_by_quantile")?;
    }
    if let Some(table) = &data.factor_cumulative_1d {
        expect_table_id(table, "returns.factor_cumulative_1d")?;
    }
    if let Some(table) = &data.quantile_cumulative_1d {
        expect_table_id(table, "returns.quantile_cumulative_1d")?;
    }
    Ok(())
}

fn validate_information_ids(data: &InformationReportData) -> Result<()> {
    expect_table_id(&data.summary, "information.summary")?;
    expect_table_id(&data.ic_by_date, "information.ic_by_date")?;
    expect_table_id(&data.ic_rolling, "information.ic_rolling")?;
    expect_table_id(&data.ic_monthly, "information.ic_monthly")?;
    expect_table_id(&data.qq_normal, "information.qq_normal")?;
    if let Some(table) = &data.ic_by_group {
        expect_table_id(table, "information.ic_by_group")?;
    }
    Ok(())
}

fn validate_turnover_ids(data: &TurnoverReportData) -> Result<()> {
    expect_table_id(&data.mean_by_quantile, "turnover.mean_by_quantile")?;
    expect_table_id(
        &data.mean_rank_autocorrelation,
        "turnover.mean_rank_autocorrelation",
    )?;
    expect_table_id(&data.quantile_series, "turnover.quantile_series")?;
    expect_table_id(
        &data.rank_autocorrelation_series,
        "turnover.rank_autocorrelation_series",
    )
}

fn validate_full_ids(data: &FullReportData) -> Result<()> {
    expect_table_id(&data.quantile_statistics, "quantile.statistics")?;
    validate_returns_ids(&data.returns)?;
    validate_information_ids(&data.information)?;
    validate_turnover_ids(&data.turnover)
}

fn validate_event_returns_ids(data: &EventReturnsReportData) -> Result<()> {
    expect_table_id(
        &data.average_cumulative_returns,
        "events.average_cumulative_returns",
    )?;
    expect_table_id(&data.coverage, "events.coverage")
}

fn validate_event_study_ids(data: &EventStudyReportData) -> Result<()> {
    expect_table_id(&data.quantile_statistics, "quantile.statistics")?;
    expect_table_id(&data.distribution, "events.distribution")?;
    if let Some(event_returns) = &data.event_returns {
        validate_event_returns_ids(event_returns)?;
    }
    expect_table_id(&data.mean_by_quantile, "returns.mean_by_quantile")?;
    expect_table_id(&data.daily_by_quantile, "returns.daily_by_quantile")
}

fn expect_table_id(table: &ReportTable, expected: &'static str) -> Result<()> {
    if table.id() == expected {
        Ok(())
    } else {
        invalid("unexpected table id")
    }
}

fn summary_tables(report: &SummaryTearSheetData) -> Vec<&ReportTable> {
    vec![
        &report.data.quantile_statistics,
        &report.data.returns_summary,
        &report.data.returns_mean_by_quantile,
        &report.data.information_summary,
        &report.data.turnover_mean_by_quantile,
        &report.data.turnover_mean_rank_autocorrelation,
    ]
}

fn returns_tables(data: &ReturnsReportData) -> Vec<&ReportTable> {
    let mut tables = vec![
        &data.summary,
        &data.mean_by_quantile,
        &data.daily_by_quantile,
        &data.daily_spread,
        &data.factor_returns,
    ];
    if let Some(table) = &data.group_mean_by_quantile {
        tables.push(table);
    }
    if let Some(table) = &data.factor_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &data.quantile_cumulative_1d {
        tables.push(table);
    }
    tables
}

fn information_tables(data: &InformationReportData) -> Vec<&ReportTable> {
    let mut tables = vec![
        &data.summary,
        &data.ic_by_date,
        &data.ic_rolling,
        &data.ic_monthly,
        &data.qq_normal,
    ];
    if let Some(table) = &data.ic_by_group {
        tables.push(table);
    }
    tables
}

fn turnover_tables(data: &TurnoverReportData) -> Vec<&ReportTable> {
    vec![
        &data.mean_by_quantile,
        &data.mean_rank_autocorrelation,
        &data.quantile_series,
        &data.rank_autocorrelation_series,
    ]
}

fn full_tables(data: &FullReportData) -> Vec<&ReportTable> {
    let mut tables = vec![
        &data.quantile_statistics,
        &data.returns.summary,
        &data.returns.mean_by_quantile,
        &data.returns.daily_by_quantile,
        &data.returns.daily_spread,
        &data.returns.factor_returns,
        &data.information.summary,
        &data.information.ic_by_date,
        &data.information.ic_rolling,
        &data.information.ic_monthly,
        &data.information.qq_normal,
        &data.turnover.mean_by_quantile,
        &data.turnover.mean_rank_autocorrelation,
        &data.turnover.quantile_series,
        &data.turnover.rank_autocorrelation_series,
    ];
    if let Some(table) = &data.returns.group_mean_by_quantile {
        tables.push(table);
    }
    if let Some(table) = &data.returns.factor_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &data.returns.quantile_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &data.information.ic_by_group {
        tables.push(table);
    }
    tables
}

fn event_returns_tables(data: &EventReturnsReportData) -> Vec<&ReportTable> {
    vec![&data.average_cumulative_returns, &data.coverage]
}

fn event_study_tables(data: &EventStudyReportData) -> Vec<&ReportTable> {
    let mut tables = vec![&data.quantile_statistics, &data.distribution];
    if let Some(event_returns) = &data.event_returns {
        tables.extend(event_returns_tables(event_returns));
    }
    tables.extend([&data.mean_by_quantile, &data.daily_by_quantile]);
    tables
}

fn compare_rows(columns: &[&ReportColumn], left: usize, right: usize) -> Option<Ordering> {
    for column in columns {
        let left = sort_value(column.data(), left)?;
        let right = sort_value(column.data(), right)?;
        let ordering = left.partial_cmp(&right)?;
        if ordering != Ordering::Equal {
            return Some(ordering);
        }
    }
    Some(Ordering::Equal)
}

fn sort_value(data: &ReportColumnData, row: usize) -> Option<SortValue<'_>> {
    match data {
        ReportColumnData::Boolean(values) => {
            values.get(row).copied().flatten().map(SortValue::Boolean)
        }
        ReportColumnData::Int32(values) => values.get(row).copied().flatten().map(SortValue::Int32),
        ReportColumnData::Int64(values) => values.get(row).copied().flatten().map(SortValue::Int64),
        ReportColumnData::UInt32(values) => {
            values.get(row).copied().flatten().map(SortValue::UInt32)
        }
        ReportColumnData::UInt64(values) => {
            values.get(row).copied().flatten().map(SortValue::UInt64)
        }
        ReportColumnData::Float64(values) => {
            values.get(row).copied().flatten().map(SortValue::Float64)
        }
        ReportColumnData::String(values) => values
            .get(row)
            .and_then(Option::as_deref)
            .map(SortValue::String),
        ReportColumnData::Datetime { values, .. } => {
            values.get(row).copied().flatten().map(SortValue::Datetime)
        }
    }
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

fn invalid<T>(reason: &'static str) -> Result<T> {
    Err(invalid_error(reason))
}

fn invalid_error(reason: &'static str) -> FerricAlphaError {
    FerricAlphaError::InvalidReport {
        operation: OPERATION,
        reason,
    }
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
