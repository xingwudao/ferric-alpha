use std::collections::BTreeSet;

use polars::prelude::{DataFrame, DataType, TimeUnit};
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::performance::frame::validate_factor_data;
use crate::report::ReportTimeUnit;

pub const CONTRACT_VERSION: &str = "ferric-alpha.tear-sheet/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportKind {
    Summary,
    Returns,
    Information,
    Turnover,
    Full,
    EventReturns,
    EventStudy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportMetadata {
    contract_version: String,
    report_kind: ReportKind,
    source_rows: u64,
    date_start: Option<i64>,
    date_end: Option<i64>,
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
    periods: Vec<String>,
    quantiles: Vec<u32>,
    groups: Vec<String>,
}

impl ReportMetadata {
    pub fn from_factor_data(frame: &DataFrame, report_kind: ReportKind) -> Result<Self> {
        let data = validate_factor_data(frame, false, false, false)?;
        let mut quantiles = BTreeSet::new();
        let mut groups = BTreeSet::new();
        let mut date_start = None;
        let mut date_end = None;

        for row in &data.rows {
            date_start = Some(date_start.map_or(row.date, |value: i64| value.min(row.date)));
            date_end = Some(date_end.map_or(row.date, |value: i64| value.max(row.date)));
            if let Some(quantile) = row.quantile
                && quantile > 0
            {
                quantiles.insert(quantile);
            }
            if let Some(group) = &row.group {
                groups.insert(group.clone());
            }
        }

        let (date_time_unit, date_timezone) = date_metadata(&data.date_dtype)?;

        Ok(Self {
            contract_version: CONTRACT_VERSION.to_string(),
            report_kind,
            source_rows: data.rows.len() as u64,
            date_start,
            date_end,
            date_time_unit,
            date_timezone,
            periods: data
                .periods
                .iter()
                .map(|period| period.label.clone())
                .collect(),
            quantiles: quantiles.into_iter().collect(),
            groups: groups.into_iter().collect(),
        })
    }

    pub fn contract_version(&self) -> &str {
        &self.contract_version
    }

    pub fn report_kind(&self) -> ReportKind {
        self.report_kind
    }

    pub fn source_rows(&self) -> u64 {
        self.source_rows
    }

    pub fn date_start(&self) -> Option<i64> {
        self.date_start
    }

    pub fn date_end(&self) -> Option<i64> {
        self.date_end
    }

    pub fn date_time_unit(&self) -> ReportTimeUnit {
        self.date_time_unit
    }

    pub fn date_timezone(&self) -> Option<&str> {
        self.date_timezone.as_deref()
    }

    pub fn periods(&self) -> &[String] {
        &self.periods
    }

    pub fn quantiles(&self) -> &[u32] {
        &self.quantiles
    }

    pub fn groups(&self) -> &[String] {
        &self.groups
    }
}

fn date_metadata(dtype: &DataType) -> Result<(ReportTimeUnit, Option<String>)> {
    match dtype {
        DataType::Datetime(unit, timezone) => {
            Ok(((*unit).into(), timezone.as_ref().map(ToString::to_string)))
        }
        actual => Err(crate::FerricAlphaError::InvalidDtype {
            column: "date",
            expected: "Datetime",
            actual: actual.to_string(),
        }),
    }
}

impl From<TimeUnit> for ReportTimeUnit {
    fn from(value: TimeUnit) -> Self {
        match value {
            TimeUnit::Nanoseconds => Self::Nanoseconds,
            TimeUnit::Microseconds => Self::Microseconds,
            TimeUnit::Milliseconds => Self::Milliseconds,
        }
    }
}
