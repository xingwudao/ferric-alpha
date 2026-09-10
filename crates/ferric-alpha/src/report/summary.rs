use std::collections::HashSet;

use polars::prelude::DataFrame;
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::calendar::Period;
use crate::report::{
    InformationReportData, InformationTearSheetOptions, ReportKind, ReportMetadata, ReportTable,
    ReturnsReportData, ReturnsTearSheetOptions, TurnoverReportData, TurnoverTearSheetOptions,
    create_information_tear_sheet_data, create_returns_tear_sheet_data,
    create_turnover_tear_sheet_data, factor_quantile_statistics,
};
use crate::{FerricAlphaError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryTearSheetOptions {
    pub long_short: bool,
    pub group_neutral: bool,
    pub turnover_periods: Option<Vec<Period>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullTearSheetOptions {
    pub long_short: bool,
    pub group_neutral: bool,
    pub by_group: bool,
    pub turnover_periods: Option<Vec<Period>>,
    pub ic_rolling_window: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryReportData {
    pub quantile_statistics: ReportTable,
    pub returns_summary: ReportTable,
    pub returns_mean_by_quantile: ReportTable,
    pub information_summary: ReportTable,
    pub turnover_mean_by_quantile: ReportTable,
    pub turnover_mean_rank_autocorrelation: ReportTable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryTearSheetData {
    pub metadata: ReportMetadata,
    pub options: SummaryTearSheetOptions,
    pub data: SummaryReportData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullReportData {
    pub quantile_statistics: ReportTable,
    pub returns: ReturnsReportData,
    pub information: InformationReportData,
    pub turnover: TurnoverReportData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullTearSheetData {
    pub metadata: ReportMetadata,
    pub options: FullTearSheetOptions,
    pub data: FullReportData,
}

impl Default for FullTearSheetOptions {
    fn default() -> Self {
        Self {
            long_short: true,
            group_neutral: false,
            by_group: false,
            turnover_periods: None,
            ic_rolling_window: 22,
        }
    }
}

impl Default for SummaryTearSheetOptions {
    fn default() -> Self {
        Self {
            long_short: true,
            group_neutral: false,
            turnover_periods: None,
        }
    }
}

pub fn create_summary_tear_sheet_data(
    factor_data: &DataFrame,
    options: SummaryTearSheetOptions,
) -> Result<SummaryTearSheetData> {
    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::Summary)?;
    let quantile_statistics = factor_quantile_statistics(factor_data)?;
    let returns = create_returns_tear_sheet_data(factor_data, options.returns_options())?;
    let information =
        create_information_tear_sheet_data(factor_data, options.information_options())?;
    let turnover = create_turnover_tear_sheet_data(factor_data, options.turnover_options())?;

    Ok(SummaryTearSheetData {
        metadata,
        options,
        data: SummaryReportData {
            quantile_statistics,
            returns_summary: returns.data.summary,
            returns_mean_by_quantile: returns.data.mean_by_quantile,
            information_summary: information.data.summary,
            turnover_mean_by_quantile: turnover.data.mean_by_quantile,
            turnover_mean_rank_autocorrelation: turnover.data.mean_rank_autocorrelation,
        },
    })
}

pub fn create_full_tear_sheet_data(
    factor_data: &DataFrame,
    options: FullTearSheetOptions,
) -> Result<FullTearSheetData> {
    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::Full)?;
    let quantile_statistics = factor_quantile_statistics(factor_data)?;
    let returns = create_returns_tear_sheet_data(factor_data, options.returns_options())?;
    let information =
        create_information_tear_sheet_data(factor_data, options.information_options())?;
    let turnover = create_turnover_tear_sheet_data(factor_data, options.turnover_options())?;
    validate_unique_full_ids(
        &quantile_statistics,
        &returns.data,
        &information.data,
        &turnover.data,
    )?;

    Ok(FullTearSheetData {
        metadata,
        options,
        data: FullReportData {
            quantile_statistics,
            returns: returns.data,
            information: information.data,
            turnover: turnover.data,
        },
    })
}

impl SummaryTearSheetOptions {
    fn returns_options(&self) -> ReturnsTearSheetOptions {
        ReturnsTearSheetOptions {
            long_short: self.long_short,
            group_neutral: self.group_neutral,
            by_group: false,
        }
    }

    fn information_options(&self) -> InformationTearSheetOptions {
        InformationTearSheetOptions {
            group_neutral: self.group_neutral,
            by_group: false,
            rolling_window: InformationTearSheetOptions::default().rolling_window,
        }
    }

    fn turnover_options(&self) -> TurnoverTearSheetOptions {
        TurnoverTearSheetOptions {
            periods: self.turnover_periods.clone(),
        }
    }
}

impl FullTearSheetOptions {
    fn returns_options(&self) -> ReturnsTearSheetOptions {
        ReturnsTearSheetOptions {
            long_short: self.long_short,
            group_neutral: self.group_neutral,
            by_group: self.by_group,
        }
    }

    fn information_options(&self) -> InformationTearSheetOptions {
        InformationTearSheetOptions {
            group_neutral: self.group_neutral,
            by_group: self.by_group,
            rolling_window: self.ic_rolling_window,
        }
    }

    fn turnover_options(&self) -> TurnoverTearSheetOptions {
        TurnoverTearSheetOptions {
            periods: self.turnover_periods.clone(),
        }
    }
}

fn validate_unique_full_ids(
    quantile_statistics: &ReportTable,
    returns: &ReturnsReportData,
    information: &InformationReportData,
    turnover: &TurnoverReportData,
) -> Result<()> {
    let mut seen = HashSet::new();
    for table in full_table_refs(quantile_statistics, returns, information, turnover) {
        if !seen.insert(table.id()) {
            return Err(FerricAlphaError::InvalidReport {
                operation: "create_full_tear_sheet_data",
                reason: "duplicate table id",
            });
        }
    }
    Ok(())
}

fn full_table_refs<'a>(
    quantile_statistics: &'a ReportTable,
    returns: &'a ReturnsReportData,
    information: &'a InformationReportData,
    turnover: &'a TurnoverReportData,
) -> Vec<&'a ReportTable> {
    let mut tables = vec![
        quantile_statistics,
        &returns.summary,
        &returns.mean_by_quantile,
        &returns.daily_by_quantile,
        &returns.daily_spread,
        &returns.factor_returns,
        &information.summary,
        &information.ic_by_date,
        &information.ic_rolling,
        &information.ic_monthly,
        &information.qq_normal,
        &turnover.mean_by_quantile,
        &turnover.mean_rank_autocorrelation,
        &turnover.quantile_series,
        &turnover.rank_autocorrelation_series,
    ];
    if let Some(table) = &returns.group_mean_by_quantile {
        tables.push(table);
    }
    if let Some(table) = &returns.factor_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &returns.quantile_cumulative_1d {
        tables.push(table);
    }
    if let Some(table) = &information.ic_by_group {
        tables.push(table);
    }
    tables
}

impl Serialize for SummaryTearSheetOptions {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SummaryTearSheetOptions", 3)?;
        state.serialize_field("long_short", &self.long_short)?;
        state.serialize_field("group_neutral", &self.group_neutral)?;
        state.serialize_field(
            "turnover_periods",
            &serialize_periods(&self.turnover_periods),
        )?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SummaryTearSheetOptions {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawOptions {
            long_short: bool,
            group_neutral: bool,
            turnover_periods: Option<Vec<String>>,
        }

        let raw = RawOptions::deserialize(deserializer)?;
        Ok(Self {
            long_short: raw.long_short,
            group_neutral: raw.group_neutral,
            turnover_periods: parse_optional_periods(raw.turnover_periods)
                .map_err(de::Error::custom)?,
        })
    }
}

impl Serialize for FullTearSheetOptions {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FullTearSheetOptions", 5)?;
        state.serialize_field("long_short", &self.long_short)?;
        state.serialize_field("group_neutral", &self.group_neutral)?;
        state.serialize_field("by_group", &self.by_group)?;
        state.serialize_field(
            "turnover_periods",
            &serialize_periods(&self.turnover_periods),
        )?;
        state.serialize_field("ic_rolling_window", &self.ic_rolling_window)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for FullTearSheetOptions {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawOptions {
            long_short: bool,
            group_neutral: bool,
            by_group: bool,
            turnover_periods: Option<Vec<String>>,
            ic_rolling_window: u32,
        }

        let raw = RawOptions::deserialize(deserializer)?;
        Ok(Self {
            long_short: raw.long_short,
            group_neutral: raw.group_neutral,
            by_group: raw.by_group,
            turnover_periods: parse_optional_periods(raw.turnover_periods)
                .map_err(de::Error::custom)?,
            ic_rolling_window: raw.ic_rolling_window,
        })
    }
}

fn serialize_periods(periods: &Option<Vec<Period>>) -> Option<Vec<String>> {
    periods
        .as_ref()
        .map(|periods| periods.iter().map(|period| period.label()).collect())
}

fn parse_optional_periods(
    labels: Option<Vec<String>>,
) -> std::result::Result<Option<Vec<Period>>, String> {
    labels
        .map(|labels| {
            labels
                .into_iter()
                .map(|label| parse_period_label(&label))
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .transpose()
}

fn parse_period_label(label: &str) -> std::result::Result<Period, String> {
    let Some(count) = label.strip_suffix('D') else {
        return Err("period must end with D".to_string());
    };
    if count.is_empty() || count.starts_with('0') {
        return Err("period must be a positive daily count".to_string());
    }
    let count = count
        .parse::<u32>()
        .map_err(|_| "period must be numeric".to_string())?;
    Period::new(count).map_err(|error| error.to_string())
}
