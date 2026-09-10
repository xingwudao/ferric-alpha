use std::collections::BTreeMap;

use polars::prelude::DataFrame;
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::calendar::{Period, validate_periods};
use crate::performance::frame::validate_factor_data;
use crate::report::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportKind,
    ReportMetadata, ReportTable, ReportTimeUnit,
};
use crate::{FerricAlphaError, Result, factor_rank_autocorrelation, quantile_turnover};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnoverTearSheetOptions {
    pub periods: Option<Vec<Period>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnoverReportData {
    pub mean_by_quantile: ReportTable,
    pub mean_rank_autocorrelation: ReportTable,
    pub quantile_series: ReportTable,
    pub rank_autocorrelation_series: ReportTable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnoverTearSheetData {
    pub metadata: ReportMetadata,
    pub options: TurnoverTearSheetOptions,
    pub data: TurnoverReportData,
}

#[derive(Debug, Clone)]
struct QuantileTurnoverRow {
    date: i64,
    factor_quantile: u32,
    period: String,
    turnover: Option<f64>,
}

#[derive(Debug, Clone)]
struct RankAutocorrelationRow {
    date: i64,
    period: String,
    autocorrelation: Option<f64>,
}

pub fn create_turnover_tear_sheet_data(
    factor_data: &DataFrame,
    options: TurnoverTearSheetOptions,
) -> Result<TurnoverTearSheetData> {
    let explicit_periods = options
        .periods
        .as_ref()
        .map(|periods| validate_periods(periods))
        .transpose()?;
    let context = validate_factor_data(factor_data, true, false, options.periods.is_none())?;
    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::Turnover)?;
    let periods = match explicit_periods {
        Some(periods) => periods,
        None => {
            let discovered = context
                .periods
                .iter()
                .map(|period| Period::new(period.observations))
                .collect::<Result<Vec<_>>>()?;
            validate_periods(&discovered)?
        }
    };
    let quantiles = context
        .rows
        .iter()
        .filter_map(|row| row.quantile.filter(|quantile| *quantile > 0))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if quantiles.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: "create_turnover_tear_sheet_data",
        });
    }

    let date_time_unit = metadata.date_time_unit();
    let date_timezone = metadata.date_timezone().map(ToOwned::to_owned);
    let quantile_rows = collect_quantile_turnover_rows(factor_data, &quantiles, &periods)?;
    let rank_rows = collect_rank_autocorrelation_rows(factor_data, &periods)?;

    let mean_by_quantile = mean_by_quantile_table(&quantile_rows, &quantiles, &periods)?;
    let mean_rank_autocorrelation = mean_rank_autocorrelation_table(&rank_rows, &periods)?;
    let quantile_series =
        quantile_series_table(&quantile_rows, date_time_unit, date_timezone.clone())?;
    let rank_autocorrelation_series =
        rank_autocorrelation_series_table(&rank_rows, date_time_unit, date_timezone)?;

    Ok(TurnoverTearSheetData {
        metadata,
        options,
        data: TurnoverReportData {
            mean_by_quantile,
            mean_rank_autocorrelation,
            quantile_series,
            rank_autocorrelation_series,
        },
    })
}

fn collect_quantile_turnover_rows(
    factor_data: &DataFrame,
    quantiles: &[u32],
    periods: &[Period],
) -> Result<Vec<QuantileTurnoverRow>> {
    let mut rows = Vec::new();
    for quantile in quantiles {
        for period in periods {
            let frame = quantile_turnover(factor_data, *quantile, *period)?;
            let dates = frame.column("date")?.as_materialized_series().datetime()?;
            let frame_quantiles = frame
                .column("factor_quantile")?
                .as_materialized_series()
                .u32()?;
            let frame_periods = frame.column("period")?.as_materialized_series().str()?;
            let turnover = frame.column("turnover")?.as_materialized_series().f64()?;
            for index in 0..frame.height() {
                rows.push(QuantileTurnoverRow {
                    date: dates
                        .physical()
                        .get(index)
                        .expect("generated turnover date should not be null"),
                    factor_quantile: frame_quantiles
                        .get(index)
                        .expect("generated turnover quantile should not be null"),
                    period: frame_periods
                        .get(index)
                        .expect("generated turnover period should not be null")
                        .to_string(),
                    turnover: turnover.get(index).filter(|value| value.is_finite()),
                });
            }
        }
    }
    rows.sort_by(|left, right| {
        left.date
            .cmp(&right.date)
            .then(left.factor_quantile.cmp(&right.factor_quantile))
            .then(left.period.cmp(&right.period))
    });
    Ok(rows)
}

fn collect_rank_autocorrelation_rows(
    factor_data: &DataFrame,
    periods: &[Period],
) -> Result<Vec<RankAutocorrelationRow>> {
    let mut rows = Vec::new();
    for period in periods {
        let frame = factor_rank_autocorrelation(factor_data, *period)?;
        let dates = frame.column("date")?.as_materialized_series().datetime()?;
        let frame_periods = frame.column("period")?.as_materialized_series().str()?;
        let autocorrelation = frame
            .column("autocorrelation")?
            .as_materialized_series()
            .f64()?;
        for index in 0..frame.height() {
            rows.push(RankAutocorrelationRow {
                date: dates
                    .physical()
                    .get(index)
                    .expect("generated autocorrelation date should not be null"),
                period: frame_periods
                    .get(index)
                    .expect("generated autocorrelation period should not be null")
                    .to_string(),
                autocorrelation: autocorrelation.get(index).filter(|value| value.is_finite()),
            });
        }
    }
    rows.sort_by(|left, right| {
        left.date
            .cmp(&right.date)
            .then(left.period.cmp(&right.period))
    });
    Ok(rows)
}

fn mean_by_quantile_table(
    rows: &[QuantileTurnoverRow],
    quantiles: &[u32],
    periods: &[Period],
) -> Result<ReportTable> {
    let mut grouped = BTreeMap::<(u32, String), Vec<f64>>::new();
    for row in rows {
        if let Some(value) = row.turnover {
            grouped
                .entry((row.factor_quantile, row.period.clone()))
                .or_default()
                .push(value);
        }
    }

    let mut out = Vec::<(u32, String, Option<f64>, u64)>::new();
    for quantile in quantiles {
        for period in periods {
            let label = period.label();
            let values = grouped
                .get(&(*quantile, label.clone()))
                .cloned()
                .unwrap_or_default();
            let count = values.len() as u64;
            let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / count as f64);
            out.push((*quantile, label, mean, count));
        }
    }
    out.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    ReportTable::new(
        "turnover.mean_by_quantile",
        "Mean Quantile Turnover",
        vec!["factor_quantile".to_string(), "period".to_string()],
        vec![
            ReportColumn::new(
                "factor_quantile",
                "Factor Quantile",
                ColumnRole::Dimension,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt32(out.iter().map(|row| Some(row.0)).collect()),
            ),
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                out.iter().map(|row| Some(row.1.clone())).collect(),
            ),
            ratio_column(
                "turnover",
                "Turnover",
                ColumnRole::Measure,
                out.iter().map(|row| row.2).collect(),
            ),
            count_column(
                "count",
                "Count",
                out.iter().map(|row| Some(row.3)).collect(),
            ),
        ],
    )
}

fn mean_rank_autocorrelation_table(
    rows: &[RankAutocorrelationRow],
    periods: &[Period],
) -> Result<ReportTable> {
    let mut grouped = BTreeMap::<String, Vec<f64>>::new();
    for row in rows {
        if let Some(value) = row.autocorrelation {
            grouped.entry(row.period.clone()).or_default().push(value);
        }
    }

    let mut out = Vec::<(String, Option<f64>, u64)>::new();
    for period in periods {
        let label = period.label();
        let values = grouped.get(&label).cloned().unwrap_or_default();
        let count = values.len() as u64;
        let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / count as f64);
        out.push((label, mean, count));
    }
    out.sort_by(|left, right| left.0.cmp(&right.0));

    ReportTable::new(
        "turnover.mean_rank_autocorrelation",
        "Mean Factor Rank Autocorrelation",
        vec!["period".to_string()],
        vec![
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                out.iter().map(|row| Some(row.0.clone())).collect(),
            ),
            ratio_column(
                "autocorrelation",
                "Autocorrelation",
                ColumnRole::Measure,
                out.iter().map(|row| row.1).collect(),
            ),
            count_column(
                "count",
                "Count",
                out.iter().map(|row| Some(row.2)).collect(),
            ),
        ],
    )
}

fn quantile_series_table(
    rows: &[QuantileTurnoverRow],
    time_unit: ReportTimeUnit,
    timezone: Option<String>,
) -> Result<ReportTable> {
    ReportTable::new(
        "turnover.quantile_series",
        "Quantile Turnover",
        vec![
            "date".to_string(),
            "factor_quantile".to_string(),
            "period".to_string(),
        ],
        vec![
            datetime_column(
                "date",
                "Date",
                rows.iter().map(|row| Some(row.date)).collect(),
                time_unit,
                timezone,
            ),
            ReportColumn::new(
                "factor_quantile",
                "Factor Quantile",
                ColumnRole::Dimension,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt32(
                    rows.iter().map(|row| Some(row.factor_quantile)).collect(),
                ),
            ),
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                rows.iter().map(|row| Some(row.period.clone())).collect(),
            ),
            ratio_column(
                "turnover",
                "Turnover",
                ColumnRole::Measure,
                rows.iter().map(|row| row.turnover).collect(),
            ),
        ],
    )
}

fn rank_autocorrelation_series_table(
    rows: &[RankAutocorrelationRow],
    time_unit: ReportTimeUnit,
    timezone: Option<String>,
) -> Result<ReportTable> {
    ReportTable::new(
        "turnover.rank_autocorrelation_series",
        "Factor Rank Autocorrelation",
        vec!["date".to_string(), "period".to_string()],
        vec![
            datetime_column(
                "date",
                "Date",
                rows.iter().map(|row| Some(row.date)).collect(),
                time_unit,
                timezone,
            ),
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                rows.iter().map(|row| Some(row.period.clone())).collect(),
            ),
            ratio_column(
                "autocorrelation",
                "Autocorrelation",
                ColumnRole::Measure,
                rows.iter().map(|row| row.autocorrelation).collect(),
            ),
        ],
    )
}

fn string_column(
    name: &'static str,
    label: &'static str,
    role: ColumnRole,
    values: Vec<Option<String>>,
) -> ReportColumn {
    ReportColumn::new(name, label, role, None, ReportColumnData::String(values))
}

fn ratio_column(
    name: &'static str,
    label: &'static str,
    role: ColumnRole,
    values: Vec<Option<f64>>,
) -> ReportColumn {
    ReportColumn::new(
        name,
        label,
        role,
        Some(DisplayFormat::new(DisplayUnit::Ratio, 6)),
        ReportColumnData::Float64(values),
    )
}

fn count_column(name: &'static str, label: &'static str, values: Vec<Option<u64>>) -> ReportColumn {
    ReportColumn::new(
        name,
        label,
        ColumnRole::Measure,
        Some(DisplayFormat::new(DisplayUnit::Count, 0)),
        ReportColumnData::UInt64(values),
    )
}

fn datetime_column(
    name: &'static str,
    label: &'static str,
    values: Vec<Option<i64>>,
    time_unit: ReportTimeUnit,
    timezone: Option<String>,
) -> ReportColumn {
    ReportColumn::new(
        name,
        label,
        ColumnRole::Dimension,
        None,
        ReportColumnData::Datetime {
            values,
            time_unit,
            timezone,
        },
    )
}

impl Serialize for TurnoverTearSheetOptions {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("TurnoverTearSheetOptions", 1)?;
        let labels = self.periods.as_ref().map(|periods| {
            periods
                .iter()
                .map(|period| period.label())
                .collect::<Vec<_>>()
        });
        state.serialize_field("periods", &labels)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TurnoverTearSheetOptions {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawOptions {
            periods: Option<Vec<String>>,
        }

        let raw = RawOptions::deserialize(deserializer)?;
        let periods = raw
            .periods
            .map(|labels| {
                labels
                    .into_iter()
                    .map(|label| parse_period_label(&label).map_err(de::Error::custom))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .transpose()?;
        Ok(Self { periods })
    }
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
