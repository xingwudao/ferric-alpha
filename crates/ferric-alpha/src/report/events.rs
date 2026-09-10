use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

use crate::performance::event_window_result;
use crate::performance::frame::validate_factor_data;
use crate::performance::{EventReturnsOptions, MeanReturnByQuantileOptions};
use crate::report::context::ReportContext;
use crate::report::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportKind,
    ReportMetadata, ReportTable, ReportTimeUnit, factor_quantile_statistics,
};
use crate::{FerricAlphaError, Result, mean_return_by_quantile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventStudyTearSheetOptions {
    pub event_window: Option<(u32, u32)>,
    pub rate_of_return: bool,
    pub histogram_bins: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventReturnsReportData {
    pub average_cumulative_returns: ReportTable,
    pub coverage: ReportTable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventReturnsTearSheetData {
    pub metadata: ReportMetadata,
    pub options: EventReturnsOptions,
    pub data: EventReturnsReportData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventStudyReportData {
    pub quantile_statistics: ReportTable,
    pub distribution: ReportTable,
    pub event_returns: Option<EventReturnsReportData>,
    pub mean_by_quantile: ReportTable,
    pub daily_by_quantile: ReportTable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventStudyTearSheetData {
    pub metadata: ReportMetadata,
    pub options: EventStudyTearSheetOptions,
    pub data: EventStudyReportData,
}

#[derive(Debug, Clone)]
struct PeriodInfo {
    label: String,
    observations: u32,
}

impl Default for EventStudyTearSheetOptions {
    fn default() -> Self {
        Self {
            event_window: Some((5, 15)),
            rate_of_return: false,
            histogram_bins: 50,
        }
    }
}

pub fn create_event_returns_tear_sheet_data(
    factor_data: &DataFrame,
    returns: &DataFrame,
    options: EventReturnsOptions,
) -> Result<EventReturnsTearSheetData> {
    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::EventReturns)?;
    let result = event_window_result(factor_data, returns, options)?;
    Ok(EventReturnsTearSheetData {
        metadata,
        options,
        data: event_returns_report_data(&result.frame, &result.coverage, options.by_group)?,
    })
}

pub fn create_event_study_tear_sheet_data(
    factor_data: &DataFrame,
    returns: Option<&DataFrame>,
    options: EventStudyTearSheetOptions,
) -> Result<EventStudyTearSheetData> {
    if options.histogram_bins == 0 {
        return Err(FerricAlphaError::InvalidOption {
            option: "histogram_bins",
            reason: "must be positive",
        });
    }

    let context = ReportContext::new(factor_data)?;
    let return_context = validate_factor_data(factor_data, true, false, true)?;
    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::EventStudy)?;
    let date_time_unit = metadata.date_time_unit();
    let date_timezone = metadata.date_timezone().map(ToOwned::to_owned);
    let periods = return_context
        .periods
        .iter()
        .map(|period| PeriodInfo {
            label: period.label.clone(),
            observations: period.observations,
        })
        .collect::<Vec<_>>();
    let base = periods
        .iter()
        .map(|period| period.observations)
        .min()
        .expect("validated returns should contain at least one period");

    let quantile_statistics = factor_quantile_statistics(context.normalized_factor())?;
    let distribution = distribution_table(
        &context,
        options.histogram_bins,
        date_time_unit,
        date_timezone.clone(),
    )?;
    let mean_options = MeanReturnByQuantileOptions::default()
        .with_demeaned(false)
        .with_group_adjust(false);
    let raw_mean_by_quantile = mean_return_by_quantile(factor_data, mean_options)?;
    let raw_daily_by_quantile =
        mean_return_by_quantile(factor_data, mean_options.with_by_date(true))?;
    let mean_by_quantile = mean_quantile_table(
        &raw_mean_by_quantile,
        "returns.mean_by_quantile",
        "Mean Period-Wise Return by Quantile",
        base,
        &periods,
        options.rate_of_return,
        false,
        date_time_unit,
        date_timezone.clone(),
    )?;
    let daily_by_quantile = mean_quantile_table(
        &raw_daily_by_quantile,
        "returns.daily_by_quantile",
        "Daily Return by Quantile",
        base,
        &periods,
        options.rate_of_return,
        true,
        date_time_unit,
        date_timezone,
    )?;
    let event_returns = match (returns, options.event_window) {
        (Some(returns), Some((periods_before, periods_after))) => {
            let event_options = EventReturnsOptions {
                periods_before,
                periods_after,
                demeaned: false,
                group_adjust: false,
                by_group: false,
            };
            let result = event_window_result(factor_data, returns, event_options)?;
            Some(event_returns_report_data(
                &result.frame,
                &result.coverage,
                false,
            )?)
        }
        _ => None,
    };

    Ok(EventStudyTearSheetData {
        metadata,
        options,
        data: EventStudyReportData {
            quantile_statistics,
            distribution,
            event_returns,
            mean_by_quantile,
            daily_by_quantile,
        },
    })
}

fn event_returns_report_data(
    frame: &DataFrame,
    coverage: &[crate::performance::EventCoverageRow],
    by_group: bool,
) -> Result<EventReturnsReportData> {
    Ok(EventReturnsReportData {
        average_cumulative_returns: average_cumulative_returns_table(frame, by_group)?,
        coverage: coverage_table(coverage)?,
    })
}

fn average_cumulative_returns_table(frame: &DataFrame, by_group: bool) -> Result<ReportTable> {
    let quantiles = frame
        .column("factor_quantile")?
        .as_materialized_series()
        .u32()?;
    let groups = frame
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;
    let offsets = frame.column("offset")?.as_materialized_series().i32()?;
    let means = frame
        .column("mean_cumulative_return")?
        .as_materialized_series()
        .f64()?;
    let stds = frame
        .column("std_cumulative_return")?
        .as_materialized_series()
        .f64()?;
    let counts = frame.column("count")?.as_materialized_series().u64()?;

    let mut columns = vec![ReportColumn::new(
        "factor_quantile",
        "Factor Quantile",
        ColumnRole::Dimension,
        Some(DisplayFormat::new(DisplayUnit::Count, 0)),
        ReportColumnData::UInt32(
            (0..frame.height())
                .map(|index| quantiles.get(index))
                .collect(),
        ),
    )];
    let mut sort_by = vec!["factor_quantile".to_string()];
    if by_group {
        columns.push(ReportColumn::new(
            "group",
            "Group",
            ColumnRole::Dimension,
            None,
            ReportColumnData::String(
                (0..frame.height())
                    .map(|index| {
                        groups
                            .as_ref()
                            .and_then(|groups| groups.get(index).map(ToOwned::to_owned))
                    })
                    .collect(),
            ),
        ));
        sort_by.push("group".to_string());
    }
    columns.extend([
        ReportColumn::new(
            "offset",
            "Offset",
            ColumnRole::Dimension,
            Some(DisplayFormat::new(DisplayUnit::Count, 0)),
            ReportColumnData::Int32(
                (0..frame.height())
                    .map(|index| offsets.get(index))
                    .collect(),
            ),
        ),
        ReportColumn::new(
            "mean_cumulative_return",
            "Mean Cumulative Return",
            ColumnRole::Measure,
            Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
            ReportColumnData::Float64((0..frame.height()).map(|index| means.get(index)).collect()),
        ),
        ReportColumn::new(
            "std_cumulative_return",
            "Std Cumulative Return",
            ColumnRole::Statistic,
            Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
            ReportColumnData::Float64((0..frame.height()).map(|index| stds.get(index)).collect()),
        ),
        ReportColumn::new(
            "count",
            "Count",
            ColumnRole::Measure,
            Some(DisplayFormat::new(DisplayUnit::Count, 0)),
            ReportColumnData::UInt64((0..frame.height()).map(|index| counts.get(index)).collect()),
        ),
    ]);
    sort_by.push("offset".to_string());

    ReportTable::new(
        "events.average_cumulative_returns",
        "Average Cumulative Returns by Event Offset",
        sort_by,
        columns,
    )
}

fn coverage_table(rows: &[crate::performance::EventCoverageRow]) -> Result<ReportTable> {
    ReportTable::new(
        "events.coverage",
        "Event Window Coverage",
        vec!["coverage_row".to_string()],
        vec![
            ReportColumn::new(
                "coverage_row",
                "Coverage Row",
                ColumnRole::Dimension,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt32((0..rows.len()).map(|index| Some(index as u32)).collect()),
            ),
            ReportColumn::new(
                "scope",
                "Scope",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(rows.iter().map(|row| Some(row.scope.clone())).collect()),
            ),
            ReportColumn::new(
                "offset",
                "Offset",
                ColumnRole::Dimension,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::Int32(rows.iter().map(|row| row.offset).collect()),
            ),
            count_column(
                "input_events",
                "Input Events",
                rows.iter().map(|row| Some(row.input_events)).collect(),
            ),
            count_column(
                "used_events",
                "Used Events",
                rows.iter().map(|row| Some(row.used_events)).collect(),
            ),
            count_column(
                "skipped_missing_date",
                "Skipped Missing Date",
                rows.iter()
                    .map(|row| Some(row.skipped_missing_date))
                    .collect(),
            ),
            count_column(
                "imputed_return_cells",
                "Imputed Return Cells",
                rows.iter()
                    .map(|row| Some(row.imputed_return_cells))
                    .collect(),
            ),
            count_column(
                "path_count",
                "Path Count",
                rows.iter().map(|row| Some(row.path_count)).collect(),
            ),
        ],
    )
}

fn distribution_table(
    context: &ReportContext,
    requested_bins: u32,
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
) -> Result<ReportTable> {
    let mut dates = context
        .factor_data()
        .rows
        .iter()
        .filter(|row| row.quantile.is_some_and(|quantile| quantile > 0))
        .map(|row| row.date)
        .collect::<Vec<_>>();
    if dates.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: "create_event_study_tear_sheet_data",
        });
    }
    dates.sort_unstable();
    let start = dates[0];
    let end = *dates.last().expect("dates should be non-empty");
    let bin_count = if start == end { 1 } else { requested_bins };
    let span = end
        .checked_sub(start)
        .ok_or(FerricAlphaError::InvalidInput {
            operation: "create_event_study_tear_sheet_data",
            reason: "date range overflow",
        })?;

    let mut bin_starts = Vec::with_capacity(bin_count as usize);
    let mut bin_ends = Vec::with_capacity(bin_count as usize);
    let mut event_counts = Vec::with_capacity(bin_count as usize);
    for index in 0..bin_count {
        let bin_start = start + ((span as i128 * i128::from(index)) / i128::from(bin_count)) as i64;
        let bin_end = if index + 1 == bin_count {
            end
        } else {
            start + ((span as i128 * i128::from(index + 1)) / i128::from(bin_count)) as i64
        };
        let count = dates
            .iter()
            .filter(|date| {
                if index + 1 == bin_count {
                    **date >= bin_start && **date <= bin_end
                } else {
                    **date >= bin_start && **date < bin_end
                }
            })
            .count() as u64;
        bin_starts.push(Some(bin_start));
        bin_ends.push(Some(bin_end));
        event_counts.push(Some(count));
    }

    ReportTable::new(
        "events.distribution",
        "Event Distribution",
        vec!["bin_start".to_string()],
        vec![
            ReportColumn::new(
                "bin_start",
                "Bin Start",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: bin_starts,
                    time_unit: date_time_unit,
                    timezone: date_timezone.clone(),
                },
            ),
            ReportColumn::new(
                "bin_end",
                "Bin End",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: bin_ends,
                    time_unit: date_time_unit,
                    timezone: date_timezone,
                },
            ),
            count_column("event_count", "Event Count", event_counts),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn mean_quantile_table(
    frame: &DataFrame,
    id: &'static str,
    title: &'static str,
    base: u32,
    periods: &[PeriodInfo],
    rate_of_return: bool,
    by_date: bool,
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
) -> Result<ReportTable> {
    let dates = frame
        .column("date")
        .ok()
        .map(|column| column.as_materialized_series().datetime())
        .transpose()?;
    let quantiles = frame
        .column("factor_quantile")?
        .as_materialized_series()
        .u32()?;
    let period_labels = frame.column("period")?.as_materialized_series().str()?;
    let means = frame
        .column("mean_return")?
        .as_materialized_series()
        .f64()?;
    let errors = frame.column("std_error")?.as_materialized_series().f64()?;
    let counts = frame.column("count")?.as_materialized_series().u32()?;

    let mut out_dates = Vec::new();
    let mut out_quantiles = Vec::with_capacity(frame.height());
    let mut out_periods = Vec::with_capacity(frame.height());
    let mut out_means = Vec::with_capacity(frame.height());
    let mut out_errors = Vec::with_capacity(frame.height());
    let mut out_counts = Vec::with_capacity(frame.height());
    for index in 0..frame.height() {
        let period = period_labels.get(index).expect("period is validated");
        if by_date {
            out_dates.push(dates.as_ref().and_then(|dates| dates.physical().get(index)));
        }
        out_quantiles.push(quantiles.get(index));
        out_periods.push(Some(period.to_string()));
        let period_observations = period_observations(periods, period);
        if rate_of_return {
            out_means.push(convert_return(means.get(index), base, period_observations));
            out_errors.push(convert_error(errors.get(index), base, period_observations));
        } else {
            out_means.push(means.get(index));
            out_errors.push(errors.get(index));
        }
        out_counts.push(counts.get(index).map(u64::from));
    }

    let mut columns = Vec::new();
    let mut sort_by = Vec::new();
    if by_date {
        columns.push(ReportColumn::new(
            "date",
            "Date",
            ColumnRole::Dimension,
            None,
            ReportColumnData::Datetime {
                values: out_dates,
                time_unit: date_time_unit,
                timezone: date_timezone,
            },
        ));
        sort_by.push("date".to_string());
    }
    columns.extend([
        ReportColumn::new(
            "factor_quantile",
            "Factor Quantile",
            ColumnRole::Dimension,
            Some(DisplayFormat::new(DisplayUnit::Count, 0)),
            ReportColumnData::UInt32(out_quantiles),
        ),
        ReportColumn::new(
            "period",
            "Period",
            ColumnRole::Dimension,
            None,
            ReportColumnData::String(out_periods),
        ),
        ReportColumn::new(
            "mean_return",
            "Mean Return",
            ColumnRole::Measure,
            Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
            ReportColumnData::Float64(out_means),
        ),
        ReportColumn::new(
            "std_error",
            "Std Error",
            ColumnRole::Statistic,
            Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
            ReportColumnData::Float64(out_errors),
        ),
        ReportColumn::new(
            "count",
            "Count",
            ColumnRole::Measure,
            Some(DisplayFormat::new(DisplayUnit::Count, 0)),
            ReportColumnData::UInt64(out_counts),
        ),
    ]);
    sort_by.extend(["factor_quantile".to_string(), "period".to_string()]);
    ReportTable::new(id, title, sort_by, columns)
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

fn period_observations(periods: &[PeriodInfo], label: &str) -> u32 {
    periods
        .iter()
        .find(|period| period.label == label)
        .map(|period| period.observations)
        .expect("period should come from factor data")
}

fn convert_return(value: Option<f64>, base: u32, period: u32) -> Option<f64> {
    let value = value?;
    if !value.is_finite() {
        return None;
    }
    let gross = 1.0 + value;
    let exponent = base as f64 / period as f64;
    if gross < 0.0 && exponent.fract() != 0.0 {
        return None;
    }
    let converted = gross.powf(exponent) - 1.0;
    converted.is_finite().then_some(converted)
}

fn convert_error(value: Option<f64>, base: u32, period: u32) -> Option<f64> {
    let value = value?;
    if !value.is_finite() {
        return None;
    }
    let converted = value / (period as f64 / base as f64).sqrt();
    converted.is_finite().then_some(converted)
}
