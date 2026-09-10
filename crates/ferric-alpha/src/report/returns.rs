use std::collections::BTreeMap;

use polars::prelude::{BooleanChunked, DataFrame, NewChunkedArray};
use serde::{Deserialize, Serialize};

use crate::performance::frame::validate_factor_data;
use crate::performance::{AlphaBetaOptions, MeanReturnByQuantileOptions, ReturnOptions};
use crate::report::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportKind,
    ReportMetadata, ReportTable, ReportTimeUnit,
};
use crate::{
    Result, WeightOptions, compute_mean_returns_spread, cumulative_returns, factor_alpha_beta,
    factor_returns, mean_return_by_quantile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReturnsTearSheetOptions {
    pub long_short: bool,
    pub group_neutral: bool,
    pub by_group: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnsReportData {
    pub summary: ReportTable,
    pub mean_by_quantile: ReportTable,
    pub daily_by_quantile: ReportTable,
    pub daily_spread: ReportTable,
    pub factor_returns: ReportTable,
    pub group_mean_by_quantile: Option<ReportTable>,
    pub factor_cumulative_1d: Option<ReportTable>,
    pub quantile_cumulative_1d: Option<ReportTable>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnsTearSheetData {
    pub metadata: ReportMetadata,
    pub options: ReturnsTearSheetOptions,
    pub data: ReturnsReportData,
}

#[derive(Debug, Clone)]
struct PeriodInfo {
    label: String,
    observations: u32,
}

impl Default for ReturnsTearSheetOptions {
    fn default() -> Self {
        Self {
            long_short: true,
            group_neutral: false,
            by_group: false,
        }
    }
}

pub fn create_returns_tear_sheet_data(
    factor_data: &DataFrame,
    options: ReturnsTearSheetOptions,
) -> Result<ReturnsTearSheetData> {
    let context = validate_factor_data(
        factor_data,
        true,
        options.group_neutral || options.by_group,
        true,
    )?;
    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::Returns)?;
    let periods = context
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
    let date_time_unit = metadata.date_time_unit();
    let date_timezone = metadata.date_timezone().map(ToOwned::to_owned);
    let quantiles = metadata.quantiles().to_vec();
    let (lower_quantile, upper_quantile) = match (quantiles.first(), quantiles.last()) {
        (Some(lower), Some(upper)) => (*lower, *upper),
        _ => {
            return Err(crate::FerricAlphaError::InsufficientData {
                operation: "create_returns_tear_sheet_data",
            });
        }
    };
    let weight_options = WeightOptions::default()
        .with_demeaned(options.long_short)
        .with_group_adjust(options.group_neutral);
    let mean_options = MeanReturnByQuantileOptions::default()
        .with_demeaned(options.long_short)
        .with_group_adjust(options.group_neutral);

    let raw_mean_by_quantile = mean_return_by_quantile(factor_data, mean_options)?;
    let raw_daily_by_quantile =
        mean_return_by_quantile(factor_data, mean_options.with_by_date(true))?;
    let raw_daily_spread = compute_mean_returns_spread(
        &raw_daily_by_quantile,
        upper_quantile,
        lower_quantile,
        Some(&raw_daily_by_quantile),
    )?;
    let raw_factor_returns = factor_returns(
        factor_data,
        ReturnOptions::default().with_weight_options(weight_options),
    )?;
    let alpha_beta = factor_alpha_beta(
        factor_data,
        Some(&raw_factor_returns),
        AlphaBetaOptions::default().with_weight_options(weight_options),
    )?;

    let group_mean_by_quantile = options
        .by_group
        .then(|| {
            mean_return_by_quantile(factor_data, mean_options.with_by_group(true)).and_then(
                |frame| {
                    mean_quantile_table(
                        &frame,
                        "returns.group_mean_by_quantile",
                        "Mean Return by Quantile and Group",
                        base,
                        &periods,
                        false,
                        true,
                        date_time_unit,
                        date_timezone.clone(),
                    )
                },
            )
        })
        .transpose()?;

    let summary = summary_table(
        &alpha_beta,
        &raw_mean_by_quantile,
        &raw_daily_spread,
        upper_quantile,
        lower_quantile,
        base,
        &periods,
    )?;
    let mean_by_quantile = mean_quantile_table(
        &raw_mean_by_quantile,
        "returns.mean_by_quantile",
        "Mean Period-Wise Return by Quantile",
        base,
        &periods,
        false,
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
        true,
        false,
        date_time_unit,
        date_timezone.clone(),
    )?;
    let daily_spread = spread_table(
        &raw_daily_spread,
        base,
        &periods,
        date_time_unit,
        date_timezone.clone(),
    )?;
    let factor_returns =
        factor_returns_table(&raw_factor_returns, date_time_unit, date_timezone.clone())?;
    let factor_cumulative_1d = if periods.iter().any(|period| period.label == "1D") {
        Some(factor_cumulative_table(
            &raw_factor_returns,
            date_time_unit,
            date_timezone.clone(),
        )?)
    } else {
        None
    };
    let quantile_cumulative_1d = if periods.iter().any(|period| period.label == "1D") {
        Some(quantile_cumulative_table(
            &raw_daily_by_quantile,
            date_time_unit,
            date_timezone,
        )?)
    } else {
        None
    };

    Ok(ReturnsTearSheetData {
        metadata,
        options,
        data: ReturnsReportData {
            summary,
            mean_by_quantile,
            daily_by_quantile,
            daily_spread,
            factor_returns,
            group_mean_by_quantile,
            factor_cumulative_1d,
            quantile_cumulative_1d,
        },
    })
}

fn summary_table(
    alpha_beta: &DataFrame,
    mean_by_quantile: &DataFrame,
    daily_spread: &DataFrame,
    upper_quantile: u32,
    lower_quantile: u32,
    base: u32,
    periods: &[PeriodInfo],
) -> Result<ReportTable> {
    let mut rows = Vec::<(String, String, Option<f64>, Option<u64>)>::new();
    let period_labels = alpha_beta
        .column("period")?
        .as_materialized_series()
        .str()?;
    let alpha = alpha_beta.column("alpha")?.as_materialized_series().f64()?;
    let annualized = alpha_beta
        .column("annualized_alpha")?
        .as_materialized_series()
        .f64()?;
    let beta = alpha_beta.column("beta")?.as_materialized_series().f64()?;
    let alpha_count = alpha_beta.column("count")?.as_materialized_series().u32()?;
    for index in 0..alpha_beta.height() {
        let period = period_labels.get(index).expect("period is validated");
        rows.push((
            "alpha_raw".to_string(),
            period.to_string(),
            alpha.get(index),
            alpha_count.get(index).map(u64::from),
        ));
        rows.push((
            "alpha_annualized".to_string(),
            period.to_string(),
            annualized.get(index),
            alpha_count.get(index).map(u64::from),
        ));
        rows.push((
            "beta".to_string(),
            period.to_string(),
            beta.get(index),
            alpha_count.get(index).map(u64::from),
        ));
    }

    collect_quantile_summary(
        mean_by_quantile,
        upper_quantile,
        "top_quantile_mean",
        base,
        periods,
        &mut rows,
    )?;
    collect_quantile_summary(
        mean_by_quantile,
        lower_quantile,
        "bottom_quantile_mean",
        base,
        periods,
        &mut rows,
    )?;
    collect_spread_summary(daily_spread, base, periods, &mut rows)?;
    rows.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    ReportTable::new(
        "returns.summary",
        "Returns Analysis",
        vec!["metric".to_string(), "period".to_string()],
        vec![
            ReportColumn::new(
                "metric",
                "Metric",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(rows.iter().map(|row| Some(row.0.clone())).collect()),
            ),
            ReportColumn::new(
                "period",
                "Period",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(rows.iter().map(|row| Some(row.1.clone())).collect()),
            ),
            ReportColumn::new(
                "value",
                "Value",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
                ReportColumnData::Float64(rows.iter().map(|row| row.2).collect()),
            ),
            ReportColumn::new(
                "count",
                "Count",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt64(rows.iter().map(|row| row.3).collect()),
            ),
        ],
    )
}

fn collect_quantile_summary(
    frame: &DataFrame,
    target_quantile: u32,
    metric: &str,
    base: u32,
    periods: &[PeriodInfo],
    rows: &mut Vec<(String, String, Option<f64>, Option<u64>)>,
) -> Result<()> {
    let quantiles = frame
        .column("factor_quantile")?
        .as_materialized_series()
        .u32()?;
    let period_labels = frame.column("period")?.as_materialized_series().str()?;
    let means = frame
        .column("mean_return")?
        .as_materialized_series()
        .f64()?;
    let counts = frame.column("count")?.as_materialized_series().u32()?;
    for index in 0..frame.height() {
        if quantiles.get(index) != Some(target_quantile) {
            continue;
        }
        let period = period_labels.get(index).expect("period is validated");
        rows.push((
            metric.to_string(),
            period.to_string(),
            convert_return(means.get(index), base, period_observations(periods, period)),
            counts.get(index).map(u64::from),
        ));
    }
    Ok(())
}

fn collect_spread_summary(
    frame: &DataFrame,
    base: u32,
    periods: &[PeriodInfo],
    rows: &mut Vec<(String, String, Option<f64>, Option<u64>)>,
) -> Result<()> {
    let period_labels = frame.column("period")?.as_materialized_series().str()?;
    let spreads = frame
        .column("mean_return_difference")?
        .as_materialized_series()
        .f64()?;
    for period in periods {
        let mut values = Vec::new();
        for index in 0..frame.height() {
            if period_labels.get(index) == Some(period.label.as_str())
                && let Some(value) = spreads.get(index).filter(|value| value.is_finite())
            {
                values.push(value);
            }
        }
        let mean = if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        };
        rows.push((
            "mean_spread".to_string(),
            period.label.clone(),
            convert_return(mean, base, period.observations),
            Some(values.len() as u64),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mean_quantile_table(
    frame: &DataFrame,
    id: &'static str,
    title: &'static str,
    base: u32,
    periods: &[PeriodInfo],
    by_date: bool,
    by_group: bool,
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
    let groups = frame
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;
    let period_labels = frame.column("period")?.as_materialized_series().str()?;
    let means = frame
        .column("mean_return")?
        .as_materialized_series()
        .f64()?;
    let errors = frame.column("std_error")?.as_materialized_series().f64()?;
    let counts = frame.column("count")?.as_materialized_series().u32()?;

    let mut rows = Vec::<(
        Option<i64>,
        Option<u32>,
        Option<String>,
        String,
        Option<f64>,
        Option<f64>,
        Option<u64>,
    )>::with_capacity(frame.height());
    for index in 0..frame.height() {
        let period = period_labels.get(index).expect("period is validated");
        rows.push((
            dates.as_ref().and_then(|dates| dates.physical().get(index)),
            quantiles.get(index),
            groups
                .as_ref()
                .and_then(|groups| groups.get(index).map(ToOwned::to_owned)),
            period.to_string(),
            convert_return(means.get(index), base, period_observations(periods, period)),
            convert_error(
                errors.get(index),
                base,
                period_observations(periods, period),
            ),
            counts.get(index).map(u64::from),
        ));
    }
    rows.sort_by(|left, right| {
        let mut ordering = if by_date {
            left.0.cmp(&right.0)
        } else {
            std::cmp::Ordering::Equal
        };
        ordering = ordering.then(left.1.cmp(&right.1));
        if by_group {
            ordering = ordering.then(left.2.cmp(&right.2));
        }
        ordering.then(left.3.cmp(&right.3))
    });

    let mut columns = Vec::new();
    let mut sort_by = Vec::new();
    if by_date {
        columns.push(ReportColumn::new(
            "date",
            "Date",
            ColumnRole::Dimension,
            None,
            ReportColumnData::Datetime {
                values: rows.iter().map(|row| row.0).collect(),
                time_unit: date_time_unit,
                timezone: date_timezone,
            },
        ));
        sort_by.push("date".to_string());
    }
    columns.push(ReportColumn::new(
        "factor_quantile",
        "Factor Quantile",
        ColumnRole::Dimension,
        Some(DisplayFormat::new(DisplayUnit::Count, 0)),
        ReportColumnData::UInt32(rows.iter().map(|row| row.1).collect()),
    ));
    sort_by.push("factor_quantile".to_string());
    if by_group {
        columns.push(ReportColumn::new(
            "group",
            "Group",
            ColumnRole::Dimension,
            None,
            ReportColumnData::String(rows.iter().map(|row| row.2.clone()).collect()),
        ));
        sort_by.push("group".to_string());
    }
    columns.extend(value_columns(
        rows.iter().map(|row| Some(row.3.clone())).collect(),
        rows.iter().map(|row| row.4).collect(),
        rows.iter().map(|row| row.5).collect(),
        rows.iter().map(|row| row.6).collect(),
    ));
    sort_by.push("period".to_string());
    ReportTable::new(id, title, sort_by, columns)
}

fn spread_table(
    frame: &DataFrame,
    base: u32,
    periods: &[PeriodInfo],
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
) -> Result<ReportTable> {
    let dates = frame.column("date")?.as_materialized_series().datetime()?;
    let period_labels = frame.column("period")?.as_materialized_series().str()?;
    let spreads = frame
        .column("mean_return_difference")?
        .as_materialized_series()
        .f64()?;
    let errors = frame
        .column("joint_std_error")?
        .as_materialized_series()
        .f64()?;
    let mut rows =
        Vec::<(Option<i64>, String, Option<f64>, Option<f64>)>::with_capacity(frame.height());
    for index in 0..frame.height() {
        let period = period_labels.get(index).expect("period is validated");
        rows.push((
            dates.physical().get(index),
            period.to_string(),
            convert_return(
                spreads.get(index),
                base,
                period_observations(periods, period),
            ),
            convert_error(
                errors.get(index),
                base,
                period_observations(periods, period),
            ),
        ));
    }
    rows.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    ReportTable::new(
        "returns.daily_spread",
        "Mean Quantile Return Spread",
        vec!["date".to_string(), "period".to_string()],
        vec![
            ReportColumn::new(
                "date",
                "Date",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: rows.iter().map(|row| row.0).collect(),
                    time_unit: date_time_unit,
                    timezone: date_timezone,
                },
            ),
            ReportColumn::new(
                "period",
                "Period",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(rows.iter().map(|row| Some(row.1.clone())).collect()),
            ),
            ReportColumn::new(
                "mean_return_difference",
                "Mean Return Difference",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
                ReportColumnData::Float64(rows.iter().map(|row| row.2).collect()),
            ),
            ReportColumn::new(
                "joint_std_error",
                "Joint Std Error",
                ColumnRole::Statistic,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
                ReportColumnData::Float64(rows.iter().map(|row| row.3).collect()),
            ),
        ],
    )
}

fn factor_returns_table(
    frame: &DataFrame,
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
) -> Result<ReportTable> {
    let dates = frame.column("date")?.as_materialized_series().datetime()?;
    let periods = frame.column("period")?.as_materialized_series().str()?;
    let returns = frame
        .column("factor_return")?
        .as_materialized_series()
        .f64()?;
    let mut rows = Vec::<(Option<i64>, Option<String>, Option<f64>)>::with_capacity(frame.height());
    for index in 0..frame.height() {
        rows.push((
            dates.physical().get(index),
            periods.get(index).map(ToOwned::to_owned),
            returns.get(index),
        ));
    }
    rows.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    ReportTable::new(
        "returns.factor_returns",
        "Factor Returns",
        vec!["date".to_string(), "period".to_string()],
        vec![
            ReportColumn::new(
                "date",
                "Date",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: rows.iter().map(|row| row.0).collect(),
                    time_unit: date_time_unit,
                    timezone: date_timezone,
                },
            ),
            ReportColumn::new(
                "period",
                "Period",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(rows.iter().map(|row| row.1.clone()).collect()),
            ),
            ReportColumn::new(
                "factor_return",
                "Factor Return",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
                ReportColumnData::Float64(rows.iter().map(|row| row.2).collect()),
            ),
        ],
    )
}

fn factor_cumulative_table(
    factor_returns_frame: &DataFrame,
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
) -> Result<ReportTable> {
    let periods = factor_returns_frame
        .column("period")?
        .as_materialized_series()
        .str()?;
    let mask = BooleanChunked::from_iter_values(
        "period_filter".into(),
        periods.iter().map(|period| period == Some("1D")),
    );
    let selected = factor_returns_frame.filter(&mask)?;
    let adapted = DataFrame::new(
        selected.height(),
        vec![
            selected.column("date")?.clone(),
            selected
                .column("factor_return")?
                .clone()
                .with_name("return".into()),
        ],
    )?;
    let cumulative = cumulative_returns(&adapted)?;
    let dates = cumulative
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let values = cumulative
        .column("cumulative_return")?
        .as_materialized_series()
        .f64()?;
    ReportTable::new(
        "returns.factor_cumulative_1d",
        "Factor Weighted Portfolio Cumulative Return (1D)",
        vec!["date".to_string()],
        vec![
            ReportColumn::new(
                "date",
                "Date",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: (0..cumulative.height())
                        .map(|index| dates.physical().get(index))
                        .collect(),
                    time_unit: date_time_unit,
                    timezone: date_timezone,
                },
            ),
            ReportColumn::new(
                "cumulative_return",
                "Cumulative Return",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
                ReportColumnData::Float64(
                    (0..cumulative.height())
                        .map(|index| values.get(index))
                        .collect(),
                ),
            ),
        ],
    )
}

fn quantile_cumulative_table(
    daily_by_quantile: &DataFrame,
    date_time_unit: ReportTimeUnit,
    date_timezone: Option<String>,
) -> Result<ReportTable> {
    let dates = daily_by_quantile
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let quantiles = daily_by_quantile
        .column("factor_quantile")?
        .as_materialized_series()
        .u32()?;
    let periods = daily_by_quantile
        .column("period")?
        .as_materialized_series()
        .str()?;
    let returns = daily_by_quantile
        .column("mean_return")?
        .as_materialized_series()
        .f64()?;
    let mut rows = Vec::<(u32, i64, f64)>::new();
    let mut wealth_by_quantile = BTreeMap::<u32, f64>::new();
    for index in 0..daily_by_quantile.height() {
        if periods.get(index) != Some("1D") {
            continue;
        }
        let quantile = quantiles.get(index).expect("quantile is validated");
        let wealth = wealth_by_quantile.entry(quantile).or_insert(1.0);
        *wealth *= 1.0 + returns.get(index).unwrap_or(0.0);
        rows.push((
            quantile,
            dates.physical().get(index).expect("date is validated"),
            *wealth,
        ));
    }
    rows.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));

    ReportTable::new(
        "returns.quantile_cumulative_1d",
        "Cumulative Return by Quantile (1D)",
        vec!["date".to_string(), "factor_quantile".to_string()],
        vec![
            ReportColumn::new(
                "date",
                "Date",
                ColumnRole::Dimension,
                None,
                ReportColumnData::Datetime {
                    values: rows.iter().map(|row| Some(row.1)).collect(),
                    time_unit: date_time_unit,
                    timezone: date_timezone,
                },
            ),
            ReportColumn::new(
                "factor_quantile",
                "Factor Quantile",
                ColumnRole::Dimension,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt32(rows.iter().map(|row| Some(row.0)).collect()),
            ),
            ReportColumn::new(
                "cumulative_return",
                "Cumulative Return",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
                ReportColumnData::Float64(rows.iter().map(|row| Some(row.2)).collect()),
            ),
        ],
    )
}

fn value_columns(
    periods: Vec<Option<String>>,
    means: Vec<Option<f64>>,
    errors: Vec<Option<f64>>,
    counts: Vec<Option<u64>>,
) -> Vec<ReportColumn> {
    vec![
        ReportColumn::new(
            "period",
            "Period",
            ColumnRole::Dimension,
            None,
            ReportColumnData::String(periods),
        ),
        ReportColumn::new(
            "mean_return",
            "Mean Return",
            ColumnRole::Measure,
            Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
            ReportColumnData::Float64(means),
        ),
        ReportColumn::new(
            "std_error",
            "Std Error",
            ColumnRole::Statistic,
            Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 6)),
            ReportColumnData::Float64(errors),
        ),
        ReportColumn::new(
            "count",
            "Count",
            ColumnRole::Measure,
            Some(DisplayFormat::new(DisplayUnit::Count, 0)),
            ReportColumnData::UInt64(counts),
        ),
    ]
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
