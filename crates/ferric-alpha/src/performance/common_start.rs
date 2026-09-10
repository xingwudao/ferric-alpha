use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use polars::prelude::{DataFrame, DataType, IntoColumn, NamedFrom, Series, SortMultipleOptions};

use crate::data::{ensure_matching_date_timezone, required_column};
use crate::error::{FerricAlphaError, Result};
use crate::performance::EventReturnsOptions;
use crate::performance::frame::{FactorRow, ensure_dtype, mean, validate_factor_data};
use crate::statistics::sample_std;

const OPERATION: &str = "common_start_returns";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonStartReturnsOptions {
    pub periods_before: u32,
    pub periods_after: u32,
    pub cumulative: bool,
    pub mean_by_date: bool,
    pub demeaned: bool,
}

impl Default for CommonStartReturnsOptions {
    fn default() -> Self {
        Self {
            periods_before: 10,
            periods_after: 15,
            cumulative: false,
            mean_by_date: false,
            demeaned: false,
        }
    }
}

impl CommonStartReturnsOptions {
    pub fn with_periods_before(mut self, periods_before: u32) -> Self {
        self.periods_before = periods_before;
        self
    }

    pub fn with_periods_after(mut self, periods_after: u32) -> Self {
        self.periods_after = periods_after;
        self
    }

    pub fn with_cumulative(mut self, cumulative: bool) -> Self {
        self.cumulative = cumulative;
        self
    }

    pub fn with_mean_by_date(mut self, mean_by_date: bool) -> Self {
        self.mean_by_date = mean_by_date;
        self
    }

    pub fn with_demeaned(mut self, demeaned: bool) -> Self {
        self.demeaned = demeaned;
        self
    }
}

#[derive(Debug, Clone)]
struct ReturnCalendar {
    dates: Vec<i64>,
    values: HashMap<(String, i64), Option<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommonStartRow {
    date: i64,
    asset: String,
    quantile: Option<u32>,
    group: Option<String>,
}

pub fn common_start_returns(
    factor_data: &DataFrame,
    returns: &DataFrame,
    options: CommonStartReturnsOptions,
) -> Result<DataFrame> {
    let data = validate_factor_data(factor_data, false, false, false)?;
    let calendar = validate_returns(factor_data, returns, options.cumulative)?;
    if calendar.dates.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: OPERATION,
        });
    }

    let rows = common_start_rows(&data.rows, false);
    let values_by_offset = aligned_common_start_values(
        &rows,
        options.demeaned.then_some(rows.as_slice()),
        &calendar,
        options.periods_before,
        options.periods_after,
        options.mean_by_date,
    );

    common_start_summary_frame(values_by_offset)
}

pub fn average_cumulative_return_by_quantile_from_prices(
    factor_data: &DataFrame,
    returns: &DataFrame,
    options: EventReturnsOptions,
) -> Result<DataFrame> {
    let data = validate_factor_data(
        factor_data,
        true,
        options.group_adjust || options.by_group,
        false,
    )?;
    if options.group_adjust || options.by_group {
        return Err(FerricAlphaError::InvalidInput {
            operation: "average_cumulative_return_by_quantile_from_prices",
            reason: "group-adjusted price compatibility is not implemented yet",
        });
    }
    let calendar = validate_returns(factor_data, returns, true)?;
    if calendar.dates.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: "average_cumulative_return_by_quantile_from_prices",
        });
    }

    let rows = common_start_rows(&data.rows, true);
    let universe = options.demeaned.then_some(rows.as_slice());
    let mut rows_by_quantile = BTreeMap::<u32, Vec<CommonStartRow>>::new();
    for row in &rows {
        if let Some(quantile) = row.quantile {
            rows_by_quantile
                .entry(quantile)
                .or_default()
                .push(row.clone());
        }
    }

    let mut factor_quantiles = Vec::new();
    let mut offsets = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut counts = Vec::new();
    for (quantile, quantile_rows) in rows_by_quantile {
        let values_by_offset = aligned_common_start_values(
            &quantile_rows,
            universe,
            &calendar,
            options.periods_before,
            options.periods_after,
            true,
        );
        for (offset, values) in values_by_offset {
            if values.is_empty() {
                continue;
            }
            factor_quantiles.push(quantile);
            offsets.push(offset);
            means.push(mean(&values));
            stds.push(sample_std(&values)?);
            counts.push(values.len() as u64);
        }
    }

    if factor_quantiles.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: "average_cumulative_return_by_quantile_from_prices",
        });
    }

    let frame = DataFrame::new(
        factor_quantiles.len(),
        vec![
            Series::new("factor_quantile".into(), factor_quantiles).into_column(),
            Series::new("offset".into(), offsets).into_column(),
            Series::new("mean_cumulative_return".into(), means).into_column(),
            Series::new("std_cumulative_return".into(), stds).into_column(),
            Series::new("count".into(), counts).into_column(),
        ],
    )?
    .sort(
        ["factor_quantile", "offset"],
        SortMultipleOptions::default(),
    )?;
    Ok(frame)
}

fn common_start_summary_frame(values_by_offset: BTreeMap<i32, Vec<f64>>) -> Result<DataFrame> {
    let mut offsets = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut counts = Vec::new();
    for (offset, values) in values_by_offset {
        if values.is_empty() {
            continue;
        }
        offsets.push(offset);
        means.push(mean(&values));
        stds.push(sample_std(&values)?);
        counts.push(values.len() as u64);
    }

    if offsets.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: OPERATION,
        });
    }

    let frame = DataFrame::new(
        offsets.len(),
        vec![
            Series::new("offset".into(), offsets).into_column(),
            Series::new("mean".into(), means).into_column(),
            Series::new("std".into(), stds).into_column(),
            Series::new("count".into(), counts).into_column(),
        ],
    )?
    .sort(["offset"], SortMultipleOptions::default())?;
    Ok(frame)
}

fn common_start_rows(rows: &[FactorRow], require_quantile: bool) -> Vec<CommonStartRow> {
    rows.iter()
        .filter(|row| row.factor.is_some())
        .filter(|row| !require_quantile || matches!(row.quantile, Some(value) if value > 0))
        .map(|row| CommonStartRow {
            date: row.date,
            asset: row.asset.clone(),
            quantile: row.quantile,
            group: row.group.clone(),
        })
        .collect()
}

fn aligned_common_start_values(
    rows: &[CommonStartRow],
    demean_rows: Option<&[CommonStartRow]>,
    calendar: &ReturnCalendar,
    periods_before: u32,
    periods_after: u32,
    mean_by_date: bool,
) -> BTreeMap<i32, Vec<f64>> {
    let date_index = calendar
        .dates
        .iter()
        .enumerate()
        .map(|(index, date)| (*date, index))
        .collect::<HashMap<_, _>>();
    let rows_by_date = rows.iter().fold(
        BTreeMap::<i64, Vec<&CommonStartRow>>::new(),
        |mut acc, row| {
            acc.entry(row.date).or_default().push(row);
            acc
        },
    );
    let demean_assets_by_date = demean_rows.map(|rows| {
        rows.iter()
            .fold(BTreeMap::<i64, BTreeSet<String>>::new(), |mut acc, row| {
                acc.entry(row.date).or_default().insert(row.asset.clone());
                acc
            })
    });

    let mut values_by_offset = BTreeMap::<i32, Vec<f64>>::new();
    let start = -(periods_before as i32);
    let end = periods_after as i32;
    for offset in start..=end {
        values_by_offset.entry(offset).or_default();
    }

    for (event_date, rows) in rows_by_date {
        let Some(day_zero_index) = date_index.get(&event_date).copied() else {
            continue;
        };
        let event_assets = rows
            .iter()
            .map(|row| row.asset.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if event_assets.is_empty() {
            continue;
        }

        for offset in start..=end {
            let target_index = day_zero_index as i32 + offset;
            if target_index < 0 || target_index >= calendar.dates.len() as i32 {
                continue;
            }
            let target_date = calendar.dates[target_index as usize];
            let universe_mean = demean_assets_by_date
                .as_ref()
                .and_then(|assets_by_date| assets_by_date.get(&event_date))
                .map(|assets| {
                    let universe_values = assets
                        .iter()
                        .filter_map(|asset| return_value(calendar, target_date, asset))
                        .collect::<Vec<_>>();
                    mean(&universe_values).unwrap_or(0.0)
                })
                .unwrap_or(0.0);
            let event_values = event_assets
                .iter()
                .filter_map(|asset| {
                    return_value(calendar, target_date, asset).map(|value| value - universe_mean)
                })
                .collect::<Vec<_>>();
            if event_values.is_empty() {
                continue;
            }
            if mean_by_date {
                values_by_offset
                    .entry(offset)
                    .or_default()
                    .push(mean(&event_values).expect("event values are non-empty"));
            } else {
                values_by_offset
                    .entry(offset)
                    .or_default()
                    .extend(event_values);
            }
        }
    }
    values_by_offset
}

fn validate_returns(
    factor_data: &DataFrame,
    returns: &DataFrame,
    cumulative: bool,
) -> Result<ReturnCalendar> {
    ensure_matching_date_timezone(factor_data, returns)?;
    let date = required_column(returns, "date")?;
    let asset = required_column(returns, "asset")?;
    let ret = required_column(returns, "return")?;
    ensure_dtype("date", date.dtype(), "Datetime", |dtype| {
        matches!(dtype, DataType::Datetime(_, _))
    })?;
    ensure_dtype("asset", asset.dtype(), "String", |dtype| {
        matches!(dtype, DataType::String)
    })?;
    ensure_dtype("return", ret.dtype(), "Float64", |dtype| {
        matches!(dtype, DataType::Float64)
    })?;
    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let returns_values = ret.as_materialized_series().f64()?;
    let mut seen = HashSet::with_capacity(returns.height());
    let mut calendar_dates = BTreeSet::new();
    let mut raw_by_asset = BTreeMap::<String, Vec<(i64, Option<f64>)>>::new();
    for ((date, asset), ret) in dates
        .physical()
        .iter()
        .zip(assets.iter())
        .zip(returns_values.iter())
    {
        let (Some(date), Some(asset)) = (date, asset) else {
            return Err(FerricAlphaError::NullKey);
        };
        if !seen.insert((date, asset.to_owned())) {
            return Err(FerricAlphaError::DuplicateKey);
        }
        calendar_dates.insert(date);
        raw_by_asset
            .entry(asset.to_owned())
            .or_default()
            .push((date, ret.filter(|value| value.is_finite())));
    }

    let calendar_dates = calendar_dates.into_iter().collect::<Vec<_>>();
    let mut values = HashMap::with_capacity(returns.height());
    for (asset, mut rows) in raw_by_asset {
        rows.sort_by_key(|(date, _)| *date);
        let mut wealth = 1.0;
        for (date, value) in rows {
            let value = if cumulative {
                value
            } else {
                let ret = value.unwrap_or(0.0);
                wealth *= 1.0 + ret;
                Some(wealth)
            };
            values.insert((asset.clone(), date), value);
        }
    }

    Ok(ReturnCalendar {
        dates: calendar_dates,
        values,
    })
}

fn return_value(calendar: &ReturnCalendar, date: i64, asset: &str) -> Option<f64> {
    calendar
        .values
        .get(&(asset.to_owned(), date))
        .copied()
        .flatten()
}
