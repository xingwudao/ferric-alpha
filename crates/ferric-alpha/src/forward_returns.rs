use std::collections::{BTreeSet, HashMap};

use polars::prelude::{
    Column, DataFrame, DataType, IntoColumn, NamedFrom, Series, SortMultipleOptions, TimeUnit,
};

use crate::calendar::{Period, TradingCalendar, validate_periods};
use crate::data::{ensure_matching_date_timezone, required_column, validate_factor_frame};
use crate::error::{FerricAlphaError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardReturnsOptions {
    pub cumulative_returns: bool,
    pub filter_zscore: Option<f64>,
}

impl Default for ForwardReturnsOptions {
    fn default() -> Self {
        Self {
            cumulative_returns: true,
            filter_zscore: None,
        }
    }
}

pub fn compute_forward_returns(
    factor: &DataFrame,
    prices: &DataFrame,
    periods: &[Period],
) -> Result<DataFrame> {
    compute_forward_returns_with_options(factor, prices, periods, ForwardReturnsOptions::default())
}

pub fn compute_forward_returns_with_options(
    factor: &DataFrame,
    prices: &DataFrame,
    periods: &[Period],
    options: ForwardReturnsOptions,
) -> Result<DataFrame> {
    let periods = validate_periods(periods)?;
    validate_filter_zscore(options.filter_zscore)?;
    let date_dtype = ensure_matching_date_timezone(factor, prices)?;
    let factor = validate_factor_frame(factor)?;
    let prices = validate_forward_price_frame(prices)?;

    let calendar = calendar_from_prices(&prices)?;
    let price_lookup = price_lookup(&prices)?;
    let period_labels = period_labels(&calendar, &periods, &date_dtype);

    let factor_dates = factor.column("date")?.as_materialized_series().datetime()?;
    let factor_assets = factor.column("asset")?.as_materialized_series().str()?;

    let mut out_dates = Vec::with_capacity(factor.height());
    let mut out_assets = Vec::with_capacity(factor.height());
    let mut out_returns = vec![Vec::with_capacity(factor.height()); periods.len()];

    for (date, asset) in factor_dates.physical().iter().zip(factor_assets.iter()) {
        let date = date.expect("validated factor date should not be null");
        let asset = asset.expect("validated factor asset should not be null");
        out_dates.push(date);
        out_assets.push(asset.to_owned());

        for (period_index, period) in periods.iter().enumerate() {
            let forward_return = calendar.offset(date, *period).and_then(|terminal_date| {
                let start_date = if options.cumulative_returns || period.count() == 1 {
                    date
                } else {
                    calendar.offset(date, Period::new(period.count() - 1).ok()?)?
                };
                let start = price_lookup
                    .get(&(asset.to_owned(), start_date))?
                    .as_ref()?;
                let end = price_lookup
                    .get(&(asset.to_owned(), terminal_date))?
                    .as_ref()?;
                Some(round_return(end / start - 1.0))
            });
            out_returns[period_index].push(forward_return);
        }
    }
    if let Some(filter_zscore) = options.filter_zscore {
        filter_outlier_forward_returns(&out_assets, &mut out_returns, filter_zscore);
    }

    let mut columns = vec![
        datetime_column("date", out_dates, &date_dtype)?,
        Series::new("asset".into(), out_assets).into(),
    ];
    for (label, values) in period_labels.iter().zip(out_returns) {
        columns.push(Series::new(format!("forward_return_{label}").into(), values).into());
    }

    Ok(DataFrame::new(factor.height(), columns)?
        .sort(["date", "asset"], SortMultipleOptions::default())?)
}

fn validate_filter_zscore(filter_zscore: Option<f64>) -> Result<()> {
    if let Some(value) = filter_zscore
        && (!value.is_finite() || value < 0.0)
    {
        return Err(FerricAlphaError::InvalidQuantiles {
            reason: "filter_zscore must be finite and non-negative",
        });
    }
    Ok(())
}

fn filter_outlier_forward_returns(
    assets: &[String],
    returns_by_period: &mut [Vec<Option<f64>>],
    filter_zscore: f64,
) {
    for returns in returns_by_period {
        let mut values_by_asset: HashMap<&str, Vec<f64>> = HashMap::new();
        for (asset, value) in assets.iter().zip(returns.iter()) {
            if let Some(value) = value {
                values_by_asset
                    .entry(asset.as_str())
                    .or_default()
                    .push(*value);
            }
        }
        let stats_by_asset = values_by_asset
            .into_iter()
            .filter_map(|(asset, values)| asset_stats(&values).map(|stats| (asset, stats)))
            .collect::<HashMap<_, _>>();
        for (asset, value) in assets.iter().zip(returns.iter_mut()) {
            let Some(current) = *value else {
                continue;
            };
            let Some((mean, std)) = stats_by_asset.get(asset.as_str()) else {
                continue;
            };
            if (current - mean).abs() > filter_zscore * std {
                *value = None;
            }
        }
    }
}

fn asset_stats(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let std = variance.sqrt();
    if std == 0.0 { None } else { Some((mean, std)) }
}

fn period_labels(calendar: &TradingCalendar, periods: &[Period], dtype: &DataType) -> Vec<String> {
    let Some(day_units) = units_per_day(dtype) else {
        return periods.iter().map(|period| period.label()).collect();
    };
    if !has_subdaily_sessions(calendar.sessions(), day_units) {
        return periods.iter().map(|period| period.label()).collect();
    }

    periods
        .iter()
        .map(|period| {
            period_span(calendar, *period)
                .and_then(|(start, terminal)| duration_label(calendar, start, terminal, day_units))
                .unwrap_or_else(|| period.label())
        })
        .collect()
}

fn has_subdaily_sessions(sessions: &[i64], day_units: i64) -> bool {
    sessions
        .windows(2)
        .any(|pair| pair[1] > pair[0] && pair[1] - pair[0] < day_units)
}

fn period_span(calendar: &TradingCalendar, period: Period) -> Option<(i64, i64)> {
    calendar.sessions().iter().copied().find_map(|date| {
        calendar
            .offset(date, period)
            .map(|terminal| (date, terminal))
    })
}

fn duration_label(
    calendar: &TradingCalendar,
    start: i64,
    terminal: i64,
    day_units: i64,
) -> Option<String> {
    let delta = terminal - start;
    if delta <= 0 {
        return None;
    }
    if delta >= day_units && start.rem_euclid(day_units) == terminal.rem_euclid(day_units) {
        return matching_session_day_label(calendar.sessions(), start, terminal, day_units);
    }
    let hour_units = day_units / 24;
    let minute_units = hour_units / 60;
    let second_units = minute_units / 60;
    if delta % day_units == 0 {
        Some(format!("{}D", delta / day_units))
    } else if hour_units > 0 && delta % hour_units == 0 {
        Some(format!("{}h", delta / hour_units))
    } else if minute_units > 0 && delta % minute_units == 0 {
        Some(format!("{}m", delta / minute_units))
    } else if second_units > 0 && delta % second_units == 0 {
        Some(format!("{}s", delta / second_units))
    } else {
        Some(format!("{}ms", delta / (day_units / 86_400_000).max(1)))
    }
}

fn matching_session_day_label(
    sessions: &[i64],
    start: i64,
    terminal: i64,
    day_units: i64,
) -> Option<String> {
    let time_of_day = start.rem_euclid(day_units);
    let day_count = sessions
        .iter()
        .copied()
        .filter(|session| {
            *session > start && *session <= terminal && session.rem_euclid(day_units) == time_of_day
        })
        .count();
    if day_count == 0 {
        None
    } else {
        Some(format!("{day_count}D"))
    }
}

fn units_per_day(dtype: &DataType) -> Option<i64> {
    match dtype {
        DataType::Datetime(TimeUnit::Nanoseconds, _) => Some(86_400_000_000_000),
        DataType::Datetime(TimeUnit::Microseconds, _) => Some(86_400_000_000),
        DataType::Datetime(TimeUnit::Milliseconds, _) => Some(86_400_000),
        _ => None,
    }
}

fn validate_forward_price_frame(frame: &DataFrame) -> Result<DataFrame> {
    let date = required_column(frame, "date")?;
    let asset = required_column(frame, "asset")?;
    let price = required_column(frame, "price")?;

    if !matches!(date.dtype(), DataType::Datetime(_, _)) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "date",
            expected: "Datetime",
            actual: date.dtype().to_string(),
        });
    }
    if !matches!(asset.dtype(), DataType::String) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "asset",
            expected: "String",
            actual: asset.dtype().to_string(),
        });
    }
    if !matches!(price.dtype(), DataType::Float64) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "price",
            expected: "Float64",
            actual: price.dtype().to_string(),
        });
    }

    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let prices = price.as_materialized_series().f64()?;
    let mut keys = BTreeSet::new();
    for ((date, asset), price) in dates
        .physical()
        .iter()
        .zip(assets.iter())
        .zip(prices.iter())
    {
        let (Some(date), Some(asset)) = (date, asset) else {
            return Err(FerricAlphaError::NullKey);
        };
        if let Some(price) = price
            && (!price.is_finite() || price <= 0.0)
        {
            return Err(FerricAlphaError::InvalidPrice);
        }
        if !keys.insert((date, asset)) {
            return Err(FerricAlphaError::DuplicateKey);
        }
    }

    Ok(frame.sort(["date", "asset"], SortMultipleOptions::default())?)
}

pub(crate) fn datetime_column(name: &str, values: Vec<i64>, dtype: &DataType) -> Result<Column> {
    match dtype {
        DataType::Datetime(unit, zone) => Ok(Series::new(name.into(), values)
            .into_datetime(*unit, zone.clone())
            .into_column()),
        actual => Err(crate::error::FerricAlphaError::InvalidDtype {
            column: "date",
            expected: "Datetime",
            actual: actual.to_string(),
        }),
    }
}

fn calendar_from_prices(prices: &DataFrame) -> Result<TradingCalendar> {
    let date_column = prices.column("date")?.as_materialized_series().datetime()?;
    let sessions = date_column
        .physical()
        .iter()
        .map(|date| date.expect("validated price date should not be null"))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let timezone = match prices.column("date")?.dtype() {
        DataType::Datetime(_, zone) => zone.as_ref().map(ToString::to_string),
        _ => None,
    };
    TradingCalendar::new(sessions, timezone)
}

fn price_lookup(prices: &DataFrame) -> Result<HashMap<(String, i64), Option<f64>>> {
    let dates = prices.column("date")?.as_materialized_series().datetime()?;
    let assets = prices.column("asset")?.as_materialized_series().str()?;
    let prices = prices.column("price")?.as_materialized_series().f64()?;
    let mut lookup = HashMap::with_capacity(prices.len());
    for ((date, asset), price) in dates
        .physical()
        .iter()
        .zip(assets.iter())
        .zip(prices.iter())
    {
        let date = date.expect("validated price date should not be null");
        let asset = asset.expect("validated price asset should not be null");
        lookup.insert((asset.to_owned(), date), price);
    }
    Ok(lookup)
}

fn round_return(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}
