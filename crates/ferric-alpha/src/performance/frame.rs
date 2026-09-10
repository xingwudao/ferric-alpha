use std::collections::{BTreeMap, HashSet};

use chrono::{Datelike, NaiveDate};
use chrono_tz::Tz;
use polars::prelude::{
    Column, DataFrame, DataType, IntoColumn, NamedFrom, Series, SortMultipleOptions, TimeUnit,
};

use crate::data::{required_column, validate_factor_frame};
use crate::error::{FerricAlphaError, Result};
use crate::forward_returns::datetime_column;
use crate::performance::TimeBucket;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ForwardPeriod {
    pub(crate) column: String,
    pub(crate) label: String,
    pub(crate) observations: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct FactorRow {
    pub(crate) date: i64,
    pub(crate) asset: String,
    pub(crate) group: Option<String>,
    pub(crate) factor: Option<f64>,
    pub(crate) quantile: Option<u32>,
    pub(crate) returns: Vec<Option<f64>>,
}

#[derive(Debug, Clone)]
pub(crate) struct FactorData {
    pub(crate) rows: Vec<FactorRow>,
    pub(crate) periods: Vec<ForwardPeriod>,
    pub(crate) date_dtype: DataType,
    pub(crate) has_group: bool,
}

pub(crate) fn validate_factor_data(
    frame: &DataFrame,
    require_quantile: bool,
    require_group: bool,
    require_returns: bool,
) -> Result<FactorData> {
    let sorted = validate_factor_frame(frame)?;
    let date_dtype = sorted.column("date")?.dtype().clone();
    let periods = discover_forward_periods(&sorted)?;
    if require_returns && periods.is_empty() {
        return Err(FerricAlphaError::MissingColumn("forward_return_*"));
    }

    let has_group = sorted.column("group").is_ok();
    if require_group && !has_group {
        return Err(FerricAlphaError::MissingColumn("group"));
    }

    if require_quantile {
        let quantile = required_column(&sorted, "factor_quantile")?;
        ensure_dtype("factor_quantile", quantile.dtype(), "UInt32", |dtype| {
            matches!(dtype, DataType::UInt32)
        })?;
    }

    let dates = sorted.column("date")?.as_materialized_series().datetime()?;
    let assets = sorted.column("asset")?.as_materialized_series().str()?;
    let factors = sorted.column("factor")?.as_materialized_series().f64()?;
    let groups = sorted
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;
    let quantiles = sorted
        .column("factor_quantile")
        .ok()
        .map(|column| column.as_materialized_series().u32())
        .transpose()?;
    let returns = periods
        .iter()
        .map(|period| {
            Ok(sorted
                .column(&period.column)?
                .as_materialized_series()
                .f64()?)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut rows = Vec::with_capacity(sorted.height());
    for index in 0..sorted.height() {
        let group = groups
            .as_ref()
            .and_then(|groups| groups.get(index).map(ToOwned::to_owned));
        if require_group && group.is_none() {
            return Err(FerricAlphaError::InvalidInput {
                operation: "validate_factor_data",
                reason: "group must not contain null values",
            });
        }

        let quantile = quantiles.as_ref().and_then(|values| values.get(index));
        if require_quantile && !matches!(quantile, Some(value) if value > 0) {
            return Err(FerricAlphaError::InvalidInput {
                operation: "validate_factor_data",
                reason: "factor_quantile must be non-null and positive",
            });
        }

        rows.push(FactorRow {
            date: dates
                .physical()
                .get(index)
                .expect("validated date should not be null"),
            asset: assets
                .get(index)
                .expect("validated asset should not be null")
                .to_owned(),
            group,
            factor: factors.get(index).filter(|value| value.is_finite()),
            quantile,
            returns: returns
                .iter()
                .map(|series| series.get(index).filter(|value| value.is_finite()))
                .collect(),
        });
    }

    Ok(FactorData {
        rows,
        periods,
        date_dtype,
        has_group,
    })
}

pub(crate) fn discover_forward_periods(frame: &DataFrame) -> Result<Vec<ForwardPeriod>> {
    let mut periods = Vec::new();
    let mut observations = HashSet::new();
    for name in frame.get_column_names() {
        let Some(suffix) = name.strip_prefix("forward_return_") else {
            continue;
        };
        let Some(number) = suffix.strip_suffix('D') else {
            return Err(FerricAlphaError::InvalidInput {
                operation: "discover_forward_periods",
                reason: "forward return period must end with D",
            });
        };
        if number.is_empty() || number.starts_with('0') {
            return Err(FerricAlphaError::InvalidInput {
                operation: "discover_forward_periods",
                reason: "forward return period must be a positive daily count",
            });
        }
        let count = number
            .parse::<u32>()
            .map_err(|_| FerricAlphaError::InvalidInput {
                operation: "discover_forward_periods",
                reason: "forward return period must be numeric",
            })?;
        if count == 0 || !observations.insert(count) {
            return Err(FerricAlphaError::InvalidInput {
                operation: "discover_forward_periods",
                reason: "forward return periods must be unique and positive",
            });
        }
        let column = frame.column(name.as_str())?;
        ensure_dtype(name.as_str(), column.dtype(), "Float64", |dtype| {
            matches!(dtype, DataType::Float64)
        })?;
        periods.push(ForwardPeriod {
            column: name.to_string(),
            label: format!("{count}D"),
            observations: count,
        });
    }
    periods.sort_by_key(|period| period.observations);
    Ok(periods)
}

pub(crate) fn ensure_dtype(
    column: &str,
    actual: &DataType,
    expected: &'static str,
    predicate: impl FnOnce(&DataType) -> bool,
) -> Result<()> {
    if predicate(actual) {
        return Ok(());
    }
    Err(FerricAlphaError::InvalidDtype {
        column: Box::leak(column.to_owned().into_boxed_str()),
        expected,
        actual: actual.to_string(),
    })
}

pub(crate) fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

pub(crate) fn date_column(name: &str, values: Vec<i64>, dtype: &DataType) -> Result<Column> {
    datetime_column(name, values, dtype)
}

pub(crate) fn date_bucket_column(name: &str, values: Vec<Option<i32>>) -> Column {
    Series::new(name.into(), values)
        .cast(&DataType::Date)
        .expect("i32 date bucket should cast to Date")
        .into_column()
}

pub(crate) fn bucket_date(date: i64, dtype: &DataType, bucket: TimeBucket) -> Result<i32> {
    let days = epoch_days(date, dtype)?;
    let base = NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid epoch date");
    let date = base + chrono::Duration::days(days as i64);
    let bucketed = match bucket {
        TimeBucket::Daily => date,
        TimeBucket::Weekly => {
            let days_until_sunday = (7 - date.weekday().num_days_from_sunday()) % 7;
            date + chrono::Duration::days(days_until_sunday as i64)
        }
        TimeBucket::Monthly => {
            NaiveDate::from_ymd_opt(date.year(), date.month(), 1).expect("valid month start")
        }
    };
    Ok((bucketed - base).num_days() as i32)
}

fn epoch_days(value: i64, dtype: &DataType) -> Result<i32> {
    let (millis, zone) = match dtype {
        DataType::Datetime(TimeUnit::Nanoseconds, zone) => (value.div_euclid(1_000_000), zone),
        DataType::Datetime(TimeUnit::Microseconds, zone) => (value.div_euclid(1_000), zone),
        DataType::Datetime(TimeUnit::Milliseconds, zone) => (value, zone),
        actual => {
            return Err(FerricAlphaError::InvalidDtype {
                column: "date",
                expected: "Datetime",
                actual: actual.to_string(),
            });
        }
    };
    let utc =
        chrono::DateTime::from_timestamp_millis(millis).ok_or(FerricAlphaError::InvalidInput {
            operation: "bucket_date",
            reason: "datetime value is outside supported range",
        })?;
    let local_date = if let Some(zone) = zone {
        let timezone = zone
            .parse::<Tz>()
            .map_err(|_| FerricAlphaError::InvalidInput {
                operation: "bucket_date",
                reason: "unsupported datetime timezone",
            })?;
        utc.with_timezone(&timezone).date_naive()
    } else {
        utc.date_naive()
    };
    let base = NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid epoch date");
    Ok((local_date - base).num_days() as i32)
}

pub(crate) fn grouped_indexes<K: Ord>(
    indexes: impl Iterator<Item = (usize, K)>,
) -> BTreeMap<K, Vec<usize>> {
    let mut groups: BTreeMap<K, Vec<usize>> = BTreeMap::new();
    for (index, key) in indexes {
        groups.entry(key).or_default().push(index);
    }
    groups
}

pub(crate) fn finish_frame(frame: DataFrame, sort: &[&str]) -> Result<DataFrame> {
    if !sort.contains(&"period") {
        return Ok(frame.sort(sort.iter().copied(), SortMultipleOptions::default())?);
    }

    let periods = frame.column("period")?.as_materialized_series().str()?;
    let period_order = periods
        .iter()
        .map(|period| {
            period
                .and_then(|period| period.strip_suffix('D'))
                .and_then(|period| period.parse::<u32>().ok())
        })
        .collect::<Vec<_>>();
    let mut frame = frame.clone();
    frame.with_column(Series::new("__period_order".into(), period_order).into())?;
    let sort_keys = sort
        .iter()
        .map(|key| {
            if *key == "period" {
                "__period_order"
            } else {
                key
            }
        })
        .collect::<Vec<_>>();
    let mut sorted = frame.sort(sort_keys, SortMultipleOptions::default())?;
    sorted.drop_in_place("__period_order")?;
    Ok(sorted)
}
