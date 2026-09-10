use std::collections::{BTreeMap, BTreeSet, HashSet};

use polars::prelude::{DataFrame, DataType, NamedFrom, Series, SortMultipleOptions};

use crate::calendar::Period;
use crate::data::required_column;
use crate::error::{FerricAlphaError, Result};
use crate::performance::frame::{date_column, validate_factor_data};
use crate::statistics::{average_rank, pearson};

pub fn quantile_turnover(
    quantile_factor: &DataFrame,
    quantile: u32,
    period: Period,
) -> Result<DataFrame> {
    if quantile == 0 {
        return Err(FerricAlphaError::InvalidOption {
            option: "quantile",
            reason: "quantile must be positive",
        });
    }

    let date = required_column(quantile_factor, "date")?;
    let asset = required_column(quantile_factor, "asset")?;
    let quantiles = required_column(quantile_factor, "factor_quantile")?;
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
    if !matches!(quantiles.dtype(), DataType::UInt32) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "factor_quantile",
            expected: "UInt32",
            actual: quantiles.dtype().to_string(),
        });
    }
    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let quantiles = quantiles.as_materialized_series().u32()?;
    let mut seen = HashSet::with_capacity(quantile_factor.height());
    let mut by_date: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    for ((date, asset), row_quantile) in dates
        .physical()
        .iter()
        .zip(assets.iter())
        .zip(quantiles.iter())
    {
        let (Some(date), Some(asset), Some(row_quantile)) = (date, asset, row_quantile) else {
            return Err(FerricAlphaError::InvalidInput {
                operation: "quantile_turnover",
                reason: "factor_quantile must be non-null",
            });
        };
        if row_quantile == 0 {
            return Err(FerricAlphaError::InvalidInput {
                operation: "quantile_turnover",
                reason: "factor_quantile must be positive",
            });
        }
        if !seen.insert((date, asset)) {
            return Err(FerricAlphaError::DuplicateKey);
        }
        let current = by_date.entry(date).or_default();
        if row_quantile == quantile {
            current.insert(asset.to_owned());
        }
    }

    let entries = by_date.into_iter().collect::<Vec<_>>();
    let lag = period.count() as usize;
    let mut output_dates = Vec::with_capacity(entries.len());
    let mut turnover = Vec::with_capacity(entries.len());
    for (index, (date, current)) in entries.iter().enumerate() {
        output_dates.push(*date);
        if index < lag || current.is_empty() {
            turnover.push(None);
            continue;
        }
        let previous = &entries[index - lag].1;
        let new_count = current.difference(previous).count();
        turnover.push(Some(new_count as f64 / current.len() as f64));
    }

    let length = output_dates.len();
    DataFrame::new(
        length,
        vec![
            date_column("date", output_dates, date.dtype())?,
            Series::new("factor_quantile".into(), vec![quantile; length]).into(),
            Series::new("period".into(), vec![period.label(); length]).into(),
            Series::new("turnover".into(), turnover).into(),
        ],
    )
    .map_err(Into::into)
}

pub fn factor_rank_autocorrelation(factor_data: &DataFrame, period: Period) -> Result<DataFrame> {
    let data = validate_factor_data(factor_data, false, false, false)?;
    let mut values_by_date: BTreeMap<i64, Vec<(String, f64)>> = BTreeMap::new();
    for row in &data.rows {
        if let Some(factor) = row.factor {
            values_by_date
                .entry(row.date)
                .or_default()
                .push((row.asset.clone(), factor));
        } else {
            values_by_date.entry(row.date).or_default();
        }
    }

    let mut ranks_by_date = Vec::with_capacity(values_by_date.len());
    for (date, values) in values_by_date {
        let raw = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        let ranks = average_rank(&raw)?;
        let by_asset = values
            .into_iter()
            .zip(ranks)
            .map(|((asset, _), rank)| (asset, rank))
            .collect::<BTreeMap<_, _>>();
        ranks_by_date.push((date, by_asset));
    }

    let lag = period.count() as usize;
    let mut dates = Vec::with_capacity(ranks_by_date.len());
    let mut correlations = Vec::with_capacity(ranks_by_date.len());
    for (index, (date, current)) in ranks_by_date.iter().enumerate() {
        dates.push(*date);
        if index < lag {
            correlations.push(None);
            continue;
        }
        let previous = &ranks_by_date[index - lag].1;
        let mut current_ranks = Vec::new();
        let mut previous_ranks = Vec::new();
        for (asset, current_rank) in current {
            if let Some(previous_rank) = previous.get(asset) {
                current_ranks.push(*current_rank);
                previous_ranks.push(*previous_rank);
            }
        }
        correlations.push(pearson(&current_ranks, &previous_ranks)?);
    }

    let length = dates.len();
    let frame = DataFrame::new(
        length,
        vec![
            date_column("date", dates, &data.date_dtype)?,
            Series::new("period".into(), vec![period.label(); length]).into(),
            Series::new("autocorrelation".into(), correlations).into(),
        ],
    )?;
    Ok(frame.sort(["date"], SortMultipleOptions::default())?)
}
