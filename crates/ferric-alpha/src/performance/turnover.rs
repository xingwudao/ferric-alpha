use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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
    if let Some(output) = try_dense_rank_autocorrelation(factor_data, period)? {
        return rank_autocorrelation_frame(output, period);
    }

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

    rank_autocorrelation_frame(
        RankAutocorrelationOutput {
            dates,
            correlations,
            date_dtype: data.date_dtype,
        },
        period,
    )
}

struct RankAutocorrelationOutput {
    dates: Vec<i64>,
    correlations: Vec<Option<f64>>,
    date_dtype: DataType,
}

fn try_dense_rank_autocorrelation(
    factor_data: &DataFrame,
    period: Period,
) -> Result<Option<RankAutocorrelationOutput>> {
    let date = required_column(factor_data, "date")?;
    let asset = required_column(factor_data, "asset")?;
    let factor = required_column(factor_data, "factor")?;
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
    if !matches!(factor.dtype(), DataType::Float64) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "factor",
            expected: "Float64",
            actual: factor.dtype().to_string(),
        });
    }
    if let Ok(group) = factor_data.column("group")
        && !matches!(group.dtype(), DataType::String)
    {
        return Err(FerricAlphaError::InvalidDtype {
            column: "group",
            expected: "String",
            actual: group.dtype().to_string(),
        });
    }
    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let factors = factor.as_materialized_series().f64()?;
    let mut unique_dates = BTreeSet::new();
    let mut unique_assets = BTreeSet::new();
    let mut seen = HashSet::with_capacity(factor_data.height());
    for (date, asset) in dates.physical().iter().zip(assets.iter()) {
        let (Some(date), Some(asset)) = (date, asset) else {
            return Err(FerricAlphaError::NullKey);
        };
        if !seen.insert((date, asset)) {
            return Err(FerricAlphaError::DuplicateKey);
        }
        unique_dates.insert(date);
        unique_assets.insert(asset);
    }

    let dense_size = unique_dates.len().checked_mul(unique_assets.len());
    if dense_size != Some(factor_data.height()) {
        return Ok(None);
    }

    let output_dates = unique_dates.into_iter().collect::<Vec<_>>();
    let date_indexes = output_dates
        .iter()
        .enumerate()
        .map(|(index, date)| (*date, index))
        .collect::<HashMap<_, _>>();
    let asset_indexes = unique_assets
        .into_iter()
        .enumerate()
        .map(|(index, asset)| (asset, index))
        .collect::<HashMap<_, _>>();
    let asset_count = asset_indexes.len();
    if asset_count == 0 {
        return Ok(Some(RankAutocorrelationOutput {
            dates: output_dates,
            correlations: Vec::new(),
            date_dtype: date.dtype().clone(),
        }));
    }
    let mut rank_matrix = vec![0.0; factor_data.height()];
    let mut all_finite = true;
    for index in 0..factor_data.height() {
        let date = dates
            .physical()
            .get(index)
            .expect("validated date should not be null");
        let asset = assets
            .get(index)
            .expect("validated asset should not be null");
        let Some(value) = factors.get(index).filter(|value| value.is_finite()) else {
            all_finite = false;
            continue;
        };
        let row = date_indexes[&date];
        let column = asset_indexes[asset];
        rank_matrix[row * asset_count + column] = value;
    }
    if !all_finite {
        return Ok(None);
    }

    for row in rank_matrix.chunks_exact_mut(asset_count) {
        let ranks = average_rank(row)?;
        row.copy_from_slice(&ranks);
    }

    let lag = period.count() as usize;
    let mut correlations = vec![None; output_dates.len()];
    for index in lag..output_dates.len() {
        let current = &rank_matrix[index * asset_count..(index + 1) * asset_count];
        let previous = &rank_matrix[(index - lag) * asset_count..(index - lag + 1) * asset_count];
        correlations[index] = pearson(current, previous)?;
    }

    Ok(Some(RankAutocorrelationOutput {
        dates: output_dates,
        correlations,
        date_dtype: date.dtype().clone(),
    }))
}

fn rank_autocorrelation_frame(
    output: RankAutocorrelationOutput,
    period: Period,
) -> Result<DataFrame> {
    let length = output.dates.len();
    let frame = DataFrame::new(
        length,
        vec![
            date_column("date", output.dates, &output.date_dtype)?,
            Series::new("period".into(), vec![period.label(); length]).into(),
            Series::new("autocorrelation".into(), output.correlations).into(),
        ],
    )?;
    Ok(frame.sort(["date"], SortMultipleOptions::default())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{IntoColumn, TimeUnit};

    #[test]
    fn dense_rank_autocorrelation_handles_shuffled_rows_ties_and_multi_lag() {
        let day_1 = 1_704_067_200_000_i64;
        let day_2 = day_1 + 86_400_000;
        let day_3 = day_2 + 86_400_000;
        let frame = DataFrame::new(
            9,
            vec![
                Series::new(
                    "date".into(),
                    [
                        day_2, day_1, day_3, day_1, day_2, day_3, day_3, day_2, day_1,
                    ],
                )
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
                .into_column(),
                Series::new(
                    "asset".into(),
                    ["C", "A", "B", "C", "A", "A", "C", "B", "B"],
                )
                .into_column(),
                Series::new(
                    "factor".into(),
                    [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 1.0],
                )
                .into_column(),
            ],
        )
        .unwrap();

        let output = try_dense_rank_autocorrelation(&frame, Period::new(2).unwrap())
            .unwrap()
            .expect("complete panel should use dense path");

        assert_eq!(output.dates, vec![day_1, day_2, day_3]);
        assert_eq!(output.correlations[0], None);
        assert_eq!(output.correlations[1], None);
        assert_eq!(output.correlations[2], Some(-0.866_025_403_784_438_7));
    }

    #[test]
    fn dense_rank_autocorrelation_accepts_an_empty_factor_frame() {
        let frame = DataFrame::new(
            0,
            vec![
                Series::new("date".into(), Vec::<i64>::new())
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                    .unwrap()
                    .into_column(),
                Series::new("asset".into(), Vec::<&str>::new()).into_column(),
                Series::new("factor".into(), Vec::<f64>::new()).into_column(),
            ],
        )
        .unwrap();

        let output = factor_rank_autocorrelation(&frame, Period::new(1).unwrap()).unwrap();

        assert_eq!(output.height(), 0);
        assert_eq!(output.column("autocorrelation").unwrap().null_count(), 0);
    }
}
