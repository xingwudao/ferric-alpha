use std::collections::BTreeMap;

use polars::prelude::{DataFrame, NamedFrom, Series};

use crate::error::Result;
use crate::performance::frame::{date_column, finish_frame, validate_factor_data};
use crate::performance::weights::weights_for_partition;
use crate::performance::{ReturnOptions, WeightOptions};

pub fn factor_returns(factor_data: &DataFrame, options: ReturnOptions) -> Result<DataFrame> {
    let data = validate_factor_data(
        factor_data,
        false,
        options.weight_options.group_adjust,
        true,
    )?;
    let weights = compute_weights(&data, options.weight_options);

    if options.by_asset {
        let mut dates = Vec::new();
        let mut assets = Vec::new();
        let mut periods = Vec::new();
        let mut weighted = Vec::new();
        for (index, row) in data.rows.iter().enumerate() {
            for (period_index, period) in data.periods.iter().enumerate() {
                dates.push(row.date);
                assets.push(row.asset.clone());
                periods.push(period.label.clone());
                weighted.push(match (weights[index], row.returns[period_index]) {
                    (Some(weight), Some(ret)) => Some(weight * ret),
                    _ => None,
                });
            }
        }
        return finish_frame(
            DataFrame::new(
                dates.len(),
                vec![
                    date_column("date", dates, &data.date_dtype)?,
                    Series::new("asset".into(), assets).into(),
                    Series::new("period".into(), periods).into(),
                    Series::new("weighted_return".into(), weighted).into(),
                ],
            )?,
            &["date", "asset", "period"],
        );
    }

    let mut dates = Vec::new();
    let mut periods = Vec::new();
    let mut values = Vec::new();
    let mut by_date: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (index, row) in data.rows.iter().enumerate() {
        by_date.entry(row.date).or_default().push(index);
    }
    for (date, indexes) in by_date {
        for (period_index, period) in data.periods.iter().enumerate() {
            let mut total = 0.0;
            let mut count = 0;
            for index in &indexes {
                if let (Some(weight), Some(ret)) =
                    (weights[*index], data.rows[*index].returns[period_index])
                {
                    total += weight * ret;
                    count += 1;
                }
            }
            dates.push(date);
            periods.push(period.label.clone());
            values.push((count > 0).then_some(total));
        }
    }
    finish_frame(
        DataFrame::new(
            dates.len(),
            vec![
                date_column("date", dates, &data.date_dtype)?,
                Series::new("period".into(), periods).into(),
                Series::new("factor_return".into(), values).into(),
            ],
        )?,
        &["date", "period"],
    )
}

pub(crate) fn compute_weights(
    data: &crate::performance::frame::FactorData,
    options: WeightOptions,
) -> Vec<Option<f64>> {
    let mut weights = vec![None; data.rows.len()];
    if options.group_adjust {
        let mut groups: BTreeMap<(i64, String), Vec<usize>> = BTreeMap::new();
        for (index, row) in data.rows.iter().enumerate() {
            groups
                .entry((row.date, row.group.clone().expect("validated group")))
                .or_default()
                .push(index);
        }
        let mut valid_by_date: BTreeMap<i64, usize> = BTreeMap::new();
        let mut locals = Vec::new();
        for ((date, group), indexes) in groups {
            let local = weights_for_partition(&indexes, &data.rows, options);
            let valid = local.iter().any(Option::is_some);
            if valid {
                *valid_by_date.entry(date).or_default() += 1;
            }
            locals.push(((date, group), indexes, local, valid));
        }
        for ((date, _), indexes, local, valid) in locals {
            if !valid {
                continue;
            }
            let scale = 1.0 / valid_by_date[&date] as f64;
            for (index, value) in indexes.iter().zip(local) {
                weights[*index] = value.map(|weight| weight * scale);
            }
        }
    } else {
        let mut by_date: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (index, row) in data.rows.iter().enumerate() {
            by_date.entry(row.date).or_default().push(index);
        }
        for indexes in by_date.values() {
            for (index, value) in indexes
                .iter()
                .zip(weights_for_partition(indexes, &data.rows, options))
            {
                weights[*index] = value;
            }
        }
    }
    weights
}
