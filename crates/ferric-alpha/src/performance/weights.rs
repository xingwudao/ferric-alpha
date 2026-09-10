use std::collections::BTreeMap;

use polars::prelude::{DataFrame, NamedFrom, Series};

use crate::error::Result;
use crate::performance::WeightOptions;
use crate::performance::frame::{
    date_column, finish_frame, grouped_indexes, mean, validate_factor_data,
};

type LocalGroupWeights = ((i64, String), Vec<usize>, Vec<Option<f64>>, bool);

pub fn factor_weights(factor_data: &DataFrame, options: WeightOptions) -> Result<DataFrame> {
    let data = validate_factor_data(factor_data, false, options.group_adjust, false)?;
    let mut weights = vec![None; data.rows.len()];

    if options.group_adjust {
        let groups = grouped_indexes(data.rows.iter().enumerate().map(|(index, row)| {
            (
                index,
                (row.date, row.group.clone().expect("validated group")),
            )
        }));
        let mut date_valid: BTreeMap<i64, usize> = BTreeMap::new();
        let mut local: Vec<LocalGroupWeights> = Vec::new();
        for (key, indexes) in groups {
            let local_weights = weights_for_partition(&indexes, &data.rows, options);
            let valid = local_weights.iter().any(Option::is_some);
            if valid {
                *date_valid.entry(key.0).or_default() += 1;
            }
            local.push((key, indexes, local_weights, valid));
        }
        for ((date, _), indexes, local_weights, valid) in local {
            if !valid {
                continue;
            }
            let scale = 1.0 / *date_valid.get(&date).expect("valid date count") as f64;
            for (index, value) in indexes.iter().zip(local_weights) {
                weights[*index] = value.map(|weight| weight * scale);
            }
        }
    } else {
        let groups = grouped_indexes(
            data.rows
                .iter()
                .enumerate()
                .map(|(index, row)| (index, row.date)),
        );
        for indexes in groups.values() {
            for (index, value) in indexes
                .iter()
                .zip(weights_for_partition(indexes, &data.rows, options))
            {
                weights[*index] = value;
            }
        }
    }

    let mut dates = Vec::with_capacity(data.rows.len());
    let mut assets = Vec::with_capacity(data.rows.len());
    let mut groups = Vec::with_capacity(data.rows.len());
    let mut out_weights = Vec::with_capacity(data.rows.len());
    for (row, weight) in data.rows.iter().zip(weights) {
        if row.factor.is_none() {
            continue;
        }
        dates.push(row.date);
        assets.push(row.asset.clone());
        if data.has_group {
            groups.push(row.group.clone());
        }
        out_weights.push(weight);
    }
    let mut columns = vec![
        date_column("date", dates, &data.date_dtype)?,
        Series::new("asset".into(), assets).into(),
    ];
    if data.has_group {
        columns.push(Series::new("group".into(), groups).into());
    }
    columns.push(Series::new("weight".into(), out_weights).into());
    finish_frame(
        DataFrame::new(columns[0].len(), columns)?,
        &["date", "asset"],
    )
}

pub(crate) fn weights_for_partition(
    indexes: &[usize],
    rows: &[crate::performance::frame::FactorRow],
    options: WeightOptions,
) -> Vec<Option<f64>> {
    if options.equal_weight {
        equal_weights(indexes, rows, options.demeaned)
    } else {
        factor_weighted(indexes, rows, options.demeaned)
    }
}

fn factor_weighted(
    indexes: &[usize],
    rows: &[crate::performance::frame::FactorRow],
    demeaned: bool,
) -> Vec<Option<f64>> {
    let values = indexes
        .iter()
        .filter_map(|index| rows[*index].factor)
        .collect::<Vec<_>>();
    let avg = if demeaned {
        mean(&values).unwrap_or(0.0)
    } else {
        0.0
    };
    let adjusted = indexes
        .iter()
        .map(|index| rows[*index].factor.map(|value| value - avg))
        .collect::<Vec<_>>();
    gross_normalize(adjusted)
}

fn equal_weights(
    indexes: &[usize],
    rows: &[crate::performance::frame::FactorRow],
    demeaned: bool,
) -> Vec<Option<f64>> {
    if !demeaned {
        let raw = indexes
            .iter()
            .map(|index| {
                rows[*index].factor.map(|value| {
                    if value > 0.0 {
                        1.0
                    } else if value < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        return gross_normalize(raw);
    }

    let mut finite = indexes
        .iter()
        .filter_map(|index| rows[*index].factor)
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return vec![None; indexes.len()];
    }
    finite.sort_by(f64::total_cmp);
    let median = if finite.len() % 2 == 0 {
        (finite[finite.len() / 2 - 1] + finite[finite.len() / 2]) / 2.0
    } else {
        finite[finite.len() / 2]
    };
    let longs = indexes
        .iter()
        .filter(|index| rows[**index].factor.is_some_and(|value| value > median))
        .count();
    let shorts = indexes
        .iter()
        .filter(|index| rows[**index].factor.is_some_and(|value| value < median))
        .count();
    let raw = indexes
        .iter()
        .map(|index| {
            rows[*index].factor.map(|value| {
                if value > median && longs > 0 {
                    0.5 / longs as f64
                } else if value < median && shorts > 0 {
                    -0.5 / shorts as f64
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    gross_normalize(raw)
}

fn gross_normalize(values: Vec<Option<f64>>) -> Vec<Option<f64>> {
    let gross = values
        .iter()
        .filter_map(|value| *value)
        .map(f64::abs)
        .sum::<f64>();
    if gross == 0.0 {
        return values.into_iter().map(|_| None).collect();
    }
    values
        .into_iter()
        .map(|value| value.map(|value| value / gross))
        .collect()
}
