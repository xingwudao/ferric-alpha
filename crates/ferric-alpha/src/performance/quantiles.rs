use std::collections::{BTreeMap, HashMap, HashSet};

use polars::prelude::{DataFrame, DataType, NamedFrom, Series};

use crate::error::{FerricAlphaError, Result};
use crate::performance::MeanReturnByQuantileOptions;
use crate::performance::frame::{
    date_column, ensure_dtype, finish_frame, mean, validate_factor_data,
};
use crate::std_error;

type QuantKey = (u32, Option<i64>, Option<String>, String);

pub fn mean_return_by_quantile(
    factor_data: &DataFrame,
    options: MeanReturnByQuantileOptions,
) -> Result<DataFrame> {
    let data = validate_factor_data(
        factor_data,
        true,
        options.by_group || options.group_adjust,
        true,
    )?;
    let mut adjusted = vec![vec![None; data.periods.len()]; data.rows.len()];
    for (period_index, _) in data.periods.iter().enumerate() {
        let mut groups: BTreeMap<(i64, Option<String>), Vec<usize>> = BTreeMap::new();
        for (index, row) in data.rows.iter().enumerate() {
            let group = options
                .group_adjust
                .then(|| row.group.clone().expect("validated group"));
            groups.entry((row.date, group)).or_default().push(index);
        }
        for indexes in groups.values() {
            let values = indexes
                .iter()
                .filter_map(|index| data.rows[*index].returns[period_index])
                .collect::<Vec<_>>();
            let avg = if options.group_adjust || options.demeaned {
                mean(&values).unwrap_or(0.0)
            } else {
                0.0
            };
            for index in indexes {
                adjusted[*index][period_index] =
                    data.rows[*index].returns[period_index].map(|value| value - avg);
            }
        }
    }

    let mut stage: BTreeMap<QuantKey, Vec<f64>> = BTreeMap::new();
    for (index, row) in data.rows.iter().enumerate() {
        for (period_index, period) in data.periods.iter().enumerate() {
            if let Some(value) = adjusted[index][period_index] {
                let group = options
                    .by_group
                    .then(|| row.group.clone().expect("validated group"));
                stage
                    .entry((
                        row.quantile.expect("validated quantile"),
                        Some(row.date),
                        group,
                        period.label.clone(),
                    ))
                    .or_default()
                    .push(value);
            }
        }
    }

    let mut final_groups: BTreeMap<QuantKey, Vec<f64>> = BTreeMap::new();
    if options.by_date {
        final_groups = stage;
    } else {
        for ((quantile, _date, group, period), values) in stage {
            if let Some(avg) = mean(&values) {
                final_groups
                    .entry((quantile, None, group, period))
                    .or_default()
                    .push(avg);
            }
        }
    }

    let mut out_quantiles = Vec::new();
    let mut out_dates = Vec::new();
    let mut out_groups = Vec::new();
    let mut out_periods = Vec::new();
    let mut out_means = Vec::new();
    let mut out_errors = Vec::new();
    let mut out_counts = Vec::new();
    for ((quantile, date, group, period), values) in final_groups {
        out_quantiles.push(quantile);
        if options.by_date {
            out_dates.push(date.expect("date retained by by_date"));
        }
        if options.by_group {
            out_groups.push(group);
        }
        out_periods.push(period);
        out_means.push(mean(&values));
        out_errors.push(std_error(&values)?);
        out_counts.push(values.len() as u32);
    }

    let mut columns = vec![Series::new("factor_quantile".into(), out_quantiles).into()];
    let mut sort = vec!["factor_quantile"];
    if options.by_date {
        columns.push(date_column("date", out_dates, &data.date_dtype)?);
        sort = vec!["date", "factor_quantile"];
    }
    if options.by_group {
        columns.push(Series::new("group".into(), out_groups).into());
        sort.push("group");
    }
    columns.push(Series::new("period".into(), out_periods).into());
    columns.push(Series::new("mean_return".into(), out_means).into());
    columns.push(Series::new("std_error".into(), out_errors).into());
    columns.push(Series::new("count".into(), out_counts).into());
    sort.push("period");
    finish_frame(DataFrame::new(columns[0].len(), columns)?, &sort)
}

pub fn compute_mean_returns_spread(
    mean_returns: &DataFrame,
    upper_quant: u32,
    lower_quant: u32,
    std_err: Option<&DataFrame>,
) -> Result<DataFrame> {
    if upper_quant == lower_quant {
        return Err(FerricAlphaError::InvalidInput {
            operation: "compute_mean_returns_spread",
            reason: "upper and lower quantiles must differ",
        });
    }
    validate_spread_frame(mean_returns, "mean_return")?;
    if let Some(std_err) = std_err {
        validate_spread_frame(std_err, "std_error")?;
        if grouping_schema(mean_returns) != grouping_schema(std_err) {
            return Err(FerricAlphaError::InvalidInput {
                operation: "compute_mean_returns_spread",
                reason: "std_err grouping schema must match mean_returns",
            });
        }
    }

    let mean_map = spread_value_map(mean_returns, "mean_return")?;
    let error_frame = std_err.unwrap_or(mean_returns);
    let error_map = if error_frame.column("std_error").is_ok() {
        Some(spread_value_map(error_frame, "std_error")?)
    } else {
        None
    };

    let mut keys = HashSet::new();
    for key in mean_map.keys() {
        if key.quantile == upper_quant || key.quantile == lower_quant {
            keys.insert((key.date, key.group.clone(), key.period.clone()));
        }
    }

    let mut dates = Vec::new();
    let mut has_date = false;
    let mut groups = Vec::new();
    let mut has_group = false;
    let mut periods = Vec::new();
    let mut diffs = Vec::new();
    let mut errors = Vec::new();
    for (date, group, period) in keys {
        let upper = SpreadKey {
            quantile: upper_quant,
            date,
            group: group.clone(),
            period: period.clone(),
        };
        let lower = SpreadKey {
            quantile: lower_quant,
            date,
            group: group.clone(),
            period: period.clone(),
        };
        let (Some(upper_value), Some(lower_value)) = (mean_map.get(&upper), mean_map.get(&lower))
        else {
            return Err(FerricAlphaError::InvalidInput {
                operation: "compute_mean_returns_spread",
                reason: "upper and lower quantile keys must both exist",
            });
        };
        if let Some(date) = date {
            has_date = true;
            dates.push(date);
        }
        if group.is_some() {
            has_group = true;
            groups.push(group);
        }
        periods.push(period.clone());
        diffs.push(Some(upper_value - lower_value));
        let joint = error_map.as_ref().and_then(|errors| {
            let upper_error = errors.get(&upper)?;
            let lower_error = errors.get(&lower)?;
            Some((upper_error.powi(2) + lower_error.powi(2)).sqrt())
        });
        errors.push(joint);
    }

    let date_dtype = mean_returns
        .column("date")
        .ok()
        .map(|column| column.dtype().clone());
    let mut columns = Vec::new();
    let mut sort = Vec::new();
    if has_date {
        columns.push(date_column(
            "date",
            dates,
            date_dtype.as_ref().expect("date column present"),
        )?);
        sort.push("date");
    }
    if has_group {
        columns.push(Series::new("group".into(), groups).into());
        sort.push("group");
    }
    columns.push(Series::new("period".into(), periods).into());
    columns.push(Series::new("mean_return_difference".into(), diffs).into());
    columns.push(Series::new("joint_std_error".into(), errors).into());
    sort.push("period");
    finish_frame(DataFrame::new(columns[0].len(), columns)?, &sort)
}

fn grouping_schema(frame: &DataFrame) -> Vec<&'static str> {
    let mut keys = Vec::new();
    if frame.column("date").is_ok() {
        keys.push("date");
    }
    if frame.column("group").is_ok() {
        keys.push("group");
    }
    keys
}

fn validate_spread_frame(frame: &DataFrame, value_column: &'static str) -> Result<()> {
    ensure_dtype(
        "factor_quantile",
        frame
            .column("factor_quantile")
            .map_err(|_| FerricAlphaError::MissingColumn("factor_quantile"))?
            .dtype(),
        "UInt32",
        |dtype| matches!(dtype, DataType::UInt32),
    )?;
    ensure_dtype(
        "period",
        frame
            .column("period")
            .map_err(|_| FerricAlphaError::MissingColumn("period"))?
            .dtype(),
        "String",
        |dtype| matches!(dtype, DataType::String),
    )?;
    ensure_dtype(
        value_column,
        frame
            .column(value_column)
            .map_err(|_| FerricAlphaError::MissingColumn(value_column))?
            .dtype(),
        "Float64",
        |dtype| matches!(dtype, DataType::Float64),
    )?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SpreadKey {
    quantile: u32,
    date: Option<i64>,
    group: Option<String>,
    period: String,
}

fn spread_value_map(
    frame: &DataFrame,
    value_column: &'static str,
) -> Result<HashMap<SpreadKey, f64>> {
    let quantiles = frame
        .column("factor_quantile")?
        .as_materialized_series()
        .u32()?;
    let dates = frame
        .column("date")
        .ok()
        .map(|column| column.as_materialized_series().datetime())
        .transpose()?;
    let groups = frame
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;
    let periods = frame.column("period")?.as_materialized_series().str()?;
    let values = frame.column(value_column)?.as_materialized_series().f64()?;
    let mut map = HashMap::new();
    for index in 0..frame.height() {
        let key = SpreadKey {
            quantile: quantiles.get(index).ok_or(FerricAlphaError::InvalidInput {
                operation: "compute_mean_returns_spread",
                reason: "factor_quantile must not be null",
            })?,
            date: dates.as_ref().and_then(|dates| dates.physical().get(index)),
            group: groups
                .as_ref()
                .and_then(|groups| groups.get(index).map(ToOwned::to_owned)),
            period: periods
                .get(index)
                .ok_or(FerricAlphaError::InvalidInput {
                    operation: "compute_mean_returns_spread",
                    reason: "period must not be null",
                })?
                .to_owned(),
        };
        if let Some(value) = values.get(index).filter(|value| value.is_finite())
            && map.insert(key, value).is_some()
        {
            return Err(FerricAlphaError::DuplicateKey);
        }
    }
    Ok(map)
}
