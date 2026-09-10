use std::collections::{BTreeMap, HashMap, HashSet};

use polars::prelude::{DataFrame, DataType, NamedFrom, Series};

use crate::error::{FerricAlphaError, Result};
use crate::performance::frame::{ensure_dtype, finish_frame, mean, validate_factor_data};
use crate::performance::{AlphaBetaOptions, ReturnOptions};
use crate::{factor_returns, simple_ols};

pub fn factor_alpha_beta(
    factor_data: &DataFrame,
    returns: Option<&DataFrame>,
    options: AlphaBetaOptions,
) -> Result<DataFrame> {
    let data = validate_factor_data(
        factor_data,
        false,
        options.weight_options.group_adjust,
        true,
    )?;
    let return_frame = match returns {
        Some(returns) => validate_returns(returns, &data.periods, &data.date_dtype)?,
        None => factor_returns(
            factor_data,
            ReturnOptions::default().with_weight_options(options.weight_options),
        )?,
    };
    let factor_lookup = factor_return_lookup(&return_frame)?;
    let mut universe: BTreeMap<(String, i64), f64> = BTreeMap::new();
    let mut by_date: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (index, row) in data.rows.iter().enumerate() {
        by_date.entry(row.date).or_default().push(index);
    }
    for (period_index, period) in data.periods.iter().enumerate() {
        for (date, indexes) in &by_date {
            let values = indexes
                .iter()
                .filter_map(|index| data.rows[*index].returns[period_index])
                .collect::<Vec<_>>();
            if let Some(avg) = mean(&values) {
                universe.insert((period.label.clone(), *date), avg);
            }
        }
    }

    let mut out_periods = Vec::new();
    let mut out_alpha = Vec::new();
    let mut out_annual = Vec::new();
    let mut out_beta = Vec::new();
    let mut out_count = Vec::new();
    for period in &data.periods {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for ((label, date), market_return) in &universe {
            if label != &period.label {
                continue;
            }
            if let Some(factor_return) = factor_lookup.get(&(period.label.clone(), *date)) {
                x.push(*market_return);
                y.push(*factor_return);
            }
        }
        let fit = simple_ols(&x, &y)?;
        out_periods.push(period.label.clone());
        out_count.push(fit.count as u32);
        out_alpha.push(fit.alpha);
        out_beta.push(fit.beta);
        out_annual.push(fit.alpha.and_then(|alpha| {
            (1.0 + alpha > 0.0)
                .then(|| (1.0 + alpha).powf(252.0 / period.observations as f64) - 1.0)
        }));
    }

    finish_frame(
        DataFrame::new(
            out_periods.len(),
            vec![
                Series::new("period".into(), out_periods).into(),
                Series::new("alpha".into(), out_alpha).into(),
                Series::new("annualized_alpha".into(), out_annual).into(),
                Series::new("beta".into(), out_beta).into(),
                Series::new("count".into(), out_count).into(),
            ],
        )?,
        &["period"],
    )
}

fn validate_returns(
    returns: &DataFrame,
    periods: &[crate::performance::frame::ForwardPeriod],
    factor_date_dtype: &DataType,
) -> Result<DataFrame> {
    let date = returns
        .column("date")
        .map_err(|_| FerricAlphaError::MissingColumn("date"))?;
    ensure_dtype("date", date.dtype(), "Datetime", |dtype| {
        matches!(dtype, DataType::Datetime(_, _))
    })?;
    if date.dtype() != factor_date_dtype {
        return Err(FerricAlphaError::TimezoneMismatch);
    }
    let period = returns
        .column("period")
        .map_err(|_| FerricAlphaError::MissingColumn("period"))?;
    ensure_dtype("period", period.dtype(), "String", |dtype| {
        matches!(dtype, DataType::String)
    })?;
    let factor_return = returns
        .column("factor_return")
        .map_err(|_| FerricAlphaError::MissingColumn("factor_return"))?;
    ensure_dtype("factor_return", factor_return.dtype(), "Float64", |dtype| {
        matches!(dtype, DataType::Float64)
    })?;

    let allowed = periods
        .iter()
        .map(|period| period.label.as_str())
        .collect::<HashSet<_>>();
    let dates = date.as_materialized_series().datetime()?;
    let labels = period.as_materialized_series().str()?;
    let mut keys = HashSet::new();
    for index in 0..returns.height() {
        let date = dates
            .physical()
            .get(index)
            .ok_or(FerricAlphaError::NullKey)?;
        let label = labels.get(index).ok_or(FerricAlphaError::InvalidInput {
            operation: "factor_alpha_beta",
            reason: "period must not be null",
        })?;
        if !allowed.contains(label) {
            return Err(FerricAlphaError::InvalidInput {
                operation: "factor_alpha_beta",
                reason: "period must exist in factor data",
            });
        }
        if !keys.insert((date, label.to_owned())) {
            return Err(FerricAlphaError::DuplicateKey);
        }
    }
    Ok(returns.clone())
}

fn factor_return_lookup(returns: &DataFrame) -> Result<HashMap<(String, i64), f64>> {
    let dates = returns
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let periods = returns.column("period")?.as_materialized_series().str()?;
    let values = returns
        .column("factor_return")?
        .as_materialized_series()
        .f64()?;
    let mut lookup = HashMap::new();
    for index in 0..returns.height() {
        if let (Some(date), Some(period), Some(value)) = (
            dates.physical().get(index),
            periods.get(index),
            values.get(index).filter(|value| value.is_finite()),
        ) {
            lookup.insert((period.to_owned(), date), value);
        }
    }
    Ok(lookup)
}
