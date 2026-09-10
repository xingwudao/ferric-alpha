use std::collections::{BTreeMap, BTreeSet};

use polars::prelude::{DataFrame, DataType, NamedFrom, Series, SortMultipleOptions};

use crate::data::required_column;
use crate::error::{FerricAlphaError, Result};
use crate::performance::frame::date_column;
use crate::performance::portfolio::filter_factor_data;
use crate::performance::{PortfolioOptions, WeightOptions};
use crate::{Period, factor_cumulative_returns, factor_positions, factor_returns};

#[derive(Debug, Clone)]
pub struct PyfolioInput {
    pub returns: DataFrame,
    pub positions: DataFrame,
    pub benchmark: Option<DataFrame>,
}

pub fn create_pyfolio_input(
    factor_data: &DataFrame,
    period: Period,
    capital: Option<f64>,
    options: PortfolioOptions,
    benchmark_period: Option<Period>,
    sessions: Option<&DataFrame>,
) -> Result<PyfolioInput> {
    if capital.is_some_and(|capital| !capital.is_finite() || capital <= 0.0) {
        return Err(FerricAlphaError::InvalidOption {
            option: "capital",
            reason: "capital must be finite and positive",
        });
    }

    let wealth = if period.count() == 1 {
        factor_cumulative_returns(factor_data, period, options.clone())?
    } else {
        overlapping_daily_wealth(factor_data, period, options.clone(), sessions)?
    };
    let returns = changes_from_wealth(&wealth, "return")?;
    let mut positions = factor_positions(factor_data, period, options, sessions)?;
    if let Some(capital) = capital {
        positions = scale_positions(&positions, &wealth, capital)?;
    }

    let benchmark = benchmark_period
        .filter(|period| {
            factor_data
                .column(&format!("forward_return_{}", period.label()))
                .is_ok()
        })
        .map(|period| {
            let benchmark_options = PortfolioOptions::default().with_weight_options(
                WeightOptions::default()
                    .with_demeaned(false)
                    .with_group_adjust(false)
                    .with_equal_weight(true),
            );
            let benchmark_wealth =
                factor_cumulative_returns(factor_data, period, benchmark_options)?;
            changes_from_wealth(&benchmark_wealth, "benchmark")
        })
        .transpose()?;

    Ok(PyfolioInput {
        returns,
        positions,
        benchmark,
    })
}

#[derive(Debug)]
struct ActiveReturnSleeve {
    expiry_index: usize,
    daily_return: f64,
}

fn overlapping_daily_wealth(
    factor_data: &DataFrame,
    period: Period,
    options: PortfolioOptions,
    sessions: Option<&DataFrame>,
) -> Result<DataFrame> {
    let filtered = filter_factor_data(factor_data, &options)?;
    let returns = factor_returns(
        &filtered,
        crate::performance::ReturnOptions::default().with_weight_options(options.weight_options),
    )?;
    let period_label = period.label();
    let dates = returns
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let periods = returns.column("period")?.as_materialized_series().str()?;
    let values = returns
        .column("factor_return")?
        .as_materialized_series()
        .f64()?;
    let date_dtype = returns.column("date")?.dtype().clone();
    let mut returns_by_date = BTreeMap::new();
    for index in 0..returns.height() {
        if periods.get(index) != Some(period_label.as_str()) {
            continue;
        }
        let date = dates
            .physical()
            .get(index)
            .ok_or(FerricAlphaError::NullKey)?;
        if returns_by_date
            .insert(date, values.get(index).filter(|value| value.is_finite()))
            .is_some()
        {
            return Err(FerricAlphaError::DuplicateKey);
        }
    }
    let session_values = pyfolio_sessions(sessions, &date_dtype, returns_by_date.keys().copied())?;
    let mut active = Vec::<ActiveReturnSleeve>::new();
    let mut output_dates = Vec::with_capacity(session_values.len());
    let mut output_wealth = Vec::with_capacity(session_values.len());
    let mut wealth = 1.0;

    for (session_index, session_date) in session_values.iter().enumerate() {
        active.retain(|sleeve| sleeve.expiry_index > session_index);
        if let Some(Some(forward_return)) = returns_by_date.get(session_date).copied() {
            let base = 1.0 + forward_return;
            if !base.is_finite() || base < 0.0 {
                return Err(FerricAlphaError::InvalidInput {
                    operation: "create_pyfolio_input",
                    reason: "forward return cannot be converted to daily sleeve return",
                });
            }
            active.push(ActiveReturnSleeve {
                expiry_index: session_index.saturating_add(period.count() as usize),
                daily_return: base.powf(1.0 / period.count() as f64) - 1.0,
            });
        }

        output_dates.push(*session_date);
        output_wealth.push(wealth);
        if !active.is_empty() {
            let daily_return =
                active.iter().map(|sleeve| sleeve.daily_return).sum::<f64>() / active.len() as f64;
            wealth *= 1.0 + daily_return;
            if !wealth.is_finite() {
                return Err(FerricAlphaError::InvalidInput {
                    operation: "create_pyfolio_input",
                    reason: "portfolio wealth must remain finite",
                });
            }
        }
    }

    let frame = DataFrame::new(
        output_dates.len(),
        vec![
            date_column("date", output_dates, &date_dtype)?,
            Series::new("cumulative_return".into(), output_wealth).into(),
        ],
    )?;
    Ok(frame.sort(["date"], SortMultipleOptions::default())?)
}

fn pyfolio_sessions(
    sessions: Option<&DataFrame>,
    date_dtype: &DataType,
    inferred: impl Iterator<Item = i64>,
) -> Result<Vec<i64>> {
    let Some(sessions) = sessions else {
        return Ok(inferred.collect::<BTreeSet<_>>().into_iter().collect());
    };
    if sessions.width() != 1 {
        return Err(FerricAlphaError::InvalidInput {
            operation: "create_pyfolio_input",
            reason: "sessions must contain exactly one date column",
        });
    }
    let date = required_column(sessions, "date")?;
    if !matches!(date.dtype(), DataType::Datetime(_, _)) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "date",
            expected: "Datetime",
            actual: date.dtype().to_string(),
        });
    }
    if date.dtype() != date_dtype {
        return Err(FerricAlphaError::TimezoneMismatch);
    }
    if date.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }
    let values = date
        .as_materialized_series()
        .datetime()?
        .physical()
        .iter()
        .map(|value| value.expect("validated session date"))
        .collect::<BTreeSet<_>>();
    Ok(values.into_iter().collect())
}

fn changes_from_wealth(wealth: &DataFrame, name: &str) -> Result<DataFrame> {
    let values = wealth
        .column("cumulative_return")?
        .as_materialized_series()
        .f64()?;
    let mut returns = Vec::with_capacity(wealth.height());
    let mut previous: Option<f64> = None;
    for value in values.iter() {
        let value = value.ok_or(FerricAlphaError::InvalidInput {
            operation: "create_pyfolio_input",
            reason: "cumulative wealth must not contain null values",
        })?;
        let change = match previous {
            None => 0.0,
            Some(previous) if previous != 0.0 => value / previous - 1.0,
            Some(_) => {
                return Err(FerricAlphaError::InvalidInput {
                    operation: "create_pyfolio_input",
                    reason: "cannot compute return after zero wealth",
                });
            }
        };
        if !change.is_finite() {
            return Err(FerricAlphaError::InvalidInput {
                operation: "create_pyfolio_input",
                reason: "portfolio returns must remain finite",
            });
        }
        returns.push(change);
        previous = Some(value);
    }
    Ok(DataFrame::new(
        wealth.height(),
        vec![
            wealth.column("date")?.clone(),
            Series::new(name.into(), returns).into(),
        ],
    )?)
}

fn scale_positions(positions: &DataFrame, wealth: &DataFrame, capital: f64) -> Result<DataFrame> {
    let wealth_dates = wealth.column("date")?.as_materialized_series().datetime()?;
    let wealth_values = wealth
        .column("cumulative_return")?
        .as_materialized_series()
        .f64()?;
    let wealth_by_date = wealth_dates
        .physical()
        .iter()
        .zip(wealth_values.iter())
        .map(|(date, value)| {
            Ok((
                date.ok_or(FerricAlphaError::NullKey)?,
                value.ok_or(FerricAlphaError::InvalidInput {
                    operation: "create_pyfolio_input",
                    reason: "cumulative wealth must not contain null values",
                })?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let position_dates = positions
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let position_values = positions
        .column("position")?
        .as_materialized_series()
        .f64()?;
    let scaled = position_dates
        .physical()
        .iter()
        .zip(position_values.iter())
        .map(|(date, position)| {
            let date = date.ok_or(FerricAlphaError::NullKey)?;
            let wealth = wealth_by_date
                .range(..=date)
                .next_back()
                .map(|(_, wealth)| *wealth)
                .unwrap_or(1.0);
            let scaled = position.ok_or(FerricAlphaError::InvalidInput {
                operation: "create_pyfolio_input",
                reason: "positions must not contain null values",
            })? * wealth
                * capital;
            if scaled.is_finite() {
                Ok(scaled)
            } else {
                Err(FerricAlphaError::InvalidInput {
                    operation: "create_pyfolio_input",
                    reason: "capital positions must remain finite",
                })
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DataFrame::new(
        positions.height(),
        vec![
            positions.column("date")?.clone(),
            positions.column("asset")?.clone(),
            Series::new("position".into(), scaled).into(),
        ],
    )?)
}
