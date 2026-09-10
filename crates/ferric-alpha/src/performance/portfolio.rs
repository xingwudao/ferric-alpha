use std::collections::{BTreeMap, BTreeSet, HashSet};

use polars::prelude::{
    BooleanChunked, DataFrame, DataType, NamedFrom, NewChunkedArray, Series, SortMultipleOptions,
};

use crate::Period;
use crate::data::required_column;
use crate::error::{FerricAlphaError, Result};
use crate::performance::frame::{date_column, finish_frame};
use crate::performance::{PortfolioOptions, ReturnOptions};
use crate::{factor_returns, factor_weights};

#[derive(Debug)]
struct ActiveSleeve {
    expiry_index: usize,
    weights: BTreeMap<String, f64>,
}

pub fn cumulative_returns(returns: &DataFrame) -> Result<DataFrame> {
    let date = required_column(returns, "date")?;
    let values = required_column(returns, "return")?;
    if !matches!(date.dtype(), DataType::Datetime(_, _)) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "date",
            expected: "Datetime",
            actual: date.dtype().to_string(),
        });
    }
    if !matches!(values.dtype(), DataType::Float64) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "return",
            expected: "Float64",
            actual: values.dtype().to_string(),
        });
    }
    let has_period = returns.column("period").is_ok();
    if has_period && !matches!(returns.column("period")?.dtype(), DataType::String) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "period",
            expected: "String",
            actual: returns.column("period")?.dtype().to_string(),
        });
    }
    if date.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let sort = if has_period {
        vec!["period", "date"]
    } else {
        vec!["date"]
    };
    let sorted = returns.sort(sort.clone(), SortMultipleOptions::default())?;
    let dates = sorted.column("date")?.as_materialized_series().datetime()?;
    let values = sorted.column("return")?.as_materialized_series().f64()?;
    let periods = sorted
        .column("period")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;

    let mut seen = HashSet::with_capacity(sorted.height());
    let mut output_dates = Vec::with_capacity(sorted.height());
    let mut output_periods = Vec::with_capacity(sorted.height());
    let mut output_wealth = Vec::with_capacity(sorted.height());
    let mut active_period: Option<String> = None;
    let mut wealth = 1.0;

    for index in 0..sorted.height() {
        let date = dates
            .physical()
            .get(index)
            .ok_or(FerricAlphaError::NullKey)?;
        let period = periods
            .as_ref()
            .map(|periods| {
                periods
                    .get(index)
                    .map(ToOwned::to_owned)
                    .ok_or(FerricAlphaError::InvalidInput {
                        operation: "cumulative_returns",
                        reason: "period must not contain null values",
                    })
            })
            .transpose()?;
        if let Some(period) = &period {
            validate_period_label(period)?;
        }
        if !seen.insert((date, period.clone())) {
            return Err(FerricAlphaError::InvalidInput {
                operation: "cumulative_returns",
                reason: "return series keys must be unique",
            });
        }
        if period != active_period {
            active_period.clone_from(&period);
            wealth = 1.0;
        }
        let value = values
            .get(index)
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        wealth *= 1.0 + value;
        if !wealth.is_finite() {
            return Err(FerricAlphaError::InvalidInput {
                operation: "cumulative_returns",
                reason: "cumulative wealth must remain finite",
            });
        }
        output_dates.push(date);
        output_periods.push(period);
        output_wealth.push(wealth);
    }

    let mut columns = vec![date_column("date", output_dates, date.dtype())?];
    if has_period {
        columns.push(Series::new("period".into(), output_periods).into());
    }
    columns.push(Series::new("cumulative_return".into(), output_wealth).into());
    let frame = DataFrame::new(columns[0].len(), columns)?;
    finish_frame(frame, &sort)
}

fn validate_period_label(period: &str) -> Result<()> {
    let valid = period
        .strip_suffix('D')
        .filter(|number| !number.is_empty() && !number.starts_with('0'))
        .and_then(|number| number.parse::<u32>().ok())
        .is_some_and(|number| number > 0);
    if valid {
        Ok(())
    } else {
        Err(FerricAlphaError::InvalidInput {
            operation: "cumulative_returns",
            reason: "period must be a positive daily count",
        })
    }
}

pub fn positions(
    weights: &DataFrame,
    period: Period,
    sessions: Option<&DataFrame>,
) -> Result<DataFrame> {
    let date = required_column(weights, "date")?;
    let asset = required_column(weights, "asset")?;
    let weight = required_column(weights, "weight")?;
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
    if !matches!(weight.dtype(), DataType::Float64) {
        return Err(FerricAlphaError::InvalidDtype {
            column: "weight",
            expected: "Float64",
            actual: weight.dtype().to_string(),
        });
    }
    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let values = weight.as_materialized_series().f64()?;
    let mut seen = HashSet::with_capacity(weights.height());
    let mut trades: BTreeMap<i64, BTreeMap<String, f64>> = BTreeMap::new();
    let mut asset_universe = BTreeSet::new();
    for ((date, asset), value) in dates
        .physical()
        .iter()
        .zip(assets.iter())
        .zip(values.iter())
    {
        let (Some(date), Some(asset), Some(value)) = (date, asset, value) else {
            return Err(FerricAlphaError::InvalidInput {
                operation: "positions",
                reason: "weights must be finite and non-null",
            });
        };
        if !value.is_finite() {
            return Err(FerricAlphaError::InvalidInput {
                operation: "positions",
                reason: "weights must be finite and non-null",
            });
        }
        if asset == "cash" {
            return Err(FerricAlphaError::InvalidInput {
                operation: "positions",
                reason: "asset name cash is reserved for synthetic cash",
            });
        }
        if !seen.insert((date, asset)) {
            return Err(FerricAlphaError::DuplicateKey);
        }
        asset_universe.insert(asset.to_owned());
        trades
            .entry(date)
            .or_default()
            .insert(asset.to_owned(), value);
    }

    let session_values = validate_sessions(sessions, date.dtype(), trades.keys().copied())?;
    let session_indexes = session_values
        .iter()
        .enumerate()
        .map(|(index, date)| (*date, index))
        .collect::<BTreeMap<_, _>>();
    if trades
        .keys()
        .any(|date| !session_indexes.contains_key(date))
    {
        return Err(FerricAlphaError::InvalidInput {
            operation: "positions",
            reason: "every trade date must be present in sessions",
        });
    }

    let assets = asset_universe.into_iter().collect::<Vec<_>>();
    let row_count = session_values.len() * (assets.len() + 1);
    let mut output_dates = Vec::with_capacity(row_count);
    let mut output_assets = Vec::with_capacity(row_count);
    let mut output_positions = Vec::with_capacity(row_count);
    let mut active = Vec::<ActiveSleeve>::new();

    for (session_index, session_date) in session_values.iter().enumerate() {
        active.retain(|sleeve| sleeve.expiry_index > session_index);
        if let Some(weights) = trades.get(session_date) {
            active.push(ActiveSleeve {
                expiry_index: session_index.saturating_add(period.count() as usize),
                weights: weights.clone(),
            });
        }

        let mut totals = BTreeMap::<String, f64>::new();
        for sleeve in &active {
            for (asset, weight) in &sleeve.weights {
                let total = totals.entry(asset.clone()).or_default();
                let updated = *total + weight;
                if !updated.is_finite() {
                    return Err(FerricAlphaError::InvalidInput {
                        operation: "positions",
                        reason: "active sleeve weights must sum to finite values",
                    });
                }
                *total = updated;
            }
        }
        let mut gross = 0.0;
        for value in totals.values() {
            gross += value.abs();
            if !gross.is_finite() {
                return Err(FerricAlphaError::InvalidInput {
                    operation: "positions",
                    reason: "gross exposure must remain finite",
                });
            }
        }
        let mut net = 0.0;
        for asset in &assets {
            let position = if gross > 0.0 {
                totals.get(asset).copied().unwrap_or(0.0) / gross
            } else {
                0.0
            };
            if !position.is_finite() {
                return Err(FerricAlphaError::InvalidInput {
                    operation: "positions",
                    reason: "normalized positions must remain finite",
                });
            }
            output_dates.push(*session_date);
            output_assets.push(asset.clone());
            output_positions.push(position);
            net += position;
            if !net.is_finite() {
                return Err(FerricAlphaError::InvalidInput {
                    operation: "positions",
                    reason: "net exposure must remain finite",
                });
            }
        }
        let cash = 1.0 - net;
        if !cash.is_finite() {
            return Err(FerricAlphaError::InvalidInput {
                operation: "positions",
                reason: "synthetic cash position must remain finite",
            });
        }
        output_dates.push(*session_date);
        output_assets.push("cash".to_owned());
        output_positions.push(cash);
    }

    let frame = DataFrame::new(
        output_dates.len(),
        vec![
            date_column("date", output_dates, date.dtype())?,
            Series::new("asset".into(), output_assets).into(),
            Series::new("position".into(), output_positions).into(),
        ],
    )?;
    finish_frame(frame, &["date", "asset"])
}

fn validate_sessions(
    sessions: Option<&DataFrame>,
    date_dtype: &DataType,
    inferred: impl Iterator<Item = i64>,
) -> Result<Vec<i64>> {
    let Some(sessions) = sessions else {
        return Ok(inferred.collect());
    };
    if sessions.width() != 1 {
        return Err(FerricAlphaError::InvalidInput {
            operation: "positions",
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
    let mut values = date
        .as_materialized_series()
        .datetime()?
        .physical()
        .iter()
        .map(|value| value.expect("validated session date"))
        .collect::<Vec<_>>();
    values.sort_unstable();
    if values.windows(2).any(|window| window[0] == window[1]) {
        return Err(FerricAlphaError::DuplicateSession);
    }
    Ok(values)
}

pub fn factor_cumulative_returns(
    factor_data: &DataFrame,
    period: Period,
    options: PortfolioOptions,
) -> Result<DataFrame> {
    let filtered = filter_factor_data(factor_data, &options)?;
    ensure_requested_period(&filtered, period)?;
    let returns = factor_returns(
        &filtered,
        ReturnOptions::default().with_weight_options(options.weight_options),
    )?;
    let period_label = period.label();
    let periods = returns.column("period")?.as_materialized_series().str()?;
    let mask = BooleanChunked::from_iter_values(
        "period_filter".into(),
        periods.iter().map(|value| value == Some(&period_label)),
    );
    let selected = returns.filter(&mask)?;
    let adapted = DataFrame::new(
        selected.height(),
        vec![
            selected.column("date")?.clone(),
            selected
                .column("factor_return")?
                .clone()
                .with_name("return".into()),
        ],
    )?;
    cumulative_returns(&adapted)
}

pub fn factor_positions(
    factor_data: &DataFrame,
    period: Period,
    options: PortfolioOptions,
    sessions: Option<&DataFrame>,
) -> Result<DataFrame> {
    let filtered = filter_factor_data(factor_data, &options)?;
    ensure_requested_period(&filtered, period)?;
    let weights = factor_weights(&filtered, options.weight_options)?;
    positions(&weights, period, sessions)
}

fn ensure_requested_period(frame: &DataFrame, period: Period) -> Result<()> {
    let column = format!("forward_return_{}", period.label());
    if frame.column(&column).is_ok() {
        Ok(())
    } else {
        Err(FerricAlphaError::InvalidOption {
            option: "period",
            reason: "requested forward return period is not present",
        })
    }
}

pub(crate) fn filter_factor_data(
    frame: &DataFrame,
    options: &PortfolioOptions,
) -> Result<DataFrame> {
    let require_quantile = options.quantiles.is_some();
    let require_group = options.groups.is_some();
    crate::performance::frame::validate_factor_data(frame, require_quantile, require_group, true)?;

    let selected_quantiles = validate_quantile_filter(options.quantiles.as_deref())?;
    let selected_groups = validate_group_filter(options.groups.as_deref())?;
    let quantiles = frame
        .column("factor_quantile")
        .ok()
        .map(|column| column.as_materialized_series().u32())
        .transpose()?;
    let groups = frame
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;
    let mask = BooleanChunked::from_iter_values(
        "portfolio_filter".into(),
        (0..frame.height()).map(|index| {
            let quantile_matches = selected_quantiles.as_ref().is_none_or(|selected| {
                quantiles
                    .as_ref()
                    .and_then(|quantiles| quantiles.get(index))
                    .is_some_and(|quantile| selected.contains(&quantile))
            });
            let group_matches = selected_groups.as_ref().is_none_or(|selected| {
                groups
                    .as_ref()
                    .and_then(|groups| groups.get(index))
                    .is_some_and(|group| selected.contains(group))
            });
            quantile_matches && group_matches
        }),
    );
    let filtered = frame.filter(&mask)?;
    if filtered.height() == 0 {
        return Err(FerricAlphaError::InsufficientData {
            operation: "portfolio filter",
        });
    }
    Ok(filtered)
}

fn validate_quantile_filter(values: Option<&[u32]>) -> Result<Option<HashSet<u32>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    let selected = values.iter().copied().collect::<HashSet<_>>();
    if values.is_empty() || selected.len() != values.len() || selected.contains(&0) {
        return Err(FerricAlphaError::InvalidOption {
            option: "quantiles",
            reason: "quantiles must be positive and unique",
        });
    }
    Ok(Some(selected))
}

fn validate_group_filter(values: Option<&[String]>) -> Result<Option<HashSet<String>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    let selected = values.iter().cloned().collect::<HashSet<_>>();
    if values.is_empty()
        || selected.len() != values.len()
        || selected.iter().any(|value| value.is_empty())
    {
        return Err(FerricAlphaError::InvalidOption {
            option: "groups",
            reason: "groups must be non-empty and unique",
        });
    }
    Ok(Some(selected))
}
