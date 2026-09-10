use std::collections::HashMap;

use polars::prelude::{DataFrame, NamedFrom, Series, SortMultipleOptions};

use crate::calendar::{Period, validate_periods};
use crate::data::{ensure_matching_date_timezone, validate_factor_frame};
use crate::error::{FerricAlphaError, Result};
use crate::forward_returns::{
    ForwardReturnsOptions, compute_forward_returns_with_options, datetime_column,
};
use crate::quantiles::{QuantizeOptions, quantize_factor};

type ReturnKey = (i64, String);
type ReturnValues = Vec<Option<f64>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CleanOptions {
    pub max_loss: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossReport {
    pub input_rows: usize,
    pub output_rows: usize,
    pub forward_return_loss: usize,
    pub quantile_loss: usize,
}

#[derive(Debug)]
pub struct CleanFactorResult {
    pub frame: DataFrame,
    pub loss: LossReport,
}

pub fn get_clean_factor_and_forward_returns(
    factor: &DataFrame,
    prices: &DataFrame,
    periods: &[Period],
    quantize_options: QuantizeOptions,
    clean_options: CleanOptions,
) -> Result<CleanFactorResult> {
    get_clean_factor_and_forward_returns_with_options(
        factor,
        prices,
        periods,
        quantize_options,
        clean_options,
        ForwardReturnsOptions::default(),
    )
}

pub fn get_clean_factor_and_forward_returns_with_options(
    factor: &DataFrame,
    prices: &DataFrame,
    periods: &[Period],
    quantize_options: QuantizeOptions,
    clean_options: CleanOptions,
    forward_returns_options: ForwardReturnsOptions,
) -> Result<CleanFactorResult> {
    let periods = validate_periods(periods)?;
    let returns =
        compute_forward_returns_with_options(factor, prices, &periods, forward_returns_options)?;
    get_clean_factor(factor, &returns, quantize_options, clean_options)
}

pub fn get_clean_factor(
    factor: &DataFrame,
    forward_returns: &DataFrame,
    quantize_options: QuantizeOptions,
    clean_options: CleanOptions,
) -> Result<CleanFactorResult> {
    let factor = validate_factor_frame(factor)?;
    let date_dtype = ensure_matching_date_timezone(&factor, forward_returns)?;
    let quantized = quantize_factor(&factor, quantize_options)?;
    let return_columns = forward_return_columns(forward_returns);
    if return_columns.is_empty() {
        return Err(FerricAlphaError::MissingColumn("forward_return_*"));
    }

    let joined =
        join_quantized_and_returns(&quantized, forward_returns, &return_columns, &date_dtype)?;
    enforce_max_loss(&joined.loss, clean_options.max_loss)?;
    Ok(joined)
}

fn forward_return_columns(frame: &DataFrame) -> Vec<String> {
    let mut names = frame
        .get_column_names()
        .into_iter()
        .map(ToString::to_string)
        .filter(|name| name.starts_with("forward_return_"))
        .collect::<Vec<_>>();
    names.sort_by_key(|name| {
        (
            parse_forward_return_duration(name).unwrap_or(u64::MAX),
            name.clone(),
        )
    });
    names
}

fn parse_forward_return_duration(name: &str) -> Option<u64> {
    let label = name.strip_prefix("forward_return_")?;
    parse_duration_suffix(label)
}

fn parse_duration_suffix(label: &str) -> Option<u64> {
    for (suffix, multiplier) in [
        ("D", 86_400_000_000_u64),
        ("ms", 1_000),
        ("h", 3_600_000_000),
        ("m", 60_000_000),
        ("s", 1_000_000),
    ] {
        if let Some(value) = label.strip_suffix(suffix) {
            return value.parse::<u64>().ok().map(|value| value * multiplier);
        }
    }
    None
}

fn join_quantized_and_returns(
    quantized: &DataFrame,
    forward_returns: &DataFrame,
    return_columns: &[String],
    date_dtype: &polars::prelude::DataType,
) -> Result<CleanFactorResult> {
    let return_lookup = build_return_lookup(forward_returns, return_columns)?;
    let dates = quantized
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let assets = quantized.column("asset")?.as_materialized_series().str()?;
    let factors = quantized.column("factor")?.as_materialized_series().f64()?;
    let quantiles = quantized
        .column("factor_quantile")?
        .as_materialized_series()
        .u32()?;
    let groups = quantized
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;

    let mut output_dates = Vec::new();
    let mut output_assets = Vec::new();
    let mut output_factors = Vec::new();
    let mut output_groups = Vec::new();
    let mut output_quantiles = Vec::new();
    let mut output_returns = vec![Vec::new(); return_columns.len()];
    let mut forward_return_loss = 0;
    let mut quantile_loss = 0;

    for index in 0..quantized.height() {
        let date = dates
            .physical()
            .get(index)
            .expect("validated date should not be null");
        let asset = assets
            .get(index)
            .expect("validated asset should not be null");
        let factor = factors.get(index);
        let quantile = quantiles.get(index);
        let returns = return_lookup
            .get(&(date, asset.to_owned()))
            .cloned()
            .unwrap_or_else(|| vec![None; return_columns.len()]);

        let row_return_loss = returns.iter().any(Option::is_none);
        let row_quantile_loss = quantile.is_none();
        forward_return_loss += usize::from(row_return_loss);
        quantile_loss += usize::from(row_quantile_loss);

        if row_return_loss || row_quantile_loss {
            continue;
        }

        output_dates.push(date);
        output_assets.push(asset.to_owned());
        output_factors.push(factor);
        if let Some(groups) = groups {
            output_groups.push(groups.get(index).map(ToOwned::to_owned));
        }
        output_quantiles.push(quantile);
        for (return_index, value) in returns.into_iter().enumerate() {
            output_returns[return_index].push(value);
        }
    }

    let height = output_dates.len();
    let mut columns = vec![
        datetime_column("date", output_dates, date_dtype)?,
        Series::new("asset".into(), output_assets).into(),
        Series::new("factor".into(), output_factors).into(),
    ];
    if groups.is_some() {
        columns.push(Series::new("group".into(), output_groups).into());
    }
    columns.push(Series::new("factor_quantile".into(), output_quantiles).into());
    for (name, values) in return_columns.iter().zip(output_returns) {
        columns.push(Series::new(name.clone().into(), values).into());
    }

    let frame =
        DataFrame::new(height, columns)?.sort(["date", "asset"], SortMultipleOptions::default())?;
    Ok(CleanFactorResult {
        frame,
        loss: LossReport {
            input_rows: quantized.height(),
            output_rows: height,
            forward_return_loss,
            quantile_loss,
        },
    })
}

fn build_return_lookup(
    forward_returns: &DataFrame,
    return_columns: &[String],
) -> Result<HashMap<ReturnKey, ReturnValues>> {
    let dates = forward_returns
        .column("date")?
        .as_materialized_series()
        .datetime()?;
    let assets = forward_returns
        .column("asset")?
        .as_materialized_series()
        .str()?;
    let return_series = return_columns
        .iter()
        .map(|name| {
            Ok(forward_returns
                .column(name)?
                .as_materialized_series()
                .f64()?)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut lookup = HashMap::with_capacity(forward_returns.height());
    for index in 0..forward_returns.height() {
        let date = dates
            .physical()
            .get(index)
            .ok_or(FerricAlphaError::NullKey)?;
        let asset = assets.get(index).ok_or(FerricAlphaError::NullKey)?;
        let values = return_series
            .iter()
            .map(|series| series.get(index).filter(|value| value.is_finite()))
            .collect::<Vec<_>>();
        if lookup.insert((date, asset.to_owned()), values).is_some() {
            return Err(FerricAlphaError::DuplicateKey);
        }
    }
    Ok(lookup)
}

fn enforce_max_loss(loss: &LossReport, max_loss: f64) -> Result<()> {
    if !max_loss.is_finite() || !(0.0..=1.0).contains(&max_loss) {
        return Err(FerricAlphaError::InvalidMaxLoss { value: max_loss });
    }

    let actual = if loss.input_rows == 0 {
        0.0
    } else {
        (loss.input_rows - loss.output_rows) as f64 / loss.input_rows as f64
    };
    if actual > max_loss {
        return Err(FerricAlphaError::MaxLossExceeded {
            actual,
            allowed: max_loss,
        });
    }

    Ok(())
}
