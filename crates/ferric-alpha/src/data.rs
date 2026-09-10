use std::collections::HashSet;

use polars::prelude::{Column, DataFrame, DataType, SortMultipleOptions};

use crate::error::{FerricAlphaError, Result};

pub fn validate_factor_frame(frame: &DataFrame) -> Result<DataFrame> {
    let date = required_column(frame, "date")?;
    let asset = required_column(frame, "asset")?;
    let factor = required_column(frame, "factor")?;

    ensure_dtype("date", date.dtype(), "Datetime", |dtype| {
        matches!(dtype, DataType::Datetime(_, _))
    })?;
    ensure_dtype("asset", asset.dtype(), "String", |dtype| {
        matches!(dtype, DataType::String)
    })?;
    ensure_dtype("factor", factor.dtype(), "Float64", |dtype| {
        matches!(dtype, DataType::Float64)
    })?;

    if let Ok(group) = frame.column("group") {
        ensure_dtype("group", group.dtype(), "String", |dtype| {
            matches!(dtype, DataType::String)
        })?;
    }

    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let mut keys = HashSet::with_capacity(frame.height());
    for (date, asset) in dates.physical().iter().zip(assets.iter()) {
        let (Some(date), Some(asset)) = (date, asset) else {
            return Err(FerricAlphaError::NullKey);
        };
        if !keys.insert((date, asset)) {
            return Err(FerricAlphaError::DuplicateKey);
        }
    }

    Ok(frame.sort(["date", "asset"], SortMultipleOptions::default())?)
}

pub(crate) fn required_column<'a>(frame: &'a DataFrame, name: &'static str) -> Result<&'a Column> {
    frame
        .column(name)
        .map_err(|_| FerricAlphaError::MissingColumn(name))
}

pub(crate) fn ensure_matching_date_timezone(
    left: &DataFrame,
    right: &DataFrame,
) -> Result<DataType> {
    let left_dtype = required_column(left, "date")?.dtype();
    let right_dtype = required_column(right, "date")?.dtype();
    match (left_dtype, right_dtype) {
        (DataType::Datetime(left_unit, left_zone), DataType::Datetime(right_unit, right_zone))
            if left_unit == right_unit && left_zone == right_zone =>
        {
            Ok(left_dtype.clone())
        }
        (DataType::Datetime(_, _), DataType::Datetime(_, _)) => {
            Err(FerricAlphaError::TimezoneMismatch)
        }
        _ => Err(FerricAlphaError::InvalidDtype {
            column: "date",
            expected: "Datetime",
            actual: left_dtype.to_string(),
        }),
    }
}

fn ensure_dtype(
    column: &'static str,
    actual: &DataType,
    expected: &'static str,
    predicate: impl FnOnce(&DataType) -> bool,
) -> Result<()> {
    if predicate(actual) {
        return Ok(());
    }

    Err(FerricAlphaError::InvalidDtype {
        column,
        expected,
        actual: actual.to_string(),
    })
}
