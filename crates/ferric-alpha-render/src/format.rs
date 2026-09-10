use chrono::{DateTime, TimeZone, Utc};
use ferric_alpha::{DisplayFormat, DisplayUnit, ReportColumnData, ReportTimeUnit};

use crate::{RenderError, RenderResult};

const EM_DASH: &str = "\u{2014}";

pub fn format_optional(value: Option<f64>, unit: DisplayUnit, decimals: u8) -> String {
    value.map_or_else(
        || EM_DASH.to_string(),
        |value| format_number(value, unit, decimals),
    )
}

pub fn format_number(value: f64, unit: DisplayUnit, decimals: u8) -> String {
    let decimals = usize::from(decimals);
    let scaled = normalize_negative_zero(match unit {
        DisplayUnit::Raw | DisplayUnit::Ratio | DisplayUnit::Count => value,
        DisplayUnit::DecimalReturn | DisplayUnit::Percent => value * 100.0,
        DisplayUnit::BasisPoints => value * 10_000.0,
    });

    match unit {
        DisplayUnit::DecimalReturn | DisplayUnit::Percent => format!("{scaled:.decimals$}%"),
        DisplayUnit::BasisPoints => format!("{scaled:.decimals$} bps"),
        DisplayUnit::Count if decimals == 0 => format!("{scaled:.0}"),
        _ => format!("{scaled:.decimals$}"),
    }
}

pub fn format_axis_value(value: f64, format: Option<&DisplayFormat>) -> String {
    match format {
        Some(format) => format_number(value, format.unit(), format.decimals()),
        None => format_number(value, DisplayUnit::Raw, 2),
    }
}

pub fn format_cell(data: &ReportColumnData, row: usize) -> RenderResult<String> {
    let out = match data {
        ReportColumnData::Boolean(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| EM_DASH.to_string(), |value| value.to_string()),
        ReportColumnData::Int32(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| EM_DASH.to_string(), |value| value.to_string()),
        ReportColumnData::Int64(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| EM_DASH.to_string(), |value| value.to_string()),
        ReportColumnData::UInt32(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| EM_DASH.to_string(), |value| value.to_string()),
        ReportColumnData::UInt64(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| EM_DASH.to_string(), |value| value.to_string()),
        ReportColumnData::Float64(values) => values.get(row).copied().flatten().map_or_else(
            || EM_DASH.to_string(),
            |value| format_number(value, DisplayUnit::Raw, 2),
        ),
        ReportColumnData::String(values) => values
            .get(row)
            .and_then(Option::as_ref)
            .map_or_else(|| EM_DASH.to_string(), ToString::to_string),
        ReportColumnData::Datetime {
            values,
            time_unit,
            timezone,
        } => values.get(row).copied().flatten().map_or_else(
            || Ok(EM_DASH.to_string()),
            |value| format_datetime(value, *time_unit, timezone.as_deref()),
        )?,
    };
    Ok(out)
}

pub fn format_datetime(
    value: i64,
    time_unit: ReportTimeUnit,
    timezone: Option<&str>,
) -> RenderResult<String> {
    let utc = timestamp_to_utc(value, time_unit)?;
    let date = match timezone {
        Some(name) => {
            let tz = name
                .parse::<chrono_tz::Tz>()
                .map_err(|_| RenderError::InvalidContract {
                    reason: format!("unknown timezone {name}"),
                })?;
            utc.with_timezone(&tz).date_naive()
        }
        None => utc.date_naive(),
    };
    Ok(date.format("%Y-%m-%d").to_string())
}

fn timestamp_to_utc(value: i64, time_unit: ReportTimeUnit) -> RenderResult<DateTime<Utc>> {
    let (divisor, nanos_multiplier): (i64, i64) = match time_unit {
        ReportTimeUnit::Nanoseconds => (1_000_000_000, 1),
        ReportTimeUnit::Microseconds => (1_000_000, 1_000),
        ReportTimeUnit::Milliseconds => (1_000, 1_000_000),
    };
    let seconds = value.div_euclid(divisor);
    let nanos = value
        .rem_euclid(divisor)
        .checked_mul(nanos_multiplier)
        .ok_or_else(|| RenderError::InvalidContract {
            reason: "timestamp overflow".to_string(),
        })?;
    let nanos = u32::try_from(nanos).map_err(|_| RenderError::InvalidContract {
        reason: "timestamp overflow".to_string(),
    })?;
    Utc.timestamp_opt(seconds, nanos)
        .single()
        .ok_or_else(|| RenderError::InvalidContract {
            reason: "timestamp overflow".to_string(),
        })
}

fn normalize_negative_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
