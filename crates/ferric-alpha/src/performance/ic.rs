use std::collections::BTreeMap;

use polars::prelude::{DataFrame, NamedFrom, Series};

use crate::error::Result;
use crate::performance::frame::{
    bucket_date, date_bucket_column, date_column, finish_frame, grouped_indexes, mean,
    validate_factor_data,
};
use crate::performance::{IcOptions, MeanIcOptions};
use crate::spearman;

pub fn factor_information_coefficient(
    factor_data: &DataFrame,
    options: IcOptions,
) -> Result<DataFrame> {
    let data = validate_factor_data(
        factor_data,
        false,
        options.by_group || options.group_adjust,
        true,
    )?;

    let mut adjusted = vec![vec![None; data.periods.len()]; data.rows.len()];
    for (period_index, _) in data.periods.iter().enumerate() {
        if options.group_adjust {
            let groups = grouped_indexes(data.rows.iter().enumerate().map(|(index, row)| {
                (
                    index,
                    (row.date, row.group.clone().expect("validated group")),
                )
            }));
            for indexes in groups.values() {
                let values = indexes
                    .iter()
                    .filter_map(|index| data.rows[*index].returns[period_index])
                    .collect::<Vec<_>>();
                let Some(avg) = mean(&values) else { continue };
                for index in indexes {
                    adjusted[*index][period_index] =
                        data.rows[*index].returns[period_index].map(|value| value - avg);
                }
            }
        } else {
            for (index, row) in data.rows.iter().enumerate() {
                adjusted[index][period_index] = row.returns[period_index];
            }
        }
    }

    let groups = grouped_indexes(data.rows.iter().enumerate().map(|(index, row)| {
        let group = options
            .by_group
            .then(|| row.group.clone().expect("validated group"));
        (index, (row.date, group))
    }));

    let mut out_dates = Vec::new();
    let mut out_groups = Vec::new();
    let mut out_periods = Vec::new();
    let mut out_values = Vec::new();
    for ((date, group), indexes) in groups {
        for (period_index, period) in data.periods.iter().enumerate() {
            let mut factors = Vec::new();
            let mut returns = Vec::new();
            for index in &indexes {
                if let (Some(factor), Some(ret)) =
                    (data.rows[*index].factor, adjusted[*index][period_index])
                {
                    factors.push(factor);
                    returns.push(ret);
                }
            }
            out_dates.push(date);
            if options.by_group {
                out_groups.push(group.clone());
            }
            out_periods.push(period.label.clone());
            out_values.push(spearman(&factors, &returns)?);
        }
    }

    let mut columns = vec![
        date_column("date", out_dates, &data.date_dtype)?,
        Series::new("period".into(), out_periods).into(),
    ];
    if options.by_group {
        columns.insert(1, Series::new("group".into(), out_groups).into());
    }
    columns.push(Series::new("ic".into(), out_values).into());
    let sort = if options.by_group {
        vec!["date", "group", "period"]
    } else {
        vec!["date", "period"]
    };
    finish_frame(DataFrame::new(columns[0].len(), columns)?, &sort)
}

pub fn mean_information_coefficient(
    factor_data: &DataFrame,
    options: MeanIcOptions,
) -> Result<DataFrame> {
    let ic_options = IcOptions::default()
        .with_group_adjust(options.group_adjust)
        .with_by_group(options.by_group);
    let ic = factor_information_coefficient(factor_data, ic_options)?;
    let data = validate_factor_data(
        factor_data,
        false,
        options.by_group || options.group_adjust,
        true,
    )?;
    let dates = ic.column("date")?.as_materialized_series().datetime()?;
    let groups = ic
        .column("group")
        .ok()
        .map(|column| column.as_materialized_series().str())
        .transpose()?;
    let periods = ic.column("period")?.as_materialized_series().str()?;
    let values = ic.column("ic")?.as_materialized_series().f64()?;

    let mut grouped: BTreeMap<(Option<i32>, Option<String>, String), Vec<f64>> = BTreeMap::new();
    for index in 0..ic.height() {
        let Some(value) = values.get(index).filter(|value| value.is_finite()) else {
            continue;
        };
        let bucket = options
            .by_time
            .map(|bucket| {
                let date = dates
                    .physical()
                    .get(index)
                    .expect("validated date should not be null");
                bucket_date(date, &data.date_dtype, bucket)
            })
            .transpose()?;
        let group = groups
            .as_ref()
            .and_then(|groups| groups.get(index).map(ToOwned::to_owned));
        let period = periods
            .get(index)
            .expect("generated period should not be null")
            .to_owned();
        grouped
            .entry((bucket, group, period))
            .or_default()
            .push(value);
    }

    let mut out_buckets = Vec::new();
    let mut out_groups = Vec::new();
    let mut out_periods = Vec::new();
    let mut out_means = Vec::new();
    let mut out_counts = Vec::new();
    for ((bucket, group, period), values) in grouped {
        if options.by_time.is_some() {
            out_buckets.push(bucket);
        }
        if options.by_group {
            out_groups.push(group);
        }
        out_periods.push(period);
        out_means.push(mean(&values));
        out_counts.push(values.len() as u32);
    }

    let mut columns = Vec::new();
    let mut sort = Vec::new();
    if options.by_time.is_some() {
        columns.push(date_bucket_column("time_bucket", out_buckets));
        sort.push("time_bucket");
    }
    if options.by_group {
        columns.push(Series::new("group".into(), out_groups).into());
        sort.push("group");
    }
    columns.push(Series::new("period".into(), out_periods).into());
    columns.push(Series::new("mean_ic".into(), out_means).into());
    columns.push(Series::new("count".into(), out_counts).into());
    sort.push("period");
    finish_frame(DataFrame::new(columns[0].len(), columns)?, &sort)
}
